"""
Legal Domain Filtering Processor
Uses BERTurk-Legal model to identify legal domain content
"""

import logging
import os
from typing import Dict, Tuple, Any, Iterator
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import BooleanType, FloatType, StructType, StructField
import pandas as pd


logger = logging.getLogger(__name__)

class LegalDomainFilter:
    """Processor for filtering legal domain content using BERTurk-Legal model"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BERTurk-Legal model configuration
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - model_name: Name of the model to use (default: KocLab-Bilkent/BERTurk-Legal)
                - cache_dir: Directory to cache the model (default: ./model_cache)
                - threshold: Probability threshold for legal domain classification (default: 0.5)
                - batch_size: Batch size for processing (default: 32)
                - device: Device to use for inference (default: auto)
        """
        self.config = config
        self.model_name = config.get('model_name', 'KocLab-Bilkent/BERTurk-Legal')
        self.cache_dir = config.get('cache_dir', './model_cache')
        self.threshold = config.get('threshold', 0.5)
        self.batch_size = config.get('batch_size', 32)
        
        # Determine device
        import torch
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create cache directory if it doesn't exist on driver
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Legal domain keywords in Turkish
        self.legal_keywords = [
            "Yönetmelik", "Kanun", "Madde", "Fıkra", "Resmî Gazete",
            "Tüzük", "Kararname", "Genelge", "Tebliğ", "Yönerge",
            "Talimat", "Sözleşme", "Dilekçe", "İstinaf", "Temyiz",
            "Dava", "Mahkeme", "Savcı", "Hakim", "Avukat"
        ]
        
    def process(self, df: DataFrame, text_column: str) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Process the DataFrame to filter legal domain content
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            text_column (str): Name of the column containing text to analyze
            
        Returns:
            Tuple[DataFrame, Dict[str, Any]]: Processed DataFrame and statistics
        """
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        
        schema = StructType([
            StructField("is_legal_domain", BooleanType(), True),
            StructField("legal_probability", FloatType(), True)
        ])
        
        model_name = self.model_name
        cache_dir = self.cache_dir
        threshold = self.threshold
        legal_keywords = self.legal_keywords
        device_pref = self.device
        
        @pandas_udf(schema)
        def predict_legal_domain(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            # Lazy initialize model inside the executor Native Python Environment
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            os.makedirs(cache_dir, exist_ok=True)
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
                device = device_pref if torch.cuda.is_available() else 'cpu'
                model.to(device)
                model.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")
                
            for series in iterator:
                results = []
                # Simple loop implementation, ideally we'd batch to the model for speed
                # but we'll keep the logic simple for now.
                for text in series:
                    if not text or not isinstance(text, str):
                        results.append({'is_legal_domain': False, 'legal_probability': 0.0})
                        continue
                        
                    text_lower = text.lower()
                    if any(keyword.lower() in text_lower for keyword in legal_keywords):
                        results.append({'is_legal_domain': True, 'legal_probability': 1.0})
                        continue
                        
                    try:
                        inputs = tokenizer(
                            text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512,
                            padding=True
                        ).to(device)
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            predictions = torch.softmax(outputs.logits, dim=1)
                            legal_prob = predictions[0][1].item()
                            
                            results.append({
                                'is_legal_domain': bool(legal_prob > threshold),
                                'legal_probability': float(legal_prob)
                            })
                    except Exception:
                        results.append({'is_legal_domain': False, 'legal_probability': 0.0})
                        
                yield pd.DataFrame(results)
        
        # Add legal domain flag and probability columns using Iterator pandas_udf
        df = df.withColumn("legal_domain_struct", predict_legal_domain(col(text_column)))
        df = df.withColumn("is_legal_domain", col("legal_domain_struct.is_legal_domain"))
        df = df.withColumn("legal_probability", col("legal_domain_struct.legal_probability"))
        df = df.drop("legal_domain_struct")
        
        # Compute total and legal-domain counts in a single Spark action
        # instead of two separate .count() calls (halves the number of jobs).
        from pyspark.sql.functions import count as spark_count, when as spark_when, lit
        count_row = df.agg(
            spark_count(lit(1)).alias("total"),
            spark_count(spark_when(col("is_legal_domain"), lit(1))).alias("legal"),
        ).collect()[0]
        total_count = count_row["total"]
        legal_count = count_row["legal"]

        stats = {
            "total_documents": total_count,
            "legal_documents": legal_count,
            "legal_percentage": (legal_count / total_count * 100) if total_count > 0 else 0,
            "model_name": self.model_name,
            "threshold": self.threshold,
            "device": self.device
        }
        
        logger.info(
            f"Legal domain filtering completed. Found {legal_count} legal documents "
            f"out of {total_count} total documents."
        )
        
        return df, stats