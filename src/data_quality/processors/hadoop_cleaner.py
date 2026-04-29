from ..core.base_processor import BaseProcessor
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf, regexp_extract, to_date, to_timestamp, regexp_replace
from pyspark.sql.types import StringType
import pandas as pd
import re
import json
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class HadoopCleanerProcessor(BaseProcessor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hadoop_columns = config.get('hadoop_columns', {})
        self.metadata_patterns = {
            'job_id': r'(job_\d+_\d+)',
            'task_id': r'(task_\d+_\d+_\d+)',
            'attempt_id': r'(attempt_\d+_\d+_\d+_\d+)',
            'container_id': r'(container_\d+_\d+_\d+_\d+)',
            'application_id': r'(application_\d+_\d+)',
            'executor_id': r'(executor_\d+)',
            'stage_id': r'(stage_\d+)',
            'partition_id': r'(partition_\d+)'
        }
        self.xml_tag_pattern = r'<[^>]+>'
        
    def process(self, df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Process the PySpark DataFrame to clean Hadoop tags
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            Tuple[DataFrame, Dict[str, Any]]: Processed DataFrame and stats
        """
        if not self.hadoop_columns:
            return df, {}
            
        stats = {}
        
        for column, settings in self.hadoop_columns.items():
            if column not in df.columns:
                continue
                
            if settings.get('extract_metadata', False):
                df = self._extract_metadata(df, column, settings)
                
            df = self._clean_hadoop_tags(df, column, settings)
            stats[column] = {"processed": True}
            
        return df, stats

    def _extract_metadata(self, df: DataFrame, column: str, settings: Dict[str, Any]) -> DataFrame:
        metadata_fields = settings.get('metadata_fields', list(self.metadata_patterns.keys()))
        for field in metadata_fields:
            if field in self.metadata_patterns:
                pattern = self.metadata_patterns[field]
                new_column = f"{column}_{field}"
                
                # Extract the 1st capture group
                df = df.withColumn(new_column, regexp_extract(col(column), pattern, 1))
                
                if 'metadata_transformations' in settings and field in settings['metadata_transformations']:
                    transform = settings['metadata_transformations'][field]
                    if transform.get('type') == 'datetime':
                        fmt = transform.get('format')
                        df = df.withColumn(new_column, to_timestamp(col(new_column), fmt))
                    elif transform.get('type') == 'numeric':
                        df = df.withColumn(new_column, col(new_column).cast("double"))
        return df

    def _clean_hadoop_tags(self, df: DataFrame, column: str, settings: Dict[str, Any]) -> DataFrame:
        # Wrap the python logic in a Pandas UDF for execution across partition batches
        
        metadata_patterns_vals = list(self.metadata_patterns.values()) if settings.get('remove_metadata', True) else []
        custom_patterns = settings.get('custom_patterns', [])
        preserve_structured_data = settings.get('preserve_structured_data', False)
        preserve_fields = settings.get('preserve_fields', [])
        xml_tag_pattern = self.xml_tag_pattern
        
        @pandas_udf(StringType())
        def clean_tags_udf(series: pd.Series) -> pd.Series:
            def extract_value_tags(text: str) -> str:
                if not text or pd.isna(text):
                    return ""
                try:
                    root = ET.fromstring(f'<root>{text}</root>')
                    values = [elem.text for elem in root.iter('value') if elem.text]
                    if values:
                        return ' '.join(values)
                    return ''.join(root.itertext()).strip()
                except Exception:
                    return re.sub(xml_tag_pattern, '', str(text))
                    
            def clean_text(text):
                try:
                    if not text or pd.isna(text): return ""
                    cleaned = extract_value_tags(str(text))
                    
                    for pattern in custom_patterns:
                        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
                        
                    for pattern in metadata_patterns_vals:
                        cleaned = re.sub(pattern, '', cleaned)
                        
                    if preserve_structured_data:
                        try:
                            t = cleaned.strip()
                            if t.startswith('{') or t.startswith('['):
                                data = json.loads(t)
                                if preserve_fields:
                                    data = {k: v for k, v in data.items() if k in preserve_fields}
                                cleaned = json.dumps(data)
                        except Exception:
                            pass
                            
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    return cleaned
                except Exception:
                    return str(text)
                    
            return series.apply(clean_text)
            
        df = df.withColumn(column, clean_tags_udf(col(column)))
        return df