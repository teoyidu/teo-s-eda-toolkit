from ..core.base_processor import BaseProcessor
"""
Processor for cleaning HTML tags from text data
"""

import logging
from typing import List, Dict, Any, Set, Optional, Union, cast, Tuple
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import re
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType

logger = logging.getLogger(__name__)

class HTMLCleanerProcessor(BaseProcessor):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the processor
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        super().__init__(config)
        self.html_columns = config.get('html_columns', {})
        
        # Default whitelist of tags to preserve
        self.default_whitelist = {'p', 'br', 'strong', 'em', 'b', 'i', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
        
    def process(self, df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Process the PySpark DataFrame to clean HTML tags
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            
        Returns:
            Tuple[DataFrame, Dict]: Processed DataFrame and stats
        """
        if not self.html_columns:
            return df, {}
            
        stats = {}
            
        for column, settings in self.html_columns.items():
            if column not in df.columns:
                continue
            
            # Apply HTML cleaning
            df = self._clean_html(df, column, settings)
            stats[column] = {"processed": True}
            
        return df, stats
        
    def _clean_html(self, df: DataFrame, column: str, settings: Dict[str, Any]) -> DataFrame:
        """Clean HTML tags from text data using pandas_udf"""
        
        # Extract variables so they can be captured by the UDF
        whitelist = set(settings.get('whitelist_tags', self.default_whitelist))
        custom_replacements = settings.get('custom_tag_replacements', {})
        preserve_attributes = settings.get('preserve_attributes', False)
        allowed_attributes = settings.get('allowed_attributes', {})
        handle_entities = settings.get('handle_entities', True)
        entity_mapping = settings.get('custom_entities', {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&apos;': "'"
        })
        custom_transformations = settings.get('custom_transformations', [])
        
        @pandas_udf(StringType())
        def clean_html_udf(series: pd.Series) -> pd.Series:
            def clean_text(text: str) -> str:
                if not text or pd.isna(text):
                    return ""
                try:
                    # Parse HTML
                    soup = BeautifulSoup(str(text), 'html.parser')
                    
                    # Handle custom tag replacements
                    for tag, replacement in custom_replacements.items():
                        for element in soup.find_all(tag):
                            if isinstance(element, Tag):
                                element.replace_with(replacement)
                    
                    # Remove non-whitelisted tags but keep their content
                    for tag in soup.find_all():
                        if isinstance(tag, Tag) and tag.name not in whitelist:
                            tag.unwrap()
                    
                    # Handle specific attributes if needed
                    if preserve_attributes:
                        for tag in soup.find_all():
                            if isinstance(tag, Tag) and tag.name in whitelist:
                                # Keep specified attributes
                                allowed_attrs = allowed_attributes.get(tag.name, [])
                                attrs_to_remove = [attr for attr in tag.attrs if attr not in allowed_attrs]
                                for attr in attrs_to_remove:
                                    del tag[attr]
                    
                    # Get text content with preserved formatting
                    cleaned_text = str(soup)
                    
                    # Handle specific HTML entities if needed
                    if handle_entities:
                        for entity, replacement in entity_mapping.items():
                            cleaned_text = cleaned_text.replace(entity, replacement)
                    
                    # Remove extra whitespace
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                    
                    # Apply custom transformations if specified
                    for transform in custom_transformations:
                        if transform['type'] == 'replace':
                            cleaned_text = re.sub(transform['pattern'], transform['replacement'], cleaned_text)
                        elif transform['type'] == 'extract':
                            match = re.search(transform['pattern'], cleaned_text)
                            if match:
                                cleaned_text = match.group(1)
                    
                    return cleaned_text
                except Exception as e:
                    logger.warning(f"Error cleaning HTML: {str(e)}")
                    return str(text)
                    
            return series.apply(clean_text)
            
        # Apply cleaning to the column
        df = df.withColumn(column, clean_html_udf(col(column)))
        return df 