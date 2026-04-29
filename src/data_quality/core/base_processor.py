from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any
from pyspark.sql import DataFrame

class BaseProcessor(ABC):
    """
    Abstract Base Class for all Data Quality Processors.
    Enforces a strict contract for data processing and configuration validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the processor with the given configuration.
        
        Args:
            config: Configuration dictionary (can be parsed from Pydantic schema)
        """
        self.config = config

    @abstractmethod
    def process(self, df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Process the given PySpark DataFrame and return the cleaned DataFrame
        along with validation statistics.
        
        Args:
            df: Input PySpark DataFrame
            
        Returns:
            Tuple containing:
            - DataFrame: Processed/Cleaned PySpark DataFrame
            - Dict: Statistics about the processing (e.g. rows dropped, missing values)
        """
        pass
        
    def validate_config(self) -> List[str]:
        """
        Validate the configuration specific to this processor.
        
        Returns:
            List of error messages. Empty list if configuration is valid.
        """
        return []
