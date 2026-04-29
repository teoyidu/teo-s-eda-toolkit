import logging
from typing import Dict, List, Tuple, Any
from pyspark.sql import DataFrame

from .base_processor import BaseProcessor
from ..processors import get_processor

logger = logging.getLogger(__name__)

class DataQualityPipeline:
    """
    Executes a sequence of Data Quality Processors.
    """
    def __init__(self, processors: List[Tuple[str, BaseProcessor]]):
        self.processors = processors
        
    def execute(self, df: DataFrame, metrics=None) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Execute all processors in sequence.
        
        Args:
            df: Input PySpark DataFrame
            metrics: Optional MetricsCollector instance
            
        Returns:
            Tuple of (final DataFrame, dictionary of stats per processor)
        """
        current_df = df
        validation_stats = {}
        
        current_count = None
        if metrics:
            current_count = current_df.count()
        
        for name, processor in self.processors:
            logger.info(f"Executing processor: {name}")
            
            start_time = None
            if metrics:
                start_time = metrics.start_timer(name)
                
            current_df, stats = processor.process(current_df)
            
            if metrics:
                metrics.end_timer(name, start_time)
                metrics.record_memory_usage(name)
                
                # Use stats to track counts without triggering Spark Action
                rows_dropped = stats.get('rows_dropped', 0)
                if current_count is not None:
                    current_count = max(0, current_count - rows_dropped)
                    metrics.record_record_count(name, current_count)
                    
                metrics.record_validation_stats(name, stats)
                
            validation_stats[name] = stats
            
        return current_df, validation_stats

class PipelineBuilder:
    """
    Builder pattern for constructing a DataQualityPipeline.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._processor_names: List[str] = []
        
    def add_processor(self, processor_name: str) -> 'PipelineBuilder':
        """
        Add a processor by its name to the pipeline.
        
        Args:
            processor_name: The registered name of the processor.
            
        Returns:
            self
        """
        self._processor_names.append(processor_name)
        return self
        
    def add_default_processors(self) -> 'PipelineBuilder':
        """
        Add the default set of processors in the standard order.
        """
        default_processors = [
            'MissingValuesProcessor',
            'MandatoryFieldsProcessor',
            'NumericalFormatsProcessor',
            'OutdatedDataProcessor',
            'ExternalValidationProcessor',
            'UniquenessProcessor',
            'CategoriesProcessor',
            'TextValidationProcessor',
            'RelationshipsProcessor',
            'EntryRulesProcessor',
            'TurkishDuplicateDetector'
        ]
        for p in default_processors:
            self.add_processor(p)
        return self

    def build(self) -> DataQualityPipeline:
        """
        Instantiate the pipeline with the configured processors.
        
        Returns:
            Configured DataQualityPipeline instance.
        """
        processors = []
        for name in self._processor_names:
            try:
                processor_cls = get_processor(name)
                processor_instance = processor_cls(self.config)
                processors.append((name, processor_instance))
            except Exception as e:
                logger.error(f"Failed to instantiate processor {name}: {str(e)}")
                raise
                
        return DataQualityPipeline(processors)
