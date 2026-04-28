import pytest
from src.data_quality.processors.missing_values import MissingValuesProcessor
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

@pytest.fixture
def spark_df(spark):
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("value1", DoubleType(), True),
        StructField("value2", StringType(), True)
    ])
    data = [
        (1, 10.5, "a"),
        (2, None, "b"),
        (3, 20.0, None),
        (4, None, None)
    ]
    return spark.createDataFrame(data, schema)

def test_drop_strategy_critical_columns(spark, spark_df):
    config = {
        'missing_value_strategy': 'drop',
        'critical_columns': ['value1']
    }
    processor = MissingValuesProcessor(config)
    cleaned_df, stats = processor.process(spark_df)
    
    assert cleaned_df.count() == 2
    assert stats['rows_dropped'] == 2

def test_drop_strategy_all_columns(spark, spark_df):
    config = {
        'missing_value_strategy': 'drop'
    }
    processor = MissingValuesProcessor(config)
    cleaned_df, stats = processor.process(spark_df)
    
    assert cleaned_df.count() == 1
    assert stats['rows_dropped'] == 3

def test_fill_strategy(spark, spark_df):
    config = {
        'missing_value_strategy': 'fill',
        'fill_values': {'value1': 0.0, 'value2': 'unknown'}
    }
    processor = MissingValuesProcessor(config)
    cleaned_df, stats = processor.process(spark_df)
    
    assert cleaned_df.count() == 4
    assert stats['rows_dropped'] == 0
    
    # Verify filled values
    filled_data = cleaned_df.collect()
    assert filled_data[1]['value1'] == 0.0
    assert filled_data[2]['value2'] == 'unknown'

def test_passthrough_strategy(spark, spark_df):
    config = {
        'missing_value_strategy': 'none'
    }
    processor = MissingValuesProcessor(config)
    cleaned_df, stats = processor.process(spark_df)
    
    assert cleaned_df.count() == 4
    assert stats['rows_dropped'] == 0

def test_stats_dict_keys(spark, spark_df):
    config = {
        'missing_value_strategy': 'drop'
    }
    processor = MissingValuesProcessor(config)
    _, stats = processor.process(spark_df)
    
    assert 'value1' in stats
    assert 'value2' in stats
    assert stats['value1']['missing_count'] == 2
    assert stats['value2']['missing_count'] == 2

def test_empty_dataframe(spark):
    schema = StructType([
        StructField("id", IntegerType(), True),
    ])
    empty_df = spark.createDataFrame([], schema)
    
    config = {'missing_value_strategy': 'drop'}
    processor = MissingValuesProcessor(config)
    cleaned_df, stats = processor.process(empty_df)
    
    assert cleaned_df.count() == 0
    assert stats['rows_dropped'] == 0
