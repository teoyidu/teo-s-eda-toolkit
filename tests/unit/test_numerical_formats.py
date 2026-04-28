import pandas as pd
import numpy as np
from src.data_quality.processors.numerical_formats import NumericalFormatsProcessor

def test_no_columns_configured(sample_df):
    processor = NumericalFormatsProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_converts_to_numeric():
    df = pd.DataFrame({
        'num_col': ['1.5', '2', 'invalid', '4.5']
    })
    processor = NumericalFormatsProcessor({
        'numerical_columns': ['num_col']
    })
    result = processor.process(df)
    assert pd.isna(result['num_col'].iloc[2])
    assert result['num_col'].iloc[0] == 1.5

def test_rounding():
    df = pd.DataFrame({'num_col': [1.2345, 2.3456]})
    processor = NumericalFormatsProcessor({
        'numerical_columns': ['num_col'],
        'decimal_places': 2
    })
    result = processor.process(df)
    assert result['num_col'].iloc[0] == 1.23
    assert result['num_col'].iloc[1] == 2.35

def test_handle_outliers_iqr():
    df = pd.DataFrame({'num_col': [10, 12, 11, 100, 10, 13]})
    processor = NumericalFormatsProcessor({
        'numerical_columns': ['num_col'],
        'handle_outliers': True
    })
    result = processor.process(df)
    # The outlier (100) should be clipped
    assert result['num_col'].iloc[3] < 100

def test_missing_column_skipped(sample_df):
    processor = NumericalFormatsProcessor({
        'numerical_columns': ['non_existent']
    })
    result = processor.process(sample_df)
    assert 'non_existent' not in result.columns
