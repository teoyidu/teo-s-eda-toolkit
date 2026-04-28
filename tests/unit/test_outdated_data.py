import pandas as pd
from datetime import datetime, timedelta
from src.data_quality.processors.outdated_data import OutdatedDataProcessor

def test_no_date_columns(sample_df):
    processor = OutdatedDataProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_removes_old_rows():
    now = datetime.now()
    old_date = now - timedelta(days=400)
    recent_date = now - timedelta(days=100)
    
    df = pd.DataFrame({
        'date_col': [old_date, recent_date, old_date]
    })
    processor = OutdatedDataProcessor({
        'date_columns': ['date_col'],
        'data_retention_days': 365
    })
    result = processor.process(df)
    assert len(result) == 1
    assert result['date_col'].iloc[0] == recent_date

def test_keeps_recent_rows():
    now = datetime.now()
    df = pd.DataFrame({
        'date_col': [now, now - timedelta(days=10)]
    })
    processor = OutdatedDataProcessor({
        'date_columns': ['date_col'],
        'data_retention_days': 30
    })
    result = processor.process(df)
    assert len(result) == 2

def test_missing_column_skipped(sample_df):
    processor = OutdatedDataProcessor({
        'date_columns': ['non_existent']
    })
    result = processor.process(sample_df)
    assert len(result) == 4
