import pandas as pd
from src.data_quality.processors.mandatory_fields import MandatoryFieldsProcessor

def test_no_mandatory_fields(sample_df):
    processor = MandatoryFieldsProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_drops_rows_with_nulls(sample_df):
    processor = MandatoryFieldsProcessor({'mandatory_fields': ['value']})
    result = processor.process(sample_df)
    assert len(result) == 3
    assert result['value'].isna().sum() == 0

def test_multiple_mandatory_fields():
    df = pd.DataFrame({
        'field1': ['a', 'b', None, 'd'],
        'field2': [1, None, 3, 4]
    })
    processor = MandatoryFieldsProcessor({'mandatory_fields': ['field1', 'field2']})
    result = processor.process(df)
    assert len(result) == 2
    assert list(result.index) == [0, 3]

def test_all_valid_rows_kept():
    df = pd.DataFrame({'field1': ['a', 'b'], 'field2': [1, 2]})
    processor = MandatoryFieldsProcessor({'mandatory_fields': ['field1', 'field2']})
    result = processor.process(df)
    assert len(result) == 2
