import pandas as pd
from src.data_quality.processors.uniqueness import UniquenessProcessor

def test_no_constraints(sample_df):
    processor = UniquenessProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_drop_duplicates_first():
    df = pd.DataFrame({'id': [1, 1, 2], 'val': ['a', 'b', 'c']})
    processor = UniquenessProcessor({
        'unique_constraints': [{'columns': ['id'], 'action': 'drop_duplicates'}]
    })
    result = processor.process(df)
    assert len(result) == 2
    assert result['val'].iloc[0] == 'a'

def test_keep_last():
    df = pd.DataFrame({'id': [1, 1, 2], 'val': ['a', 'b', 'c']})
    processor = UniquenessProcessor({
        'unique_constraints': [{'columns': ['id'], 'action': 'keep_last'}]
    })
    result = processor.process(df)
    assert len(result) == 2
    assert result['val'].iloc[0] == 'b'

def test_mark_action():
    df = pd.DataFrame({'id': [1, 1, 2], 'val': ['a', 'b', 'c']})
    processor = UniquenessProcessor({
        'unique_constraints': [{'columns': ['id'], 'action': 'mark'}]
    })
    result = processor.process(df)
    assert len(result) == 3
    assert 'is_duplicate' in result.columns
    assert list(result['is_duplicate']) == [False, True, False]

def test_empty_columns_list_skipped(sample_df):
    processor = UniquenessProcessor({
        'unique_constraints': [{'columns': []}]
    })
    result = processor.process(sample_df)
    assert len(result) == 4
