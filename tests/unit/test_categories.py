import pandas as pd
from src.data_quality.processors.categories import CategoriesProcessor

def test_no_columns_configured(sample_df):
    processor = CategoriesProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_case_insensitive_lowercasing():
    df = pd.DataFrame({'cat': ['A', 'b', 'C']})
    processor = CategoriesProcessor({
        'category_columns': {'cat': {'case_sensitive': False}}
    })
    result = processor.process(df)
    assert list(result['cat']) == ['a', 'b', 'c']

def test_value_mapping():
    df = pd.DataFrame({'cat': ['yes', 'no', 'unknown']})
    processor = CategoriesProcessor({
        'category_columns': {
            'cat': {'value_mapping': {'yes': 1, 'no': 0}, 'default_value': -1}
        }
    })
    result = processor.process(df)
    assert list(result['cat']) == [1.0, 0.0, -1.0]

def test_allowed_values_default():
    df = pd.DataFrame({'cat': ['A', 'B', 'C']})
    processor = CategoriesProcessor({'category_columns': {'cat': {'allowed_values': ['a', 'b']}}})
    result = processor.process(df)
    assert list(result['cat']) == ['a', 'b', 'unknown']

def test_allowed_values_remove():
    df = pd.DataFrame({'cat': ['A', 'B', 'C']})
    processor = CategoriesProcessor({
        'category_columns': {
            'cat': {'allowed_values': ['a', 'b'], 'action': 'remove'}
        }
    })
    result = processor.process(df)
    assert len(result) == 2
    assert list(result['cat']) == ['a', 'b']
