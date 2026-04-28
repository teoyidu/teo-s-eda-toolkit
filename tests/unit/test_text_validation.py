import pandas as pd
from src.data_quality.processors.text_validation import TextValidationProcessor

def test_no_columns_configured(sample_df):
    processor = TextValidationProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_strip_and_collapse_whitespace():
    df = pd.DataFrame({'text': ['  hello   world  ', 'test']})
    processor = TextValidationProcessor({'text_columns': {'text': {}}})
    result = processor.process(df)
    assert list(result['text']) == ['hello world', 'test']

def test_case_lower_upper_title():
    df = pd.DataFrame({'text': ['hello world', 'HELLO WORLD', 'hello world']})
    processor = TextValidationProcessor({
        'text_columns': {
            'text': {'case': 'upper'}
        }
    })
    df['text1'] = ['hello world', 'TEST', 'abc']
    df['text2'] = ['hello world', 'TEST', 'def']
    df['text3'] = ['hello world', 'TEST', 'ghi']
    processor = TextValidationProcessor({
        'text_columns': {
            'text1': {'case': 'upper'},
            'text2': {'case': 'lower'},
            'text3': {'case': 'title'},
        }
    })
    result = processor.process(df)
    assert list(result['text1']) == ['HELLO WORLD', 'TEST', 'ABC']
    assert list(result['text2']) == ['hello world', 'test', 'def']
    assert list(result['text3']) == ['Hello World', 'Test', 'Ghi']

def test_remove_special_chars():
    df = pd.DataFrame({'text': ['hello!@# world$', 'test']})
    processor = TextValidationProcessor({'text_columns': {'text': {'remove_special_chars': True}}})
    result = processor.process(df)
    assert list(result['text']) == ['hello world', 'test']

def test_min_length_remove():
    df = pd.DataFrame({'text': ['a', 'abc', 'abcd']})
    processor = TextValidationProcessor({
        'text_columns': {'text': {'min_length': 3, 'action': 'remove'}}
    })
    result = processor.process(df)
    assert len(result) == 2
    assert list(result['text']) == ['abc', 'abcd']

def test_max_length_remove():
    df = pd.DataFrame({'text': ['a', 'abc', 'abcd']})
    processor = TextValidationProcessor({
        'text_columns': {'text': {'max_length': 3, 'action': 'remove'}}
    })
    result = processor.process(df)
    assert len(result) == 2
    assert list(result['text']) == ['a', 'abc']

def test_regex_pattern_remove():
    df = pd.DataFrame({'text': ['123', 'abc', '456']})
    processor = TextValidationProcessor({
        'text_columns': {'text': {'pattern': r'^\d+$', 'action': 'remove'}}
    })
    result = processor.process(df)
    assert len(result) == 2
    assert list(result['text']) == ['123', '456']
