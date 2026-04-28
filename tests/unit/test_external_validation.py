import pandas as pd
from src.data_quality.processors.external_validation import ExternalValidationProcessor

def test_no_rules(sample_df):
    processor = ExternalValidationProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_regex_rule_warn_and_remove():
    df = pd.DataFrame({'email': ['test@test.com', 'invalid', 'a@b.c']})
    processor = ExternalValidationProcessor({
        'validation_rules': {
            'email': [{
                'type': 'regex',
                'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$',
                'action': 'remove'
            }]
        }
    })
    result = processor.process(df)
    assert len(result) == 2
    assert 'invalid' not in list(result['email'])

def test_lookup_rule_warn_and_remove():
    df = pd.DataFrame({'status': ['active', 'pending', 'invalid']})
    processor = ExternalValidationProcessor({
        'validation_rules': {
            'status': [{
                'type': 'lookup',
                'valid_values': ['active', 'pending'],
                'action': 'remove'
            }]
        }
    })
    result = processor.process(df)
    assert len(result) == 2
    assert 'invalid' not in list(result['status'])

def test_range_rule_min_max():
    df = pd.DataFrame({'age': [10, 20, 50, 100]})
    processor = ExternalValidationProcessor({
        'validation_rules': {
            'age': [{
                'type': 'range',
                'min_value': 18,
                'max_value': 60
            }]
        }
    })
    result = processor.process(df)
    assert len(result) == 2
    assert list(result['age']) == [20, 50]

def test_missing_column_skipped(sample_df):
    processor = ExternalValidationProcessor({
        'validation_rules': {'non_existent': [{'type': 'regex', 'pattern': '.*'}]}
    })
    result = processor.process(sample_df)
    assert len(result) == 4
