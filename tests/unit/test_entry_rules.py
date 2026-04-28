import pandas as pd
from src.data_quality.processors.entry_rules import EntryRulesProcessor

def test_no_rules(sample_df):
    processor = EntryRulesProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_conditional_set_value():
    df = pd.DataFrame({'col': [1, 2, 3]})
    processor = EntryRulesProcessor({
        'entry_rules': [{
            'type': 'conditional',
            'condition': 'col > 1',
            'target_column': 'flag',
            'action': {'type': 'set_value', 'value': 'high'}
        }]
    })
    df['flag'] = 'low'
    result = processor.process(df)
    assert list(result['flag']) == ['low', 'high', 'high']

def test_conditional_remove():
    df = pd.DataFrame({'col': [1, 2, 3]})
    processor = EntryRulesProcessor({
        'entry_rules': [{
            'type': 'conditional',
            'condition': 'col > 2',
            'target_column': 'col',  # Needed for validation in code
            'action': {'type': 'remove'}
        }]
    })
    result = processor.process(df)
    assert len(result) == 2
    assert list(result['col']) == [1, 2]

def test_derived_column():
    df = pd.DataFrame({'price': [10, 20], 'qty': [2, 3]})
    processor = EntryRulesProcessor({
        'entry_rules': [{
            'type': 'derived',
            'target_column': 'total',
            'expression': 'price * qty'
        }]
    })
    result = processor.process(df)
    assert list(result['total']) == [20, 60]

def test_validation_remove():
    df = pd.DataFrame({'col': [1, 2, 3]})
    processor = EntryRulesProcessor({
        'entry_rules': [{
            'type': 'validation',
            'condition': 'col < 3',
            'action': {'type': 'remove'}
        }]
    })
    result = processor.process(df)
    assert len(result) == 2
    assert list(result['col']) == [1, 2]

def test_validation_set_value():
    df = pd.DataFrame({'col': [1, 2, 3], 'flag': ['a', 'b', 'c']})
    processor = EntryRulesProcessor({
        'entry_rules': [{
            'type': 'validation',
            'condition': 'col < 3',
            'action': {'type': 'set_value', 'target_column': 'flag', 'value': 'INVALID'}
        }]
    })
    result = processor.process(df)
    assert list(result['flag']) == ['a', 'b', 'INVALID']

def test_missing_params_logged_but_no_crash(sample_df):
    processor = EntryRulesProcessor({
        'entry_rules': [{'type': 'conditional', 'condition': 'col > 1'}]
    })
    result = processor.process(sample_df)
    assert len(result) == 4
