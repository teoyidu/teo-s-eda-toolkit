import pandas as pd
from src.data_quality.processors.relationships import RelationshipsProcessor

def test_no_relationships(sample_df):
    processor = RelationshipsProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_foreign_key_remove_invalid():
    df = pd.DataFrame({'fk': [1, 2, 3, 4]})
    processor = RelationshipsProcessor({
        'relationships': [{
            'type': 'foreign_key',
            'source_column': 'fk',
            'target_column': 'ref_id',
            'target_values': [1, 2, 3],
            'action': 'remove'
        }]
    })
    result = processor.process(df)
    assert len(result) == 3
    assert list(result['fk']) == [1, 2, 3]

def test_parent_child_remove_orphans():
    df = pd.DataFrame({'parent': [1, 1, 2, 3], 'child': [1, 2, 3, 4]})
    processor = RelationshipsProcessor({
        'relationships': [{
            'type': 'parent_child',
            'parent_column': 'parent',
            'child_column': 'child',
            'action': 'remove'
        }]
    })
    result = processor.process(df)
    assert len(result) == 3
    assert list(result['child']) == [1, 2, 3]

def test_many_to_many_remove_invalid_combinations():
    df = pd.DataFrame({'src': [1, 1, 2], 'tgt': ['a', 'b', 'c']})
    processor = RelationshipsProcessor({
        'relationships': [{
            'type': 'many_to_many',
            'source_column': 'src',
            'target_column': 'tgt',
            'valid_combinations': [[1, 'a'], [2, 'c']],
            'action': 'remove'
        }]
    })
    result = processor.process(df)
    assert len(result) == 2
    assert list(result['src']) == [1, 2]
    assert list(result['tgt']) == ['a', 'c']

def test_missing_params_skipped():
    df = pd.DataFrame({'id': [1, 2]})
    processor = RelationshipsProcessor({
        'relationships': [{'type': 'foreign_key', 'source_column': 'id'}]
    })
    result = processor.process(df)
    assert len(result) == 2
