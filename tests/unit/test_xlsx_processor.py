import pandas as pd
import numpy as np
from src.data_quality.processors.xlsx_processor import XLSXProcessor

def test_clean_data_fill_strategy():
    config = {'missing_value_strategy': 'fill', 'fill_values': {'A': 0, 'B': 0}}
    processor = XLSXProcessor(config)
    df = pd.DataFrame({'A': [1, np.nan], 'B': [np.nan, 2]})
    result = processor._clean_data(df)
    assert result.isna().sum().sum() == 0
    assert result['A'].iloc[1] == 0

def test_clean_data_drop_strategy():
    config = {'missing_value_strategy': 'drop', 'mandatory_fields': ['A', 'B']}
    processor = XLSXProcessor(config)
    df = pd.DataFrame({'A': [1, np.nan], 'B': [np.nan, 2]})
    result = processor._clean_data(df)
    assert len(result) == 0

def test_apply_filters_min_max_allowed_regex():
    config = {
        'filters': {
            'A': {'min_value': 2, 'max_value': 5},
            'cat': {'allowed_values': ['A', 'B']},
            'text': {'regex_pattern': r'^[a-z_]+$'}
        }
    }
    processor = XLSXProcessor(config)
    df = pd.DataFrame({
        'A': [1, 3, 6],
        'cat': ['A', 'C', 'B'],
        'text': ['valid_text', 'INVALID', 'also_valid']
    })
    result = processor._apply_filters(df)
    assert len(result) == 0  # no row matches all rules
    
    # Check individual
    df2 = pd.DataFrame({'A': [3]})
    assert len(processor._apply_filters(df2)) == 1
    
    df3 = pd.DataFrame({'A': [10]})
    assert len(processor._apply_filters(df3)) == 0

def test_deduplicate_first_last():
    config = {
        'unique_constraints': [
            {'columns': ['id'], 'action': 'drop_duplicates'}
        ]
    }
    processor = XLSXProcessor(config)
    df = pd.DataFrame({'id': [1, 1, 2], 'val': ['a', 'b', 'c']})
    res_first = processor._deduplicate(df)
    assert len(res_first) == 2
    assert res_first['val'].iloc[0] == 'a'
    
    config['unique_constraints'][0]['action'] = 'keep_last'
    res_last = processor._deduplicate(df)
    assert res_last['val'].iloc[0] == 'b'

def test_decontaminate_strip_lowercase():
    config = {'case_sensitive': False}
    processor = XLSXProcessor(config)
    df = pd.DataFrame({'text': [' Hello ', 'WORLD']})
    res = processor._decontaminate(df)
    assert list(res['text']) == ['hello', 'world']

def test_process_file_roundtrip(tmp_path):
    df = pd.DataFrame({'A': [1, 2, 3]})
    file_path = tmp_path / "test.xlsx"
    df.to_excel(file_path, index=False)
    
    processor = XLSXProcessor({})
    result = processor.process_file(str(file_path))
    assert len(result) == 3

def test_save_processed_file(tmp_path):
    df = pd.DataFrame({'A': [1]})
    processor = XLSXProcessor({})
    out_path = tmp_path / "out.xlsx"
    processor.save_processed_file(df, str(out_path))
    assert out_path.exists()
    
    loaded = pd.read_excel(out_path)
    assert len(loaded) == 1
    assert loaded['A'].iloc[0] == 1
