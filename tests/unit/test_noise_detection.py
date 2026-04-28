import pandas as pd
from src.data_quality.processors.noise_detection import NoiseDetectionProcessor

def test_no_columns_configured(sample_df):
    processor = NoiseDetectionProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_alpha_ratio_column_added():
    df = pd.DataFrame({'text': ['hello', '12345', 'h3ll0']})
    processor = NoiseDetectionProcessor({'text_columns': {'text': {}}})
    result = processor.process(df)
    assert 'text_alpha_ratio' in result.columns
    assert result['text_alpha_ratio'].iloc[0] == 1.0
    assert result['text_alpha_ratio'].iloc[1] == 0.0
    assert result['text_alpha_ratio'].iloc[2] == 0.6

def test_bad_char_ratio_column_added():
    df = pd.DataFrame({'text': ['hello', 'hello@#$', 'test']})
    processor = NoiseDetectionProcessor({'text_columns': {'text': {}}})
    result = processor.process(df)
    assert 'text_bad_char_ratio' in result.columns
    assert result['text_bad_char_ratio'].iloc[0] == 0.0
    assert result['text_bad_char_ratio'].iloc[1] == 0.375

def test_filter_noise_remove():
    df = pd.DataFrame({'text': ['valid text', '1234567890', 'invalid@#$']})
    processor = NoiseDetectionProcessor({
        'text_columns': {
            'text': {
                'min_alpha_ratio': 0.5,
                'max_bad_char_ratio': 0.2,
                'action': 'remove',
                'filter_noise': True
            }
        }
    })
    result = processor.process(df)
    assert len(result) == 1
    assert result['text'].iloc[0] == 'valid text'

def test_empty_string_ratios():
    df = pd.DataFrame({'text': ['']})
    processor = NoiseDetectionProcessor({'text_columns': {'text': {}}})
    result = processor.process(df)
    assert result['text_alpha_ratio'].iloc[0] == 0.0
    assert result['text_bad_char_ratio'].iloc[0] == 0.0
