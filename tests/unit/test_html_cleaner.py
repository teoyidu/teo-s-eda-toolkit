import pandas as pd
from src.data_quality.processors.html_cleaner import HTMLCleanerProcessor

def test_no_columns_configured(sample_df):
    processor = HTMLCleanerProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_strips_non_whitelisted_tags():
    df = pd.DataFrame({'html': ['<div><p>Hello</p><span>World</span></div>']})
    processor = HTMLCleanerProcessor({'html_columns': {'html': {}}})
    result = processor.process(df)
    assert '<p>Hello</p>World' in list(result['html'])[0] or '<p>Hello</p> World' in list(result['html'])[0]

def test_preserves_whitelisted_tags():
    df = pd.DataFrame({'html': ['<p><b>Hello</b></p>']})
    processor = HTMLCleanerProcessor({'html_columns': {'html': {'whitelist_tags': ['p', 'b']}}})
    result = processor.process(df)
    assert list(result['html']) == ['<p><b>Hello</b></p>']

def test_custom_tag_replacement():
    df = pd.DataFrame({'html': ['<img src="test.jpg"/> Hello']})
    processor = HTMLCleanerProcessor({
        'html_columns': {
            'html': {
                'custom_tag_replacements': {'img': '[IMAGE]'},
                'whitelist_tags': []
            }
        }
    })
    result = processor.process(df)
    assert list(result['html']) == ['[IMAGE] Hello']

def test_entity_handling():
    df = pd.DataFrame({'html': ['Hello &amp; World &nbsp;']})
    processor = HTMLCleanerProcessor({'html_columns': {'html': {'handle_entities': True}}})
    result = processor.process(df)
    assert list(result['html']) == ['Hello & World']

def test_custom_transformations_replace_and_extract():
    df = pd.DataFrame({'html': ['Contact: admin@test.com']})
    processor = HTMLCleanerProcessor({
        'html_columns': {
            'html': {
                'custom_transformations': [
                    {'type': 'replace', 'pattern': r'admin@\w+\.com', 'replacement': '[EMAIL]'},
                    {'type': 'extract', 'pattern': r'Contact:\s*(.*)'}
                ],
                'whitelist_tags': []
            }
        }
    })
    result = processor.process(df)
    assert list(result['html']) == ['[EMAIL]']

def test_error_returns_original_text():
    # Pass a non-string object that fails beautiful soup parsing
    df = pd.DataFrame({'html': [123]})
    processor = HTMLCleanerProcessor({'html_columns': {'html': {'custom_transformations': [{'type': 'extract', 'pattern': '('}]}}}) # Invalid regex to trigger exception
    result = processor.process(df)
    # The processor converts to string at the beginning: df[column] = df[column].astype(str)
    # But invalid regex will cause exception in apply
    assert list(result['html']) == ['123']
