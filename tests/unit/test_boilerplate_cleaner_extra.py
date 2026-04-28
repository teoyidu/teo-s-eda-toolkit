import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_quality.processors.boilerplate_cleaner import TurkishBoilerplateCleanerProcessor

def test_boilerplate_cleaner_edge_cases():
    # Test debug mode and no columns
    processor = TurkishBoilerplateCleanerProcessor({'debug_mode': True})
    df = pd.DataFrame({'other_col': ['test']})
    result = processor.process(df)
    assert len(result) == 1
    
    # Test column not in df
    processor = TurkishBoilerplateCleanerProcessor({'boilerplate_columns': {'missing_col': {}}})
    result = processor.process(df)
    assert len(result) == 1

def test_boilerplate_cleaner_remove_header_footer_complex():
    df = pd.DataFrame({'text': ['Tarih: 01.01.2023\n\nSome main content here.\n\nİmza: John Doe\nEk 1: Document']})
    processor = TurkishBoilerplateCleanerProcessor({
        'boilerplate_columns': {'text': {'remove_header_footer': True}}
    })
    result = processor.process(df)
    assert 'Some main content here.' in result['text'].iloc[0]

def test_boilerplate_cleaner_remove_duplicates_simple():
    df = pd.DataFrame({'text': ['hello world\nhello world\nunique string test']})
    processor = TurkishBoilerplateCleanerProcessor({
        'boilerplate_columns': {'text': {'remove_duplicates': True, 'remove_duplicates_tfidf': False}}
    })
    result = processor.process(df)
    # Simple duplicate removal uses SequenceMatcher
    assert 'unique string test' in result['text'].iloc[0]

