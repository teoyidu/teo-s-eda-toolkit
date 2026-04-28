import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_quality.processors.semhash_turkish_detector import SemHashTurkishDetector

def test_semhash_not_available():
    with patch('src.data_quality.processors.semhash_turkish_detector.SEMHASH_AVAILABLE', False):
        with pytest.raises(ImportError):
            processor = SemHashTurkishDetector({'text_column': 'text'})

@patch('src.data_quality.processors.semhash_turkish_detector.SemHash')
def test_semhash_process_action_group(mock_semhash):
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Apple']})
    processor = SemHashTurkishDetector({'text_column': 'text', 'action': 'group'})
    processor._find_duplicates = MagicMock(return_value=[[0, 2]])
    result = processor.process(df)
    assert 'duplicate_group' in result.columns

@patch('src.data_quality.processors.semhash_turkish_detector.SemHash')
def test_semhash_process_action_remove(mock_semhash):
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Apple']})
    processor = SemHashTurkishDetector({'text_column': 'text', 'action': 'remove'})
    processor._find_duplicates = MagicMock(return_value=[[0, 2]])
    result = processor.process(df)
    assert len(result) == 3

