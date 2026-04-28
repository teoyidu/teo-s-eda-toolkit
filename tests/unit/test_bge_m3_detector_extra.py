import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_quality.processors.bge_m3_turkish_detector import BGEM3TurkishDetector

def test_bge_m3_not_available():
    with patch('src.data_quality.processors.bge_m3_turkish_detector.BGE_M3_AVAILABLE', False):
        with pytest.raises(ImportError):
            processor = BGEM3TurkishDetector({'text_column': 'text'})

@patch('src.data_quality.processors.bge_m3_turkish_detector.BGEM3FlagModel')
def test_bge_m3_process_action_group(mock_model):
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Apple']})
    processor = BGEM3TurkishDetector({'text_column': 'text', 'action': 'group'})
    processor._find_duplicates = MagicMock(return_value=[[0, 2]])
    result = processor.process(df)
    assert 'duplicate_group' in result.columns

@patch('src.data_quality.processors.bge_m3_turkish_detector.BGEM3FlagModel')
def test_bge_m3_process_action_remove(mock_model):
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Apple']})
    processor = BGEM3TurkishDetector({'text_column': 'text', 'action': 'remove'})
    processor._find_duplicates = MagicMock(return_value=[[0, 2]])
    result = processor.process(df)
    assert len(result) == 3

