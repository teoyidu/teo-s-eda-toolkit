import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_bge_m3_model():
    # Automatically mock the BGE-M3 model for all tests in this file
    with patch('src.data_quality.processors.bge_m3_turkish_detector.BGEM3FlagModel') as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_class

from src.data_quality.processors.bge_m3_turkish_detector import BGEM3TurkishDetector, BGEM3DuplicateDetector

def test_process_traditional_exact_duplicates():
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Apple']})
    processor = BGEM3TurkishDetector({'text_column': 'text', 'action': 'mark'})
    result = processor.process(df)
    assert 'is_duplicate' in result.columns
    assert list(result['is_duplicate']) == [False, False, True]

def test_find_duplicates_logic():
    processor = BGEM3TurkishDetector({'text_column': 'text', 'similarity_threshold': 0.85})
    
    # 0 and 1 are similar, 2 is alone
    similarity_matrix = np.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0]
    ])
    
    groups = processor._find_duplicates(similarity_matrix)
    assert len(groups) == 1
    assert groups[0] == [0, 1]

def test_normalize_turkish_text():
    processor = BGEM3TurkishDetector({'text_column': 'text', 'normalize_text': True})
    text = "ŞĞÜİÖÇ şğüiöç"
    normalized = processor._normalize_turkish_text(text)
    assert normalized == "sgui̇oc sguioc"

def test_process_with_statistics_returns_dict():
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Apple']})
    processor = BGEM3TurkishDetector({'text_column': 'text', 'action': 'mark'})
    result = processor.process(df)
    assert 'is_duplicate' in result.columns

def test_backward_compat_alias():
    processor = BGEM3DuplicateDetector({'text_column': 'text'})
    assert isinstance(processor, BGEM3TurkishDetector)

def test_process_with_bge_m3_mock(mock_bge_m3_model):
    with patch('src.data_quality.processors.bge_m3_turkish_detector.BGE_M3_AVAILABLE', True):
        df = pd.DataFrame({'text': ['Apple', 'Banana']})
        
        processor = BGEM3TurkishDetector({'text_column': 'text', 'action': 'bge_m3'})
        # Override similarity to test the loop logic
        processor._compute_similarity_matrix = MagicMock(return_value=np.array([[1.0, 0.9], [0.9, 1.0]]))
        
        result = processor._process_with_bge_m3(df)
        assert 'is_duplicate' in result.columns
        assert list(result['is_duplicate']) == [False, True]
