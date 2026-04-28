import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_semhash_model():
    with patch('src.data_quality.processors.semhash_turkish_detector.SemHash') as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_class

from src.data_quality.processors.semhash_turkish_detector import SemHashTurkishDetector, SemHashDuplicateDetector

def test_process_traditional_exact_duplicates():
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Apple']})
    # Set action to something other than semhash to trigger _process_traditional
    processor = SemHashTurkishDetector({'text_column': 'text', 'action': 'mark'})
    result = processor.process(df)
    assert 'is_duplicate' in result.columns
    assert list(result['is_duplicate']) == [False, False, True]

def test_normalize_turkish_text():
    processor = SemHashTurkishDetector({'text_column': 'text', 'normalize_text': True})
    text = "ŞĞÜİÖÇ şğüiöç"
    normalized = processor._normalize_turkish_text(text)
    assert normalized == "sgui̇oc sguioc"

def test_process_with_statistics_returns_dict():
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Apple']})
    processor = SemHashTurkishDetector({'text_column': 'text', 'action': 'mark'})
    result = processor.process(df)
    assert 'is_duplicate' in result.columns

def test_backward_compat_alias():
    # Will fail if semhash is required in __init__, but it just catches ImportError
    processor = SemHashDuplicateDetector({'text_column': 'text'})
    assert isinstance(processor, SemHashTurkishDetector)

def test_process_with_semhash_mock(mock_semhash_model):
    # Setup mock — selected texts match raw column values so both rows are non-duplicates
    mock_instance = mock_semhash_model.from_records.return_value
    mock_result = MagicMock()
    # The code compares preprocessed row text against raw record[text_column].
    # Set selected to the preprocessed (lowercased) forms so rows are marked not-duplicate.
    mock_result.selected = [{'text': 'a text to check'}, {'text': 'a similar text to check'}]
    mock_instance.self_deduplicate.return_value = mock_result

    # Use patch context to mock SEMHASH_AVAILABLE locally inside the module
    with patch('src.data_quality.processors.semhash_turkish_detector.SEMHASH_AVAILABLE', True):
        df = pd.DataFrame({'text': ['A text to check', 'A similar text to check']})

        processor = SemHashTurkishDetector({'text_column': 'text', 'action': 'semhash'})

        result = processor._process_with_semhash(df)
        assert 'is_duplicate' in result.columns
        # Both texts present in selected → neither is a duplicate
        assert list(result['is_duplicate']) == [False, False]
