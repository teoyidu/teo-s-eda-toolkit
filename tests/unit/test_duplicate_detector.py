import pandas as pd
import pytest
from src.data_quality.processors.duplicate_detector import TurkishDuplicateDetector, DuplicateDetector

def test_requires_text_column():
    with pytest.raises(ValueError, match="text_column must be specified in config"):
        TurkishDuplicateDetector({})

def test_mark_action_adds_columns():
    df = pd.DataFrame({'text': ['This is a really long document about apples', 'This is a really long document about bananas', 'This is a really long document about apples']})
    processor = TurkishDuplicateDetector({'text_column': 'text', 'action': 'mark'})
    result = processor.process(df)
    assert 'is_duplicate' in result.columns
    assert 'duplicate_group' in result.columns
    # Check that duplicates are grouped
    assert list(result['is_duplicate']) == [True, False, True]

def test_remove_action_drops_rows():
    df = pd.DataFrame({'text': ['This is a really long document about apples', 'This is a really long document about bananas', 'This is a really long document about apples']})
    processor = TurkishDuplicateDetector({'text_column': 'text', 'action': 'remove'})
    result = processor.process(df)
    assert len(result) == 2
    assert list(result['text']) == ['This is a really long document about apples', 'This is a really long document about bananas']

def test_group_action_adds_group_col():
    df = pd.DataFrame({'text': ['This is a really long document about apples', 'This is a really long document about bananas', 'This is a really long document about apples']})
    processor = TurkishDuplicateDetector({'text_column': 'text', 'action': 'group'})
    result = processor.process(df)
    assert 'duplicate_group' in result.columns

def test_normalize_turkish_text():
    processor = TurkishDuplicateDetector({'text_column': 'text', 'normalize_text': True})
    text = "ŞĞÜİÖÇ şğüiöç"
    normalized = processor._normalize_turkish_text(text)
    assert normalized == "sgui̇oc sguioc"

def test_identical_texts_detected():
    df = pd.DataFrame({'text': ['This is a test document.', 'This is a test document.']})
    processor = TurkishDuplicateDetector({'text_column': 'text', 'action': 'mark'})
    result = processor.process(df)
    # The stats aren't actually returned by duplicate detector process method.
    assert 'is_duplicate' in result.columns
    assert sum(result['is_duplicate']) == 2

def test_backward_compat_alias():
    processor = DuplicateDetector({'text_column': 'text'})
    assert isinstance(processor, TurkishDuplicateDetector)
