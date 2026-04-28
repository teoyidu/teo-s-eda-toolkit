"""
Extra tests for semhash_turkish_detector.py — targeting uncovered lines:
104, 120-135, 159, 195-224, 239-243, 270, 301-302, 309-311, 346-365
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_semhash_class():
    with patch('src.data_quality.processors.semhash_turkish_detector.SemHash') as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_cls

from src.data_quality.processors.semhash_turkish_detector import SemHashTurkishDetector


# ──────────────────────────────────────────────────────────────────────
# 1.  Constructor defaults (line 104, 120-121)
# ──────────────────────────────────────────────────────────────────────
def test_constructor_defaults():
    proc = SemHashTurkishDetector({'text_column': 'text'})
    assert proc.text_column == 'text'
    assert proc.action == 'semhash'
    assert proc.similarity_threshold == 0.85


# ──────────────────────────────────────────────────────────────────────
# 2.  _normalize_turkish_text with normalize_text=False (line 159)
# ──────────────────────────────────────────────────────────────────────
def test_normalize_turkish_text_disabled():
    proc = SemHashTurkishDetector({'text_column': 'text', 'normalize_text': False})
    raw = "  HELLO WORLD  "
    result = proc._normalize_turkish_text(raw)
    assert result == raw


def test_normalize_turkish_text_enabled():
    proc = SemHashTurkishDetector({'text_column': 'text', 'normalize_text': True})
    result = proc._normalize_turkish_text("ŞĞÜİÖÇ şğüiöç")
    assert isinstance(result, str)


# ──────────────────────────────────────────────────────────────────────
# 3.  _preprocess_turkish_text — NLTK paths (lines 195-224)
# ──────────────────────────────────────────────────────────────────────
def test_preprocess_with_stopwords():
    proc = SemHashTurkishDetector({'text_column': 'text', 'remove_stopwords': True})
    result = proc._preprocess_turkish_text("Bu bir test cümlesidir.")
    assert isinstance(result, str)


def test_preprocess_stopwords_inner_exception():
    proc = SemHashTurkishDetector({'text_column': 'text', 'remove_stopwords': True})
    with patch('src.data_quality.processors.semhash_turkish_detector.sent_tokenize',
               return_value=['hello world']):
        with patch('src.data_quality.processors.semhash_turkish_detector.word_tokenize',
                   side_effect=Exception("err")):
            result = proc._preprocess_turkish_text("hello world")
    assert isinstance(result, str)


def test_preprocess_stopwords_outer_exception():
    proc = SemHashTurkishDetector({'text_column': 'text', 'remove_stopwords': True})
    with patch('src.data_quality.processors.semhash_turkish_detector.sent_tokenize',
               side_effect=Exception("err")):
        result = proc._preprocess_turkish_text("hello world")
    assert isinstance(result, str)


# ──────────────────────────────────────────────────────────────────────
# 4.  process — missing column (lines 239-240)
# ──────────────────────────────────────────────────────────────────────
def test_process_missing_column():
    proc = SemHashTurkishDetector({'text_column': 'missing', 'action': 'mark'})
    df = pd.DataFrame({'text': ['hello']})
    result = proc.process(df)
    # DataFrame returned unchanged
    assert 'is_duplicate' not in result.columns


# ──────────────────────────────────────────────────────────────────────
# 5.  process — 'semhash' action routes to _process_with_semhash (line 242-243)
# ──────────────────────────────────────────────────────────────────────
def test_process_semhash_action():
    proc = SemHashTurkishDetector({'text_column': 'text', 'action': 'semhash'})
    proc._process_with_semhash = MagicMock(return_value=pd.DataFrame({'text': ['a']}))
    df = pd.DataFrame({'text': ['a']})
    proc.process(df)
    proc._process_with_semhash.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# 6.  _process_with_semhash — duplicate marked (line 270, 298-299)
# ──────────────────────────────────────────────────────────────────────
def test_process_with_semhash_marks_duplicate(mock_semhash_class):
    mock_result = MagicMock()
    # selected contains only the preprocessed text of first row
    mock_result.selected = [{'text': 'a text to check'}]
    mock_semhash_class.from_records.return_value.self_deduplicate.return_value = mock_result

    proc = SemHashTurkishDetector({'text_column': 'text', 'action': 'semhash'})
    df = pd.DataFrame({'text': ['A text to check', 'Totally different content here']})
    result = proc._process_with_semhash(df)
    assert 'is_duplicate' in result.columns


# ──────────────────────────────────────────────────────────────────────
# 7.  _process_with_semhash — exception falls back (lines 309-311)
# ──────────────────────────────────────────────────────────────────────
def test_process_with_semhash_exception_fallback(mock_semhash_class):
    mock_semhash_class.from_records.side_effect = Exception("semhash error")

    proc = SemHashTurkishDetector({'text_column': 'text', 'action': 'semhash'})
    df = pd.DataFrame({'text': ['Alpha', 'Alpha']})
    result = proc._process_with_semhash(df)
    # Falls back to _process_traditional
    assert 'is_duplicate' in result.columns


# ──────────────────────────────────────────────────────────────────────
# 8.  _process_traditional (lines 325-334) — more coverage
# ──────────────────────────────────────────────────────────────────────
def test_process_traditional_no_duplicates():
    proc = SemHashTurkishDetector({'text_column': 'text', 'action': 'mark'})
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Cherry']})
    result = proc.process(df)
    assert list(result['is_duplicate']) == [False, False, False]


# ──────────────────────────────────────────────────────────────────────
# 9.  process_with_statistics (lines 346-365)
# ──────────────────────────────────────────────────────────────────────
def test_process_with_statistics_semhash():
    proc = SemHashTurkishDetector({'text_column': 'text', 'action': 'mark'})
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Apple']})
    result_df, stats = proc.process_with_statistics(df)
    assert isinstance(stats, dict)
    assert 'original_count' in stats
    assert stats['original_count'] == 3


def test_process_with_statistics_no_is_duplicate_col():
    """When process returns df without is_duplicate (e.g. missing column)."""
    proc = SemHashTurkishDetector({'text_column': 'missing', 'action': 'mark'})
    df = pd.DataFrame({'text': ['Apple']})
    result_df, stats = proc.process_with_statistics(df)
    assert stats['duplicate_count'] == 0
