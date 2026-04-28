"""
Extra tests for bge_m3_turkish_detector.py — targeting uncovered lines:
203-232, 246-317, 363-364, 367, 454-481
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Autouse fixture ensures the real BGE-M3 model is never loaded
@pytest.fixture(autouse=True)
def mock_bge_m3_model():
    with patch('src.data_quality.processors.bge_m3_turkish_detector.BGEM3FlagModel') as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_cls

from src.data_quality.processors.bge_m3_turkish_detector import BGEM3TurkishDetector


# ──────────────────────────────────────────────────────────────────────
# 1.  _normalize_turkish_text (line 107) and _preprocess_turkish_text
#     with remove_stopwords=True and NLTK paths (lines 123-154)
# ──────────────────────────────────────────────────────────────────────
def test_normalize_text_enabled():
    proc = BGEM3TurkishDetector({'text_column': 'text', 'normalize_text': True})
    result = proc._normalize_turkish_text("ŞĞÜİÖÇ")
    assert isinstance(result, str)


def test_preprocess_with_stopwords():
    proc = BGEM3TurkishDetector({
        'text_column': 'text',
        'remove_stopwords': True,
    })
    result = proc._preprocess_turkish_text("Bu bir deneme cümlesidir.")
    assert isinstance(result, str)


def test_preprocess_stopwords_tokenize_exception():
    proc = BGEM3TurkishDetector({'text_column': 'text', 'remove_stopwords': True})
    with patch('src.data_quality.processors.bge_m3_turkish_detector.sent_tokenize',
               return_value=['hello world']):
        with patch('src.data_quality.processors.bge_m3_turkish_detector.word_tokenize',
                   side_effect=Exception("err")):
            result = proc._preprocess_turkish_text("hello world")
    assert isinstance(result, str)


def test_preprocess_stopwords_outer_exception():
    proc = BGEM3TurkishDetector({'text_column': 'text', 'remove_stopwords': True})
    with patch('src.data_quality.processors.bge_m3_turkish_detector.sent_tokenize',
               side_effect=Exception("err")):
        result = proc._preprocess_turkish_text("hello world")
    assert isinstance(result, str)


# ──────────────────────────────────────────────────────────────────────
# 2.  _compute_similarity_matrix — all four retrieval modes (lines 246-317)
# ──────────────────────────────────────────────────────────────────────
def _make_proc(mode='dense'):
    return BGEM3TurkishDetector({'text_column': 'text', 'retrieval_mode': mode})


def test_compute_similarity_matrix_dense():
    proc = _make_proc('dense')
    proc.bge_model = MagicMock()
    proc.bge_model.encode.return_value = {
        'dense_vecs': np.array([[1.0, 0.0], [0.0, 1.0]])
    }
    matrix = proc._compute_similarity_matrix(['a', 'b'])
    assert matrix.shape == (2, 2)


def test_compute_similarity_matrix_dense_non_ndarray():
    """Covers the isinstance branch that converts to np.array."""
    proc = _make_proc('dense')
    proc.bge_model = MagicMock()
    proc.bge_model.encode.return_value = {
        'dense_vecs': [[1.0, 0.0], [0.0, 1.0]]  # list, not ndarray
    }
    matrix = proc._compute_similarity_matrix(['a', 'b'])
    assert matrix.shape == (2, 2)


def test_compute_similarity_matrix_sparse():
    proc = _make_proc('sparse')
    proc.bge_model = MagicMock()
    proc.bge_model.encode.return_value = {
        'lexical_weights': [{'word': 1.0}, {'word': 0.5}]
    }
    proc.bge_model.compute_lexical_matching_score.return_value = 0.5
    matrix = proc._compute_similarity_matrix(['a', 'b'])
    assert matrix.shape == (2, 2)


def test_compute_similarity_matrix_sparse_ndarray_weights():
    """Covers the isinstance ndarray → tolist() branch for sparse weights."""
    proc = _make_proc('sparse')
    proc.bge_model = MagicMock()
    proc.bge_model.encode.return_value = {
        'lexical_weights': [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    }
    proc.bge_model.compute_lexical_matching_score.return_value = 0.3
    matrix = proc._compute_similarity_matrix(['a', 'b'])
    assert matrix.shape == (2, 2)


def test_compute_similarity_matrix_colbert():
    proc = _make_proc('colbert')
    proc.bge_model = MagicMock()
    proc.bge_model.encode.return_value = {
        'colbert_vecs': [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    }
    proc.bge_model.colbert_score.return_value = 0.4
    matrix = proc._compute_similarity_matrix(['a', 'b'])
    assert matrix.shape == (2, 2)


def test_compute_similarity_matrix_hybrid():
    proc = _make_proc('hybrid')
    proc.bge_model = MagicMock()
    proc.bge_model.compute_score.return_value = {
        'colbert+sparse+dense': [1.0, 0.5, 0.5, 1.0]
    }
    matrix = proc._compute_similarity_matrix(['a', 'b'])
    assert matrix.shape == (2, 2)


def test_compute_similarity_matrix_unknown_mode_raises():
    proc = _make_proc('unknown')
    proc.bge_model = MagicMock()
    with pytest.raises(Exception):
        proc._compute_similarity_matrix(['a', 'b'])


def test_compute_similarity_matrix_exception_propagates():
    proc = _make_proc('dense')
    proc.bge_model = MagicMock()
    proc.bge_model.encode.side_effect = Exception("encode error")
    with pytest.raises(Exception):
        proc._compute_similarity_matrix(['a', 'b'])


# ──────────────────────────────────────────────────────────────────────
# 3.  process — missing column (lines 363-364) and bge_m3 action (line 367)
# ──────────────────────────────────────────────────────────────────────
def test_process_missing_column_returns_df():
    proc = BGEM3TurkishDetector({'text_column': 'missing', 'action': 'mark'})
    df = pd.DataFrame({'text': ['a', 'b']})
    result = proc.process(df)
    assert 'is_duplicate' not in result.columns


def test_process_bge_m3_action_calls_bge_m3():
    proc = BGEM3TurkishDetector({'text_column': 'text', 'action': 'bge_m3'})
    proc._process_with_bge_m3 = MagicMock(return_value=pd.DataFrame({'text': ['a']}))
    df = pd.DataFrame({'text': ['a']})
    proc.process(df)
    proc._process_with_bge_m3.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# 4.  _process_with_bge_m3 — no text_column (lines 203-232)
# ──────────────────────────────────────────────────────────────────────
def test_process_with_bge_m3_returns_df():
    proc = BGEM3TurkishDetector({'text_column': 'text', 'action': 'bge_m3'})
    proc._compute_similarity_matrix = MagicMock(
        return_value=np.array([[1.0, 0.95], [0.95, 1.0]])
    )
    df = pd.DataFrame({'text': ['Alpha text', 'Beta text']})
    result = proc._process_with_bge_m3(df)
    assert 'is_duplicate' in result.columns
    assert list(result['is_duplicate']) == [False, True]


def test_process_with_bge_m3_exception_propagates():
    proc = BGEM3TurkishDetector({'text_column': 'text', 'action': 'bge_m3'})
    proc._compute_similarity_matrix = MagicMock(side_effect=Exception("compute error"))
    df = pd.DataFrame({'text': ['Alpha', 'Beta']})
    with pytest.raises(Exception, match="compute error"):
        proc._process_with_bge_m3(df)



# ──────────────────────────────────────────────────────────────────────
# 5.  process_with_statistics (lines 454-481)
# ──────────────────────────────────────────────────────────────────────
def test_process_with_statistics():
    proc = BGEM3TurkishDetector({'text_column': 'text', 'action': 'mark'})
    df = pd.DataFrame({'text': ['Apple', 'Banana', 'Apple']})
    result_df, stats = proc.process_with_statistics(df)
    assert isinstance(stats, dict)
    assert 'original_count' in stats
    assert 'duplicate_count' in stats
    assert stats['original_count'] == 3


def test_process_with_statistics_no_dup_cols():
    """When result has no is_duplicate/duplicate_group columns."""
    proc = BGEM3TurkishDetector({'text_column': 'missing', 'action': 'mark'})
    df = pd.DataFrame({'text': ['Apple']})
    result_df, stats = proc.process_with_statistics(df)
    assert stats['duplicate_count'] == 0
