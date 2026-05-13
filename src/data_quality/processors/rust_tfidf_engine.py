"""
Rust-accelerated TF-IDF cosine similarity engine for duplicate detection.

This module is a thin Python shim around the ``data_quality.rust_ext``
PyO3 extension.  It exposes the same surface API so callers never need to
know whether the Rust binary is available — the scikit-learn fallback is
used transparently.

Public API
----------
find_duplicate_indices_tfidf(lines, similarity_threshold) -> list[int]
RUST_TFIDF_AVAILABLE : bool
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  Try to load the compiled Rust extension
# ──────────────────────────────────────────────────────────────

try:
    from data_quality.rust_ext import (  # type: ignore[import]
        find_duplicate_indices_tfidf as _rust_find_duplicates,
    )

    RUST_TFIDF_AVAILABLE = True
    logger.debug(
        "rust_tfidf_engine: Rust extension loaded — TF-IDF fast path active."
    )
except ImportError:
    _rust_find_duplicates = None  # type: ignore[assignment]
    RUST_TFIDF_AVAILABLE = False
    logger.info(
        "rust_tfidf_engine: Rust extension not found — falling back to scikit-learn. "
        "Run `pip install -e .` (with setuptools-rust) or rebuild the extension "
        "to enable the 5-10x faster Rust path."
    )


# ──────────────────────────────────────────────────────────────
#  Python fallback (scikit-learn)
# ──────────────────────────────────────────────────────────────

def _py_find_duplicate_indices(
    lines: List[str],
    similarity_threshold: float,
) -> List[int]:
    """Pure-Python fallback using scikit-learn TF-IDF + cosine similarity."""
    n = len(lines)
    if n <= 1:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            lowercase=True,
        )
        # Handle empty or near-empty inputs gracefully
        non_empty = [line if line.strip() else " " for line in lines]
        tfidf_matrix = vectorizer.fit_transform(non_empty)
        sim_matrix = cosine_similarity(tfidf_matrix)

        duplicates: set[int] = set()
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > similarity_threshold:
                    duplicates.add(j)

        result = sorted(duplicates)
        return result

    except Exception as exc:
        logger.warning(
            "rust_tfidf_engine: scikit-learn fallback failed (%s). "
            "Returning no duplicates.",
            exc,
        )
        return []


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

def find_duplicate_indices_tfidf(
    lines: List[str],
    similarity_threshold: float = 0.8,
) -> List[int]:
    """Return indices of duplicate lines based on TF-IDF cosine similarity.

    For each pair of lines whose cosine similarity exceeds
    ``similarity_threshold``, the *later* line's index (``j``) is
    included in the output.  The first occurrence of a near-duplicate
    cluster is always kept.

    Uses the Rust extension when available (5-10× faster); falls back to
    scikit-learn otherwise.

    Parameters
    ----------
    lines:
        List of text strings to deduplicate.
    similarity_threshold:
        Cosine similarity threshold above which two lines are considered
        duplicates (default ``0.8``).

    Returns
    -------
    list[int]
        Sorted list of indices to remove (always empty when ``len(lines) <= 1``).
    """
    if RUST_TFIDF_AVAILABLE:
        return list(  # type: ignore[return-value]
            _rust_find_duplicates(lines, similarity_threshold)
        )
    return _py_find_duplicate_indices(lines, similarity_threshold)


__all__ = [
    "RUST_TFIDF_AVAILABLE",
    "find_duplicate_indices_tfidf",
]
