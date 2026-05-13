"""
Processor-level benchmarks for teo-eda-toolkit.

Run with:
    pytest tests/benchmarks/bench_processors.py --benchmark-autosave -v
"""

from __future__ import annotations

import sys
import os
from typing import List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

try:
    import pytest_benchmark  # noqa: F401
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

from data_quality.processors.rust_regex_engine import (
    RUST_AVAILABLE as RUST_REGEX_AVAILABLE,
    clean_texts,
    _py_clean_one,
)
from data_quality.processors.rust_tfidf_engine import (
    RUST_TFIDF_AVAILABLE,
    find_duplicate_indices_tfidf,
    _py_find_duplicate_indices,
)

# ──────────────────────────────────────────────────────────────
#  Shared corpus
# ──────────────────────────────────────────────────────────────

_BOILERPLATE_TEXT = (
    "TÜRKİYE CUMHURİYETİ Adalet Bakanlığı\n"
    "Dosya No: 2024/1234\n"
    "Bu karar kesinleşmiştir.\n"
    "Mahkeme kararı kesinleşmiştir.\n"
    "Bu metin asıl içeriktir ve temizlenmemelidir.\n"
    "Tarih: 15.06.2024\n"
    "Sayfa 1 / 3\n"
)

_BATCH_200: List[str] = [_BOILERPLATE_TEXT] * 200

_CORPUS: List[str] = (
    [
        f"Türk hukuku kapsamında belge {i}: hukuki düzenlemeler ve yükümlülükler."
        for i in range(100)
    ]
    + [
        f"Türk hukuku kapsamında belge {i}: hukuki düzenlemeler ve yükümlülükler."
        for i in range(25)  # near-duplicates
    ]
)


def _py_clean_batch(texts: List[str]) -> List[str]:
    return [_py_clean_one(t, True, True, []) for t in texts]


# ──────────────────────────────────────────────────────────────
#  Correctness smoke tests (always run)
# ──────────────────────────────────────────────────────────────

def test_batch_cleans_boilerplate() -> None:
    results = clean_texts([_BOILERPLATE_TEXT] * 10)
    assert len(results) == 10
    for r in results:
        assert "TÜRKİYE CUMHURİYETİ" not in r
        assert "Dosya No:" not in r
        assert "asıl içeriktir" in r


def test_tfidf_finds_duplicates_in_corpus() -> None:
    result = find_duplicate_indices_tfidf(_CORPUS, 0.9)
    assert len(result) >= 20, f"Expected >=20 duplicates, got {len(result)}"


def test_minhash_duplicate_detector_marks() -> None:
    import pandas as pd
    from data_quality.processors.duplicate_detector import TurkishDuplicateDetector

    texts = (
        ["Bu metin tamamen aynıdır ve kopyalanmıştır."] * 5
        + ["Farklı bir metin içeriği burada yer almaktadır."] * 3
        + [f"Benzersiz metin örneği sayı {i}." for i in range(5)]
    )
    df = pd.DataFrame({"text": texts})
    detector = TurkishDuplicateDetector(
        config={"text_column": "text", "similarity_threshold": 0.85, "action": "mark"}
    )
    result_df = detector.process(df)
    assert "is_duplicate" in result_df.columns
    assert result_df["is_duplicate"].sum() >= 4


# ──────────────────────────────────────────────────────────────
#  Benchmarks
# ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
def test_bench_regex_python_batch(benchmark):  # type: ignore[no-untyped-def]
    """Python regex cleaning: batch of 200 texts."""
    benchmark(lambda: _py_clean_batch(_BATCH_200))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.skipif(not RUST_REGEX_AVAILABLE, reason="Rust regex extension not compiled")
def test_bench_regex_rust_batch(benchmark):  # type: ignore[no-untyped-def]
    """Rust regex cleaning: batch of 200 texts (Rayon parallel)."""
    benchmark(lambda: clean_texts(_BATCH_200))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
def test_bench_tfidf_python_corpus(benchmark):  # type: ignore[no-untyped-def]
    """Scikit-learn TF-IDF deduplication on 125-text corpus."""
    benchmark(lambda: _py_find_duplicate_indices(_CORPUS, 0.9))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.skipif(not RUST_TFIDF_AVAILABLE, reason="Rust TF-IDF not compiled")
def test_bench_tfidf_rust_corpus(benchmark):  # type: ignore[no-untyped-def]
    """Rust TF-IDF deduplication on 125-text corpus."""
    benchmark(lambda: find_duplicate_indices_tfidf(_CORPUS, 0.9))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
def test_bench_minhash_detector(benchmark):  # type: ignore[no-untyped-def]
    """MinHash/LSH TurkishDuplicateDetector on 125-text corpus."""
    import pandas as pd
    from data_quality.processors.duplicate_detector import TurkishDuplicateDetector

    df = pd.DataFrame({"text": _CORPUS})
    detector = TurkishDuplicateDetector(
        config={
            "text_column": "text",
            "similarity_threshold": 0.85,
            "num_perm": 64,
            "action": "mark",
        }
    )
    benchmark(lambda: detector.process(df.copy()))
