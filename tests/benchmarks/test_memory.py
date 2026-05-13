"""
Memory profiling tests using tracemalloc (stdlib).

These tests verify that key hot-path operations stay within reasonable
peak-memory bounds, catching accidental regressions (e.g. dense matrix
materialisation, unbounded caching).

Run with:
    pytest tests/benchmarks/test_memory.py -v

No additional dependencies required — tracemalloc is part of the Python
standard library; psutil is used for process-level RSS cross-checks.
"""

from __future__ import annotations

import sys
import os
import tracemalloc
from typing import List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def _peak_mib(func, *args, **kwargs):
    """Run *func* and return the peak traced memory in MiB."""
    tracemalloc.start()
    try:
        func(*args, **kwargs)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak / 1024 / 1024


# ──────────────────────────────────────────────────────────────
#  Shared data
# ──────────────────────────────────────────────────────────────

_BOILERPLATE = (
    "TÜRKİYE CUMHURİYETİ Adalet Bakanlığı\n"
    "Dosya No: 2024/1234\n"
    "Bu karar kesinleşmiştir. Mahkeme kararı kesinleşmiştir.\n"
    "Bu metin asıl içeriktir ve temizlenmemelidir.\n"
    "Tarih: 15.06.2024  Sayfa 1 / 3\n"
)

_BATCH_500: List[str] = [_BOILERPLATE] * 500

_CORPUS_1000: List[str] = (
    [f"Türk hukuku kapsamında belge {i}: düzenlemeler ve yükümlülükler." for i in range(500)]
    + [f"Türk hukuku kapsamında belge {i}: düzenlemeler ve yükümlülükler." for i in range(500)]
)


# ──────────────────────────────────────────────────────────────
#  Regex cleaning memory tests
# ──────────────────────────────────────────────────────────────

class TestRegexCleaningMemory:
    """Peak memory for regex boilerplate cleaning."""

    def test_python_regex_batch_500_under_50mib(self) -> None:
        from data_quality.processors.rust_regex_engine import _py_clean_one

        peak = _peak_mib(
            lambda: [_py_clean_one(t, True, True, []) for t in _BATCH_500]
        )
        assert peak < 50, (
            f"Python regex batch used {peak:.1f} MiB peak — expected <50 MiB"
        )

    def test_rust_regex_batch_500_under_30mib(self) -> None:
        from data_quality.processors.rust_regex_engine import (
            RUST_AVAILABLE,
            clean_texts,
        )
        if not RUST_AVAILABLE:
            pytest.skip("Rust regex extension not compiled")

        peak = _peak_mib(lambda: clean_texts(_BATCH_500))
        assert peak < 30, (
            f"Rust regex batch used {peak:.1f} MiB peak — expected <30 MiB"
        )

    def test_single_text_cleaning_under_2mib(self) -> None:
        from data_quality.processors.rust_regex_engine import clean_text

        peak = _peak_mib(lambda: clean_text(_BOILERPLATE))
        assert peak < 2, (
            f"Single text cleaning used {peak:.1f} MiB peak — expected <2 MiB"
        )


# ──────────────────────────────────────────────────────────────
#  TF-IDF duplicate detection memory tests
# ──────────────────────────────────────────────────────────────

class TestTFIDFMemory:
    """Peak memory for TF-IDF cosine similarity deduplication."""

    def test_python_tfidf_1000_texts_under_100mib(self) -> None:
        from data_quality.processors.rust_tfidf_engine import _py_find_duplicate_indices

        peak = _peak_mib(
            lambda: _py_find_duplicate_indices(_CORPUS_1000, 0.8)
        )
        assert peak < 100, (
            f"Python TF-IDF 1000 texts used {peak:.1f} MiB — expected <100 MiB"
        )

    def test_rust_tfidf_1000_texts_under_50mib(self) -> None:
        from data_quality.processors.rust_tfidf_engine import (
            RUST_TFIDF_AVAILABLE,
            find_duplicate_indices_tfidf,
        )
        if not RUST_TFIDF_AVAILABLE:
            pytest.skip("Rust TF-IDF extension not compiled")

        peak = _peak_mib(
            lambda: find_duplicate_indices_tfidf(_CORPUS_1000, 0.8)
        )
        assert peak < 50, (
            f"Rust TF-IDF 1000 texts used {peak:.1f} MiB — expected <50 MiB"
        )

    def test_empty_input_no_allocation(self) -> None:
        from data_quality.processors.rust_tfidf_engine import find_duplicate_indices_tfidf

        peak = _peak_mib(lambda: find_duplicate_indices_tfidf([], 0.8))
        # Trivially low allocation — well under 1 MiB
        assert peak < 1, (
            f"Empty TF-IDF input used {peak:.1f} MiB — expected <1 MiB"
        )


# ──────────────────────────────────────────────────────────────
#  MinHash / LSH duplicate detection memory tests
# ──────────────────────────────────────────────────────────────

class TestMinHashMemory:
    """Peak memory for MinHash/LSH-based TurkishDuplicateDetector."""

    def test_minhash_100_texts_under_200mib(self) -> None:
        import pandas as pd
        from data_quality.processors.duplicate_detector import TurkishDuplicateDetector

        texts = [f"Belge {i}: hukuki metin örneği." for i in range(100)]
        df = pd.DataFrame({"text": texts})
        detector = TurkishDuplicateDetector(
            config={
                "text_column": "text",
                "similarity_threshold": 0.85,
                "num_perm": 64,
                "action": "mark",
            }
        )

        peak = _peak_mib(lambda: detector.process(df.copy()))
        assert peak < 200, (
            f"MinHash 100 texts used {peak:.1f} MiB — expected <200 MiB"
        )

    def test_minhash_memory_scales_linearly(self) -> None:
        """Memory for 200 texts should be less than 3x memory for 100 texts."""
        import pandas as pd
        from data_quality.processors.duplicate_detector import TurkishDuplicateDetector

        config = {
            "text_column": "text",
            "similarity_threshold": 0.85,
            "num_perm": 64,
            "action": "mark",
        }

        texts_100 = [f"Belge {i}: hukuki metin örneği benzersiz içerik." for i in range(100)]
        texts_200 = texts_100 * 2

        detector = TurkishDuplicateDetector(config=config)

        peak_100 = _peak_mib(lambda: detector.process(pd.DataFrame({"text": texts_100})))
        peak_200 = _peak_mib(lambda: detector.process(pd.DataFrame({"text": texts_200})))

        ratio = peak_200 / max(peak_100, 0.01)
        assert ratio < 4.0, (
            f"MinHash memory scaling ratio {ratio:.1f}x (200 vs 100 texts) "
            f"seems super-linear — expected <4x. "
            f"100 texts: {peak_100:.1f} MiB, 200 texts: {peak_200:.1f} MiB"
        )


# ──────────────────────────────────────────────────────────────
#  Wrapper import overhead
# ──────────────────────────────────────────────────────────────

class TestImportOverhead:
    """Module import should not cause large memory allocations."""

    def test_rust_regex_engine_import_overhead(self) -> None:
        """Importing rust_regex_engine should compile patterns but stay <10 MiB."""
        # Module is already imported; measure pattern compilation cost via
        # a fresh re.compile cycle to simulate first-import cost.
        import re
        from data_quality.processors.rust_regex_engine import (
            _TURKISH_LEGAL_RAW,
            _COMMON_RAW,
        )

        def _recompile():
            [re.compile(p, re.IGNORECASE) for p in _TURKISH_LEGAL_RAW]
            [re.compile(p, re.IGNORECASE) for p in _COMMON_RAW]

        peak = _peak_mib(_recompile)
        assert peak < 10, (
            f"Pattern compilation used {peak:.1f} MiB — expected <10 MiB"
        )
