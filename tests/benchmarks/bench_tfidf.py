"""
pytest-benchmark comparison: Python scikit-learn vs Rust TF-IDF duplicate detection.

Run with:
    pytest tests/benchmarks/bench_tfidf.py --benchmark-autosave -v

Or just for the benchmark numbers (no correctness tests):
    pytest tests/benchmarks/bench_tfidf.py --benchmark-only -v
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from data_quality.processors.rust_tfidf_engine import (
    RUST_TFIDF_AVAILABLE,
    find_duplicate_indices_tfidf,
    _py_find_duplicate_indices,
)

# ──────────────────────────────────────────────────────────────
#  Test fixtures
# ──────────────────────────────────────────────────────────────

# 50 lines: 25 unique + 25 near-duplicates (cosine > 0.9 for pairs)
_UNIQUE_LINES = [
    "Türkiye Cumhuriyeti Anayasası kapsamında birey hakları güvence altına alınmıştır.",
    "Hukuki süreçlerde taraflar eşit haklara sahiptir ve adil yargılanma hakkı tanınmıştır.",
    "Sözleşme hukuku çerçevesinde tarafların yükümlülükleri açıkça belirlenmiştir.",
    "İdare hukuku devletin bireylerle ilişkisini düzenleyen hukuk dalıdır.",
    "Ceza hukuku toplumsal düzeni korumak amacıyla suç ve cezaları tanımlar.",
    "Medeni hukuk kişilerin özel ilişkilerini düzenleyen hukuk alanıdır.",
    "Ticaret hukuku ticari işlemleri ve tüccarlar arasındaki ilişkileri kapsar.",
    "İş hukuku işçi ve işveren arasındaki ilişkileri düzenleyen hukuk dalıdır.",
    "Vergi hukuku devletin vergi toplama yetkisini ve prosedürlerini düzenler.",
    "Miras hukuku kişilerin ölümünden sonra mal varlıklarının devrine ilişkindir.",
    "Aile hukuku evlilik boşanma velayet ve nafaka gibi aile ilişkilerini düzenler.",
    "Eşya hukuku taşınır ve taşınmaz mallar üzerindeki hakları düzenler.",
    "Borçlar hukuku alacak ve borç ilişkilerini düzenleyen temel hukuk dalıdır.",
    "Fikir ve sanat eserleri hukuku telif haklarını ve fikri mülkiyeti korur.",
    "Rekabet hukuku piyasada adil rekabet ortamının sağlanmasını amaçlar.",
    "Uluslararası hukuk devletler arasındaki ilişkileri düzenleyen hukuk sistemidir.",
    "Anayasa mahkemesi bireysel başvuru yoluyla temel hakları koruma görevi üstlenir.",
    "Danıştay idari uyuşmazlıkları çözen en yüksek idare mahkemesidir.",
    "Yargıtay hukuk ve ceza davalarında temyiz mercii olarak görev yapar.",
    "Avrupa insan hakları mahkemesi temel hakların ihlali halinde başvurulabilir.",
    "Hukuk fakültesi mezunları avukat hakim veya noter olabilmektedir.",
    "Baro avukatların mesleki örgütü olup hukuki etik kurallarını belirler.",
    "Noterlik hukuki belgeleri onaylayan ve tasdik eden kamu görevidir.",
    "İcra ve iflas hukuku borçların zorla tahsil edilmesini düzenler.",
    "Tüketici hukuku tüketicilerin ticari işlemlerde korunmasını sağlar.",
]

SMALL_LINES = _UNIQUE_LINES + [line + " Bu konu son derece önemlidir." for line in _UNIQUE_LINES]  # 50 lines
LARGE_LINES = _UNIQUE_LINES * 10 + [line + " Ayrıca belirtmek gerekir ki." for line in _UNIQUE_LINES * 10]  # 500 lines

THRESHOLD = 0.8


# ──────────────────────────────────────────────────────────────
#  Correctness tests (run always)
# ──────────────────────────────────────────────────────────────

def test_rust_and_python_agree_small() -> None:
    """Rust and Python fallback must agree on duplicate indices for small input."""
    py_result = _py_find_duplicate_indices(SMALL_LINES, THRESHOLD)
    if RUST_TFIDF_AVAILABLE:
        rs_result = find_duplicate_indices_tfidf(SMALL_LINES, THRESHOLD)
        assert py_result == rs_result, (
            f"Rust and Python paths disagree:\n  Rust: {rs_result}\n  Python: {py_result}"
        )


def test_identical_lines_are_duplicates() -> None:
    """Identical lines must always be detected as duplicates."""
    lines = ["Bu metin tamamen aynıdır."] * 5
    result = find_duplicate_indices_tfidf(lines, 0.5)
    # Indices 1,2,3,4 should be flagged (0 is kept)
    assert result == [1, 2, 3, 4], f"Expected [1,2,3,4], got {result}"


def test_empty_input() -> None:
    """Empty and single-line inputs return empty list."""
    assert find_duplicate_indices_tfidf([], 0.9) == []
    assert find_duplicate_indices_tfidf(["single line"], 0.9) == []


def test_completely_different_lines() -> None:
    """Completely different lines should have no duplicates."""
    lines = ["alpha beta gamma", "delta epsilon zeta", "theta iota kappa"]
    result = find_duplicate_indices_tfidf(lines, 0.95)
    assert result == [], f"Unexpected duplicates: {result}"


def test_rust_available_flag() -> None:
    """RUST_TFIDF_AVAILABLE must be a bool."""
    assert isinstance(RUST_TFIDF_AVAILABLE, bool)


# ──────────────────────────────────────────────────────────────
#  Benchmarks (only run when pytest-benchmark is installed)
# ──────────────────────────────────────────────────────────────

try:
    import pytest_benchmark  # noqa: F401
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
def test_bench_python_tfidf_small(benchmark):  # type: ignore[no-untyped-def]
    """Scikit-learn TF-IDF on 50 lines."""
    benchmark(lambda: _py_find_duplicate_indices(SMALL_LINES, THRESHOLD))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.skipif(not RUST_TFIDF_AVAILABLE, reason="Rust extension not compiled")
def test_bench_rust_tfidf_small(benchmark):  # type: ignore[no-untyped-def]
    """Rust TF-IDF on 50 lines."""
    benchmark(lambda: find_duplicate_indices_tfidf(SMALL_LINES, THRESHOLD))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
def test_bench_python_tfidf_large(benchmark):  # type: ignore[no-untyped-def]
    """Scikit-learn TF-IDF on 500 lines."""
    benchmark(lambda: _py_find_duplicate_indices(LARGE_LINES, THRESHOLD))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.skipif(not RUST_TFIDF_AVAILABLE, reason="Rust extension not compiled")
def test_bench_rust_tfidf_large(benchmark):  # type: ignore[no-untyped-def]
    """Rust TF-IDF on 500 lines."""
    benchmark(lambda: find_duplicate_indices_tfidf(LARGE_LINES, THRESHOLD))
