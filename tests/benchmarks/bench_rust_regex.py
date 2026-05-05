"""
pytest-benchmark comparison: Python regex fallback vs. Rust extension.

Run with:
    pytest tests/benchmarks/bench_rust_regex.py --benchmark-autosave -v
"""

from __future__ import annotations

import sys
import os

import pytest

# Make sure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from data_quality.processors.rust_regex_engine import (
    RUST_AVAILABLE,
    clean_text,
    clean_texts,
    _py_clean_one,
    _COMPILED_TURKISH,
    _COMPILED_COMMON,
    _WS_RE,
)

# ──────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────

SAMPLE_SHORT = (
    "TÜRKİYE CUMHURİYETİ Adalet Bakanlığı\n"
    "Dosya No: 2024/1234\n"
    "Bu karar kesinleşmiştir.\n"
    "Mahkeme kararı kesinleşmiştir.\n"
    "Bu metin asıl içeriktir ve temizlenmemelidir."
)

SAMPLE_LONG = (SAMPLE_SHORT + "\n") * 200  # ~200-line document


def _py_clean(text: str) -> str:
    """Isolated Python path (no Rust)."""
    return _py_clean_one(text, True, True, [])


# ──────────────────────────────────────────────────────────────
#  Correctness sanity check (runs as a normal test too)
# ──────────────────────────────────────────────────────────────


def test_rust_cleans_same_as_python() -> None:
    py_result = _py_clean(SAMPLE_SHORT)
    rs_result = clean_text(SAMPLE_SHORT)
    # Both must strip the known boilerplate tokens
    assert "TÜRKİYE CUMHURİYETİ" not in rs_result
    assert "Dosya No:" not in rs_result
    assert "asıl içeriktir" in rs_result, "Payload must survive cleaning"
    if RUST_AVAILABLE:
        assert py_result == rs_result, "Rust and Python paths must produce identical output"


def test_clean_texts_batch() -> None:
    texts = [SAMPLE_SHORT] * 50
    results = clean_texts(texts)
    assert len(results) == 50
    for r in results:
        assert "TÜRKİYE CUMHURİYETİ" not in r
        assert "asıl içeriktir" in r


def test_custom_patterns() -> None:
    text = "SECRET_TOKEN: abc123\nNormal content here."
    result = clean_text(text, custom_patterns=[r"SECRET_TOKEN:\s*\S+"])
    assert "SECRET_TOKEN" not in result
    assert "Normal content" in result


def test_rust_available_flag() -> None:
    """Just assert the flag is a bool (extension may or may not be present)."""
    assert isinstance(RUST_AVAILABLE, bool)


# ──────────────────────────────────────────────────────────────
#  Benchmarks (only run when pytest-benchmark is installed)
# ──────────────────────────────────────────────────────────────

try:
    import pytest_benchmark  # noqa: F401

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
def test_bench_python_short(benchmark):  # type: ignore[no-untyped-def]
    benchmark(lambda: _py_clean(SAMPLE_SHORT))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not compiled")
def test_bench_rust_short(benchmark):  # type: ignore[no-untyped-def]
    benchmark(lambda: clean_text(SAMPLE_SHORT))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
def test_bench_python_long(benchmark):  # type: ignore[no-untyped-def]
    benchmark(lambda: _py_clean(SAMPLE_LONG))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not compiled")
def test_bench_rust_long(benchmark):  # type: ignore[no-untyped-def]
    benchmark(lambda: clean_text(SAMPLE_LONG))


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
def test_bench_python_batch(benchmark):  # type: ignore[no-untyped-def]
    texts = [SAMPLE_SHORT] * 100
    benchmark(lambda: [_py_clean(t) for t in texts])


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not compiled")
def test_bench_rust_batch(benchmark):  # type: ignore[no-untyped-def]
    texts = [SAMPLE_SHORT] * 100
    benchmark(lambda: clean_texts(texts))
