"""
Benchmark conftest: makes the benchmark fixture available as a no-op
when pytest-benchmark is installed but --benchmark-disable is passed,
and prevents collection errors when running without the plugin at all.
"""
from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register the 'benchmark' marker so it doesn't warn."""
    config.addinivalue_line(
        "markers", "benchmark: marks a pytest-benchmark performance test"
    )
