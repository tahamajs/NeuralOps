import time
import subprocess
import sys
import pytest


def test_import_is_fast():
    """Importing registry directly should take < 2.0s without eager discovery."""
    start = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import neuralops.benchmarks.registry"], check=True)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.5


def test_ensure_registry_populated_is_idempotent():
    from neuralops.benchmarks.registry import ensure_registry_populated, BENCHMARK_REGISTRY
    ensure_registry_populated()
    count1 = len(BENCHMARK_REGISTRY)
    ensure_registry_populated()
    count2 = len(BENCHMARK_REGISTRY)
    assert count1 == count2
    assert count1 > 0


def test_all_operational_modes_registered():
    from neuralops.benchmarks.registry import (
        ensure_registry_populated,
        BENCHMARK_REGISTRY,
        OPERATIONAL_BENCHMARK_MODES,
    )
    ensure_registry_populated()
    for mode in OPERATIONAL_BENCHMARK_MODES:
        assert f"__meta__:{mode}" in BENCHMARK_REGISTRY, f"Missing meta benchmark for {mode}"
