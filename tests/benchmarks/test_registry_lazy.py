# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying lazy initialization and import performance of benchmark registry."""

from __future__ import annotations

import subprocess
import sys


def test_import_is_fast() -> None:
    """Assert import neuralops.benchmarks.registry takes < 200ms."""
    code = (
        "import time\n"
        "t0 = time.perf_counter()\n"
        "import neuralops.benchmarks.registry\n"
        "elapsed_ms = (time.perf_counter() - t0) * 1000\n"
        "assert elapsed_ms < 200, f'Import took {elapsed_ms:.1f}ms (threshold: 200ms)'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Import test failed:\n{result.stderr}"


def test_ensure_registry_populated_is_idempotent() -> None:
    """Verify that ensure_registry_populated can be called repeatedly idempotently."""
    from neuralops.benchmarks.registry import BENCHMARK_REGISTRY, ensure_registry_populated

    ensure_registry_populated()
    count1 = len(BENCHMARK_REGISTRY)
    ensure_registry_populated()
    count2 = len(BENCHMARK_REGISTRY)
    assert count1 == count2
    assert count1 > 0


def test_get_registry_accessor() -> None:
    """Verify that get_registry() returns the fully populated canonical registry."""
    from neuralops.benchmarks.registry import BenchmarkRegistry, get_registry

    reg1 = get_registry()
    reg2 = BenchmarkRegistry.get_registry()
    assert len(reg1) > 0
    assert reg1 is reg2
