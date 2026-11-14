#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for lazy benchmark registry population and import latency verification."""

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


def test_registry_import_latency():
    """Assert that importing neuralops.benchmarks.registry takes < 200ms in a fresh process."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import time\n"
            "t0 = time.perf_counter()\n"
            "import neuralops.benchmarks.registry as reg\n"
            "t_import = time.perf_counter() - t0\n"
            "assert not reg._REGISTRY_POPULATED, 'Registry populated prematurely on import'\n"
            "print(f'{t_import:.6f}')\n"
        ),
    ]
    samples = []
    for _ in range(3):
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        samples.append(float(proc.stdout.strip()))
    min_import_time = min(samples)
    assert min_import_time < 0.200, (
        f"Module import took {min_import_time:.4f}s minimum (samples: {samples}), expected < 0.200s"
    )


def test_registry_lazy_access():
    """Verify that accessing BENCHMARK_REGISTRY triggers lazy population across dict operations."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import neuralops.benchmarks.registry as reg\n"
            "assert not reg._REGISTRY_POPULATED\n"
            "# 1. Test contains operator\n"
            "assert 'attention:full_attention' in reg.BENCHMARK_REGISTRY\n"
            "assert reg._REGISTRY_POPULATED\n"
            "# 2. Test len\n"
            "assert len(reg.BENCHMARK_REGISTRY) >= 369\n"
            "# 3. Test getitem\n"
            "spec = reg.BENCHMARK_REGISTRY['attention:full_attention']\n"
            "assert spec.canonical_name == 'attention:full_attention'\n"
            "# 4. Test get\n"
            "assert reg.BENCHMARK_REGISTRY.get('nonexistent') is None\n"
            "# 5. Test keys, values, items\n"
            "assert len(list(reg.BENCHMARK_REGISTRY.keys())) >= 369\n"
            "assert len(list(reg.BENCHMARK_REGISTRY.values())) >= 369\n"
            "assert len(list(reg.BENCHMARK_REGISTRY.items())) >= 369\n"
            "# 6. Test functions\n"
            "assert reg.get_benchmark('attention:full_attention') is not None\n"
            "assert len(reg.list_benchmarks()) >= 369\n"
            "assert len(reg.get_registry()) >= 369\n"
            "assert reg.ensure_populated() >= 369\n"
        ),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def test_registry_api_in_process():
    """Test registry accessor methods directly within the test process."""
    from neuralops.benchmarks.registry import (
        BENCHMARK_REGISTRY,
        ensure_populated,
        get_benchmark,
        get_registry,
        list_benchmarks,
    )

    count = ensure_populated()
    assert count >= 369

    reg = get_registry()
    assert "attention:full_attention" in reg

    bm = get_benchmark("attention:full_attention")
    assert bm is not None

    bms = list_benchmarks()
    assert len(bms) >= 369
    assert "attention:full_attention" in bms

    # Subscript and len checks on BENCHMARK_REGISTRY
    assert len(BENCHMARK_REGISTRY) >= 369
    assert BENCHMARK_REGISTRY["attention:full_attention"] is not None
