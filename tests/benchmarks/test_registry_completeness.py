#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Benchmark registry completeness verification suite (AGT-57 / Gap 4).

Verifies that:
  1. Every physical CUDA and Triton kernel module on disk maps to a valid BENCHMARK_REGISTRY entry.
  2. BENCHMARK_REGISTRY houses the full physical canonical inventory (131 attentions + 72 compressors + 140 CUDA + 26 Triton = 369 benchmarks).
  3. Every BenchmarkSpec carries canonical names, hardware requirement flags, and default timeout.
  4. 'neuralops benchmark list' CLI output matches BENCHMARK_REGISTRY count exactly.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from neuralops.benchmarks.adapters.cuda_adapter import CANONICAL_CUDA_MODULES
from neuralops.benchmarks.adapters.triton_adapter import CANONICAL_TRITON_MODULES
from neuralops.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    _BENCHMARK_METADATA,
    BenchmarkRegistry,
    BenchmarkSpec,
)
from neuralops.cli.list_cmd import cmd_list


def test_registry_matches_disk():
    """Verify that every on-disk CUDA and Triton physical module maps to a BENCHMARK_REGISTRY entry."""
    repo_root = Path(__file__).resolve().parent.parent.parent

    # 1. Audit on-disk CUDA packages
    cuda_dir = repo_root / "neuralops" / "ops" / "cuda"
    on_disk_cuda_dirs = {
        d.name
        for d in cuda_dir.iterdir()
        if d.is_dir() and not d.name.startswith(("_", ".")) and d.name != "plugins"
    }
    registered_cuda = {
        spec.target_name
        for spec in BENCHMARK_REGISTRY.values()
        if isinstance(spec, BenchmarkSpec) and spec.category == "cuda"
    }
    # All on-disk CUDA module directories must be registered
    missing_cuda = on_disk_cuda_dirs - registered_cuda
    assert not missing_cuda, f"CUDA modules on disk missing from BENCHMARK_REGISTRY: {missing_cuda}"

    # 2. Audit canonical Triton modules
    triton_dir = repo_root / "neuralops" / "core" / "triton_kernels"
    registered_triton = {
        spec.target_name
        for spec in BENCHMARK_REGISTRY.values()
        if isinstance(spec, BenchmarkSpec) and spec.category == "triton"
    }
    assert len(registered_triton) == 26, f"Expected 26 canonical Triton modules, got {len(registered_triton)}"
    for mod in CANONICAL_TRITON_MODULES:
        assert mod in registered_triton, f"Canonical Triton module '{mod}' missing from registry"


def test_registry_total_canonical():
    """Verify BENCHMARK_REGISTRY total count and categorical breakdown."""
    attns = [k for k in BENCHMARK_REGISTRY if k.startswith("attention:")]
    comps = [k for k in BENCHMARK_REGISTRY if k.startswith("compressor:")]
    cudas = [k for k in BENCHMARK_REGISTRY if k.startswith("cuda:")]
    tritons = [k for k in BENCHMARK_REGISTRY if k.startswith("triton:")]

    assert len(attns) == 131, f"Expected 131 attentions, got {len(attns)}"
    assert len(comps) == 72, f"Expected 72 compressors, got {len(comps)}"
    assert len(tritons) == 26, f"Expected 26 Triton modules, got {len(tritons)}"
    # On disk we have 139 CUDA modules + 1 verified op in OPS_REGISTRY = 140 CUDA benchmarks
    assert len(cudas) == 140, f"Expected 140 CUDA modules, got {len(cudas)}"

    total_expected = 131 + 72 + 140 + 26  # 369
    assert len(BENCHMARK_REGISTRY) == total_expected, (
        f"Expected {total_expected} total benchmarks, got {len(BENCHMARK_REGISTRY)}"
    )


def test_benchmark_spec_requirements_and_timeout():
    """Verify BenchmarkSpec contracts for CUDA and Triton modules."""
    for key, spec in BENCHMARK_REGISTRY.items():
        assert isinstance(spec, BenchmarkSpec), f"Entry '{key}' is not a BenchmarkSpec"
        assert spec.canonical_name == key
        assert spec.default_timeout_s > 0
        if spec.category == "cuda":
            assert spec.requires_cuda is True
        elif spec.category == "triton":
            assert spec.requires_triton is True


def test_list_shows_registry_count():
    """Verify that 'neuralops benchmark list --json' reports the exact BENCHMARK_REGISTRY count."""
    args = argparse.Namespace(
        type="all",
        family=None,
        backend="all",
        supports=None,
        json=True,
    )

    captured_stdout = io.StringIO()
    with patch("sys.stdout", captured_stdout):
        exit_code = cmd_list(args)

    assert exit_code == 0
    raw_output = captured_stdout.getvalue()
    # Strip any potential leading non-json logging lines (like setuptools warnings)
    json_start = raw_output.find("{")
    assert json_start != -1, "No JSON object found in output"
    payload = json.loads(raw_output[json_start:])

    assert payload["total_count"] == len(BENCHMARK_REGISTRY)
    assert len(payload["benchmarks"]) == len(BENCHMARK_REGISTRY)
    assert payload["canonical_counts"]["total_benchmarks"] == len(BENCHMARK_REGISTRY)
    assert payload["canonical_counts"]["attention_mechanisms"] == 131
    assert payload["canonical_counts"]["compression_engines"] == 72
    assert payload["canonical_counts"]["triton_kernel_modules"] == 26
    assert payload["canonical_counts"]["cuda_kernel_implementations"] == 140
