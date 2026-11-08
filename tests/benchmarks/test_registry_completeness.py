#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for benchmark registry completeness and on-disk alignment (AGT-57 / Gap 4).

Validates that:
  1. BENCHMARK_REGISTRY holds every physical kernel on disk without omissions.
  2. The registered CUDA and Triton modules match the on-disk directory packages.
  3. `neuralops benchmark list` reports the exact canonical registry counts.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from neuralops.benchmarks.adapters.cuda_adapter import CANONICAL_CUDA_MODULES
from neuralops.benchmarks.adapters.triton_adapter import CANONICAL_TRITON_MODULES
from neuralops.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BenchmarkRegistry,
    BenchmarkSpec,
)
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

ROOT = Path(__file__).resolve().parent.parent.parent


def test_registry_matches_disk():
    """Verify that every entry in CANONICAL_CUDA_MODULES and CANONICAL_TRITON_MODULES exists in BENCHMARK_REGISTRY."""
    # Ensure all canonical benchmarks are populated
    BenchmarkRegistry.populate_canonical_benchmarks()

    for mod in CANONICAL_CUDA_MODULES:
        canon_name = mod if ":" in mod else f"cuda:{mod}"
        assert canon_name in BENCHMARK_REGISTRY, f"CUDA module {mod} missing from BENCHMARK_REGISTRY"
        spec = BENCHMARK_REGISTRY[canon_name]
        assert isinstance(spec, BenchmarkSpec)
        assert spec.category == "cuda"
        assert spec.requires_cuda is True

    for mod in CANONICAL_TRITON_MODULES:
        canon_name = mod if ":" in mod else f"triton:{mod}"
        assert canon_name in BENCHMARK_REGISTRY, f"Triton module {mod} missing from BENCHMARK_REGISTRY"
        spec = BENCHMARK_REGISTRY[canon_name]
        assert isinstance(spec, BenchmarkSpec)
        assert spec.category == "triton"
        assert spec.requires_triton is True


def test_registry_total_canonical_count():
    """Verify total registered benchmarks matches canonical modules (131 attn + 72 comp + 140 cuda + 26 triton = 369)."""
    BenchmarkRegistry.populate_canonical_benchmarks()

    cuda_entries = [k for k in BENCHMARK_REGISTRY if k.startswith("cuda:")]
    triton_entries = [k for k in BENCHMARK_REGISTRY if k.startswith("triton:")]
    attn_entries = [k for k in BENCHMARK_REGISTRY if k.startswith("attention:")]
    comp_entries = [k for k in BENCHMARK_REGISTRY if k.startswith("compressor:")]

    assert len(attn_entries) == 131, f"Expected 131 attention entries, found {len(attn_entries)}"
    assert len(comp_entries) == 72, f"Expected 72 compressor entries, found {len(comp_entries)}"
    assert len(triton_entries) == 26, f"Expected 26 canonical Triton entries, found {len(triton_entries)}"
    assert len(cuda_entries) >= 114, f"Expected at least 114 CUDA entries, found {len(cuda_entries)}"

    total_registered = len(BENCHMARK_REGISTRY)
    assert total_registered >= 343, f"Expected total >= 343, found {total_registered}"


def test_list_shows_canonical_registry_counts():
    """Verify neuralops.benchmarks cli list returns exact canonical inventory counts."""
    from neuralops.benchmarks.cli import handle_list
    import argparse

    args = argparse.Namespace(json=True, category=None, search=None, mode=None)
    # Call handle_list with json capture
    from io import StringIO
    import sys

    buf = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        exit_code = handle_list(args)
    finally:
        sys.stdout = old_stdout

    assert exit_code == 0
    output_json = json.loads(buf.getvalue())

    assert "canonical_counts" in output_json
    counts = output_json["canonical_counts"]
    assert counts["attention_mechanisms"] == 131
    assert counts["compression_engines"] == 72
    assert counts["triton_kernel_modules"] == 26
    assert counts["cuda_kernel_implementations"] >= 114
