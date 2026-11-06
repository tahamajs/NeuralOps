#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for the canonical unified benchmark registry and status integrity schema."""

import pytest

from neuralops.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BenchmarkRegistry,
    BenchmarkSpec,
    list_benchmarks,
)
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus


def test_benchmark_status_normalization():
    """Verify legacy strings and abbreviations normalize to canonical BenchmarkStatus."""
    assert BenchmarkStatus.normalize("PASS") == BenchmarkStatus.SUCCESS
    assert BenchmarkStatus.normalize("passed") == BenchmarkStatus.SUCCESS
    assert BenchmarkStatus.normalize("ok") == BenchmarkStatus.SUCCESS
    assert BenchmarkStatus.normalize("true") == BenchmarkStatus.SUCCESS
    assert BenchmarkStatus.normalize("success") == BenchmarkStatus.SUCCESS

    assert BenchmarkStatus.normalize("FAIL") == BenchmarkStatus.FAILED
    assert BenchmarkStatus.normalize("failed") == BenchmarkStatus.FAILED
    assert BenchmarkStatus.normalize("error") == BenchmarkStatus.FAILED

    assert BenchmarkStatus.normalize("timeout") == BenchmarkStatus.TIMEOUT
    assert BenchmarkStatus.normalize("timed_out") == BenchmarkStatus.TIMEOUT

    assert BenchmarkStatus.normalize("fail_delta") == BenchmarkStatus.FAIL_NUMERICS
    assert BenchmarkStatus.normalize("numerical_error") == BenchmarkStatus.FAIL_NUMERICS

    assert BenchmarkStatus.normalize("cuda_error") == BenchmarkStatus.HARDWARE_ERROR
    assert BenchmarkStatus.normalize("missing_dependency") == BenchmarkStatus.DEPENDENCY_ERROR
    assert BenchmarkStatus.normalize("deferred") == BenchmarkStatus.DEFERRED
    assert BenchmarkStatus.normalize("not_implemented") == BenchmarkStatus.NOT_IMPLEMENTED


def test_benchmark_result_enforces_canonical_status():
    """Verify BenchmarkResult __post_init__ normalizes any input status."""
    res = BenchmarkResult(status="PASS")
    assert res.status == "success"

    res_numerics = BenchmarkResult(status="fail_delta")
    assert res_numerics.status == "fail_numerics"


def test_registry_canonical_invariants():
    """Verify unified registry houses exactly 131 attentions, 72 compressors, and ops."""
    attns = [k for k in BENCHMARK_REGISTRY if k.startswith("attention:")]
    comps = [k for k in BENCHMARK_REGISTRY if k.startswith("compressor:")]
    cudas = [k for k in BENCHMARK_REGISTRY if k.startswith("cuda:")]
    tritons = [k for k in BENCHMARK_REGISTRY if k.startswith("triton:")]

    assert len(attns) == 131, f"Expected 131 attentions, got {len(attns)}"
    assert len(comps) == 72, f"Expected 72 compressors, got {len(comps)}"
    assert len(cudas) >= 4, f"Expected at least 4 CUDA ops, got {len(cudas)}"
    assert len(tritons) >= 1, f"Expected at least 1 Triton op, got {len(tritons)}"


def test_registry_benchmark_spec_contracts():
    """Verify BenchmarkSpec contracts for representative modules."""
    spec_attn = BENCHMARK_REGISTRY.get("attention:cosformer")
    assert isinstance(spec_attn, BenchmarkSpec)
    assert spec_attn.canonical_name == "attention:cosformer"
    assert spec_attn.category == "attention"
    assert spec_attn.target_name == "cosformer"
    assert spec_attn.supports_causal is True

    spec_comp = BENCHMARK_REGISTRY.get("compressor:direct_svd")
    assert isinstance(spec_comp, BenchmarkSpec)
    assert spec_comp.canonical_name == "compressor:direct_svd"
    assert spec_comp.category == "compressor"
    assert spec_comp.target_name == "direct_svd"
