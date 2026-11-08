#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unified benchmark adapter contract and lifecycle tests (AGT-56 / Roadmap Gap 3).

Verifies that all concrete benchmark adapters (CUDAKernelAdapter, TritonKernelAdapter,
AttentionBenchmarkAdapter, CompressorModuleAdapter) inherit from BaseBenchmarkAdapter,
execute the 6-gate lifecycle, and return canonical BenchmarkResult objects.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.benchmarks.adapters.attention import AttentionBenchmarkAdapter
from neuralops.benchmarks.adapters.base import BaseBenchmarkAdapter
from neuralops.benchmarks.adapters.compressor import CompressorModuleAdapter
from neuralops.benchmarks.adapters.cuda_adapter import CUDAKernelAdapter
from neuralops.benchmarks.adapters.triton_adapter import TritonKernelAdapter
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


# ---------------------------------------------------------------------------
# 1. BaseBenchmarkAdapter Inheritance
# ---------------------------------------------------------------------------


def test_all_adapters_subclass_base_benchmark_adapter():
    """Verify that all four concrete adapters inherit from BaseBenchmarkAdapter."""
    assert issubclass(CUDAKernelAdapter, BaseBenchmarkAdapter)
    assert issubclass(TritonKernelAdapter, BaseBenchmarkAdapter)
    assert issubclass(AttentionBenchmarkAdapter, BaseBenchmarkAdapter)
    assert issubclass(CompressorModuleAdapter, BaseBenchmarkAdapter)


# ---------------------------------------------------------------------------
# 2. Unified BenchmarkResult Return Types
# ---------------------------------------------------------------------------


def test_cuda_kernel_adapter_returns_benchmark_result():
    """Verify CUDAKernelAdapter returns a BenchmarkResult with valid BenchmarkStatus."""
    adapter = CUDAKernelAdapter(kernel_name="cosformer", device="cpu")
    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.status, str)
    assert result.status in [s.value for s in BenchmarkStatus]
    assert "benchmark_name" in result.metadata
    assert result.module["name"] == "cosformer"


def test_triton_kernel_adapter_returns_benchmark_result():
    """Verify TritonKernelAdapter returns a BenchmarkResult with valid BenchmarkStatus."""
    adapter = TritonKernelAdapter(kernel_name="elementwise", device="cpu")
    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.status, str)
    assert result.status in [s.value for s in BenchmarkStatus]
    assert "benchmark_name" in result.metadata
    assert result.module["name"] == "elementwise"


def test_attention_benchmark_adapter_returns_benchmark_result():
    """Verify AttentionBenchmarkAdapter returns a BenchmarkResult with valid BenchmarkStatus."""
    # Register lightweight mock attention if needed or use existing registry module
    test_attn_name = next(iter(ATTENTION_REGISTRY.keys())) if len(ATTENTION_REGISTRY) > 0 else "test_mha"
    adapter = AttentionBenchmarkAdapter(
        attention_name=test_attn_name,
        causal=False,
        dim=32,
        num_heads=2,
        seq_len=16,
        batch_size=2,
        device="cpu",
    )
    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.status, str)
    assert result.status in [s.value for s in BenchmarkStatus]
    assert result.module["name"] == test_attn_name


def test_compressor_module_adapter_returns_benchmark_result():
    """Verify CompressorModuleAdapter returns a BenchmarkResult with valid BenchmarkStatus."""
    adapter = CompressorModuleAdapter(
        compressor_name="direct_svd",
        device="cpu",
        hidden_size=64,
        rank=8,
    )
    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.status, str)
    assert result.status == BenchmarkStatus.SUCCESS.value
    assert "relative_frob_error" in result.metadata
    assert "sqnr_db" in result.metadata


# ---------------------------------------------------------------------------
# 3. Real Batch Calibration & Distortion
# ---------------------------------------------------------------------------


def test_compressor_distortion_real_batch():
    """Assert CompressorModuleAdapter computes Frobenius distortion and SQNR on a real batch."""
    # Create deterministic real input calibration batch: (batch=4, seq=16, hidden=64)
    torch.manual_seed(1234)
    real_batch = torch.randn(4, 16, 64)

    adapter = CompressorModuleAdapter(
        compressor_name="direct_svd",
        device="cpu",
        hidden_size=64,
        rank=8,
        calibration_batch=real_batch,
    )
    result = adapter.execute()

    assert result.status == BenchmarkStatus.SUCCESS.value
    assert result.metadata.get("is_real_calibration") is True
    assert adapter.relative_frob_error >= 0.0
    assert not torch.isnan(torch.tensor(adapter.relative_frob_error))
    assert not torch.isinf(torch.tensor(adapter.relative_frob_error))
    assert "sqnr_db" in result.metadata
    assert adapter.sqnr_db != 0.0


# ---------------------------------------------------------------------------
# 4. Attention Adapter Autograd Gradient Check
# ---------------------------------------------------------------------------


def test_attention_adapter_gradient_check(monkeypatch):
    """Assert AttentionBenchmarkAdapter executes autograd gradient check and certifies non-zero gradients."""
    class ToyAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(32, 32)

        def forward(self, x, **kwargs):
            return self.linear(x)

    # Temporarily register in ATTENTION_REGISTRY
    monkeypatch.setitem(ATTENTION_REGISTRY._dict, "_test_grad_toy_attention", ToyAttention)
    adapter = AttentionBenchmarkAdapter(
        attention_name="_test_grad_toy_attention",
        causal=False,
        dim=32,
        num_heads=2,
        seq_len=16,
        batch_size=2,
        device="cpu",
    )
    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert adapter.gradient_check_passed is True
    assert result.metadata.get("gradient_check_passed") == 1.0 or adapter.gradient_check_passed is True
