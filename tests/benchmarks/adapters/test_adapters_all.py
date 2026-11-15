#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for all benchmark adapters verifying kwargs acceptance in __init__
and execute() returning canonical BenchmarkResult instances.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.benchmarks.adapters.attention import AttentionBenchmarkAdapter
from neuralops.benchmarks.adapters.base import BaseBenchmarkAdapter
from neuralops.benchmarks.adapters.classification import ClassificationAdapter
from neuralops.benchmarks.adapters.compressor import CompressorModuleAdapter
from neuralops.benchmarks.adapters.cuda_adapter import CUDAKernelAdapter
from neuralops.benchmarks.adapters.perplexity import PerplexityAdapter
from neuralops.benchmarks.adapters.ruler import RulerAdapter
from neuralops.benchmarks.adapters.triton_adapter import TritonKernelAdapter
from neuralops.benchmarks.adapters.zeroshot import ZeroShotAdapter
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus
from neuralops.core.registry import ATTENTION_REGISTRY


def test_base_adapter_kwargs_acceptance():
    """Verify BaseBenchmarkAdapter and subclasses accept extra kwargs."""

    class DummyAdapter(BaseBenchmarkAdapter):
        def capability_check(self):
            return {"available": True, "reason": "", "backend": "cpu"}

        def build_inputs(self):
            from neuralops.benchmarks.adapters.base import BenchmarkInputs

            return BenchmarkInputs()

        def run_reference(self, inputs):
            return torch.tensor([1.0])

        def run_candidate(self, inputs):
            return torch.tensor([1.0]), "cpu"

        def validate_numerics(self, reference, candidate):
            return self._default_validate_numerics(reference, candidate)

        def measure_performance(self, inputs):
            return {"latency_ms_mean": 1.0}

    adapter = DummyAdapter(
        benchmark_name="dummy",
        custom_kwarg_1="test",
        custom_kwarg_2=42,
    )
    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert result.status == BenchmarkStatus.SUCCESS.value


def test_cuda_adapter_kwargs_and_execute():
    """Verify CUDAKernelAdapter accepts extra kwargs and execute() returns BenchmarkResult."""
    adapter = CUDAKernelAdapter(
        kernel_name="layernorm",
        device="cpu",
        warmup_steps=1,
        measure_steps=2,
        custom_kwarg="extra_val",
    )
    assert adapter.extra_kwargs.get("custom_kwarg") == "extra_val"
    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert result.status == BenchmarkStatus.SUCCESS.value


def test_triton_adapter_kwargs_and_execute():
    """Verify TritonKernelAdapter accepts extra kwargs and execute() returns BenchmarkResult."""
    adapter = TritonKernelAdapter(
        kernel_name="elementwise",
        device="cpu",
        warmup_steps=1,
        measure_steps=2,
        custom_kwarg="extra_val",
    )
    assert adapter.extra_kwargs.get("custom_kwarg") == "extra_val"
    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert result.status == BenchmarkStatus.SUCCESS.value


def test_attention_adapter_kwargs_and_execute():
    """Verify AttentionBenchmarkAdapter accepts extra kwargs and execute() returns BenchmarkResult."""
    test_attn_name = next(iter(ATTENTION_REGISTRY.keys())) if len(ATTENTION_REGISTRY) > 0 else "test_mha"
    adapter = AttentionBenchmarkAdapter(
        attention_name=test_attn_name,
        device="cpu",
        dim=32,
        num_heads=2,
        seq_len=16,
        batch_size=2,
        warmup_steps=1,
        measure_steps=2,
        custom_kwarg="extra_val",
    )
    assert adapter.extra_kwargs.get("custom_kwarg") == "extra_val"
    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert result.status in [s.value for s in BenchmarkStatus]


def test_compressor_adapter_kwargs_and_execute():
    """Verify CompressorModuleAdapter accepts extra kwargs and execute() returns BenchmarkResult."""
    adapter = CompressorModuleAdapter(
        compressor_name="direct_svd",
        device="cpu",
        hidden_size=32,
        rank=4,
        warmup_steps=1,
        measure_steps=2,
        custom_kwarg="extra_val",
    )
    assert adapter.extra_kwargs.get("custom_kwarg") == "extra_val"
    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert result.status == BenchmarkStatus.SUCCESS.value


def test_zeroshot_adapter_execute():
    """Verify ZeroShotAdapter execute() returns canonical BenchmarkResult."""
    adapter = ZeroShotAdapter(
        module_name="none",
        module_type="baseline",
        model_name="mock_model",
        num_samples=2,
        device="cpu",
        custom_kwarg="extra_val",
    )
    assert adapter.extra_kwargs.get("custom_kwarg") == "extra_val"

    class MockModel(nn.Module):
        def forward(self, input_ids, attention_mask=None, **kwargs):
            class Out:
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 10)

            return Out()

    class MockTokenizer:
        def __call__(self, text, return_tensors="pt", **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    adapter.model = MockModel()
    adapter.tokenizer = MockTokenizer()
    adapter.setup = lambda: None
    adapter.apply_module = lambda: None

    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert result.status == BenchmarkStatus.SUCCESS.value
    assert "accuracy" in result.metrics


def test_ruler_adapter_execute():
    """Verify RulerAdapter execute() returns canonical BenchmarkResult."""
    adapter = RulerAdapter(
        module_name="none",
        module_type="baseline",
        model_name="mock_ruler",
        context_length=64,
        num_samples=2,
        device="cpu",
        custom_kwarg="extra_val",
    )
    assert adapter.extra_kwargs.get("custom_kwarg") == "extra_val"

    class MockGenModel(nn.Module):
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def decode(self, tokens, **kwargs):
            return "passcode 123456"

    adapter.model = MockGenModel()
    adapter.tokenizer = MockTokenizer()
    adapter.setup = lambda: None
    adapter.apply_module = lambda: None

    result = adapter.execute()
    assert isinstance(result, BenchmarkResult)
    assert result.status == BenchmarkStatus.SUCCESS.value
    assert "accuracy" in result.metrics
