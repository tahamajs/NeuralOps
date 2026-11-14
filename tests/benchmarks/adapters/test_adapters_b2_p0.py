#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for benchmark adapters: parameter isolation, input filtering, and device fail-closed."""

from __future__ import annotations

from typing import Any
import pytest
import torch
import torch.nn as nn

from neuralops.benchmarks.adapters.base import BaseBenchmarkAdapter, BenchmarkInputs, NumericsReport
from neuralops.benchmarks.adapters.zeroshot import ZeroShotAdapter
from neuralops.benchmarks.adapters.ruler import RulerAdapter


class MinimalAdapter(BaseBenchmarkAdapter):
    """Concrete minimal adapter to verify BaseBenchmarkAdapter behavior."""

    def capability_check(self) -> dict[str, Any]:
        return {"available": True, "reason": "", "backend": "cpu"}

    def build_inputs(self) -> BenchmarkInputs:
        return BenchmarkInputs(tensors={"x": torch.randn(2, 4)})

    def run_reference(self, inputs: BenchmarkInputs) -> torch.Tensor:
        return inputs.tensors["x"] * 2.0

    def run_candidate(self, inputs: BenchmarkInputs) -> tuple[torch.Tensor, str]:
        return inputs.tensors["x"] * 2.0, "pytorch"

    def validate_numerics(self, reference: torch.Tensor, candidate: torch.Tensor) -> NumericsReport:
        return self._default_validate_numerics(reference, candidate)

    def measure_performance(self, inputs: BenchmarkInputs) -> dict[str, float]:
        return {"latency_ms_mean": 1.0}


def test_base_adapter_device_fail_closed():
    """Verify BaseBenchmarkAdapter fails closed when requested CUDA is unavailable."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="Device 'cuda' was requested but CUDA is unavailable"):
            MinimalAdapter(benchmark_name="test_device_fail", device="cuda")


def test_zeroshot_token_type_ids_filtering():
    """Verify ZeroShotAdapter filters extraneous token_type_ids to prevent model signature mismatch."""
    adapter = ZeroShotAdapter(
        module_name="none",
        module_type="baseline",
        model_name="mock_model",
        num_samples=2,
        device="cpu",
    )

    class MockModelNoTypeIds(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 10)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
            if "token_type_ids" in kwargs:
                raise TypeError("Model does not accept token_type_ids")

            class Out:
                def __init__(self):
                    self.logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 10)

            return Out()

    class MockTokenizer:
        def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "token_type_ids": torch.tensor([[0, 0, 0]]),  # extraneous token_type_ids
            }

    adapter.model = MockModelNoTypeIds()
    adapter.tokenizer = MockTokenizer()

    # Must succeed without TypeError
    logprob = adapter._compute_logprobs("Question:", "Answer")
    assert isinstance(logprob, float)


def test_ruler_adapter_input_filtering():
    """Verify RulerAdapter filters inputs to only input_ids and attention_mask."""
    adapter = RulerAdapter(
        module_name="none",
        module_type="baseline",
        model_name="mock_ruler",
        context_length=64,
        num_samples=2,
        device="cpu",
    )

    class MockGenModel(nn.Module):
        def generate(self, **kwargs):
            assert "token_type_ids" not in kwargs
            assert "input_ids" in kwargs
            assert "attention_mask" in kwargs
            return torch.tensor([[1, 2, 3]])

    class MockTokenizer:
        def __call__(self, text: str, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "token_type_ids": torch.tensor([[0, 0, 0]]),
            }

        def decode(self, tokens, **kwargs):
            return "passcode 123456"

    adapter.model = MockGenModel()
    adapter.tokenizer = MockTokenizer()
    adapter.setup = lambda: None
    adapter.apply_module = lambda: None

    report = adapter.run()
    assert report["eval_metrics"]["num_samples"] == 2
