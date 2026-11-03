#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for benchmark adapters, Triton/CUDA dispatchers, and pipeline controllers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pytest
import torch
import torch.nn as nn

from neuralops.benchmarks.adapters.cuda_adapter import CUDAKernelAdapter
from neuralops.benchmarks.adapters.model import (
    BaseModelAdapter,
    HuggingFaceModelAdapter,
    SurgerizedModelAdapter,
    UniversalModelAdapter,
)
from neuralops.benchmarks.adapters.triton_adapter import (
    CANONICAL_TRITON_MODULES,
    EXTENSION_TRITON_MODULES,
    TritonKernelAdapter,
)
from neuralops.benchmarks.pipeline import BenchmarkPipeline, SweepController, sanitize_memory
from neuralops.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BaseBenchmark,
    BenchmarkRegistry,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


class MockModel(nn.Module):
    def __init__(self, vocab_size: int = 100, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([MockTransformerLayer(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> Any:
        h = self.embedding(input_ids)
        states = [h]
        for layer in self.layers:
            h = layer(h)
            states.append(h)
        logits = self.head(h)

        class Output:
            def __init__(self, logits: torch.Tensor, hidden_states: Optional[Tuple[torch.Tensor, ...]]) -> None:
                self.logits = logits
                self.hidden_states = hidden_states

        return Output(logits=logits, hidden_states=tuple(states) if output_hidden_states else None)


def test_base_and_hf_model_adapter() -> None:
    model = MockModel(vocab_size=50, hidden_dim=32, num_layers=2)
    adapter = HuggingFaceModelAdapter(model=model, model_name="mock_hf")

    assert adapter.model_name == "mock_hf"
    assert adapter.get_layer_count() == 2
    meta = adapter.get_model_metadata()
    assert meta["num_layers"] == 2
    assert meta["total_params"] > 0

    input_ids = torch.randint(0, 50, (2, 8))
    logits = adapter.get_logits(input_ids)
    assert logits.shape == (2, 8, 50)

    hidden = adapter.get_hidden_states(input_ids, layer_idx=1)
    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape == (2, 8, 32)

    gen = adapter.generate(input_ids, max_new_tokens=4)
    assert gen.shape == (2, 12)


def test_surgerized_model_adapter(tmp_path: Path) -> None:
    model = MockModel()
    manifest = {
        "transplanted_attention": "linear_attention",
        "compressor_key": "svd",
        "calibrated": True,
        "damping": 1e-4,
    }
    adapter = SurgerizedModelAdapter(model=model, model_name="surgerized_test", surgery_manifest=manifest)
    lineage = adapter.get_surgery_lineage()
    assert lineage["is_transplanted"] is True
    assert lineage["transplanted_attention"] == "linear_attention"
    assert lineage["compressor_key"] == "svd"
    assert lineage["calibrated"] is True


def test_universal_model_adapter_factory() -> None:
    model = MockModel()
    adapter = UniversalModelAdapter.from_pretrained(model, model_name="factory_test")
    assert isinstance(adapter, BaseModelAdapter)
    assert adapter.model_name == "factory_test"


def test_triton_adapter_catalogs() -> None:
    assert len(CANONICAL_TRITON_MODULES) == 26
    assert len(EXTENSION_TRITON_MODULES) >= 20
    all_kernels = TritonKernelAdapter.list_all_triton_kernels()
    assert "linear_attn" in all_kernels
    assert "cosformer_triton_kernel" in all_kernels

    # Instantiate adapter on CPU
    adapter = TritonKernelAdapter("linear_attn", is_extension=False, device="cpu")
    assert adapter.kernel_name == "linear_attn"


def test_cuda_adapter_catalogs() -> None:
    modules = CUDAKernelAdapter.list_all_modules()
    assert len(modules) == 110
    kernels = CUDAKernelAdapter.list_all_kernels()
    assert len(kernels) >= 114

    # Adapter instantiation on CPU
    adapter = CUDAKernelAdapter("layernorm", device="cpu")
    assert adapter.kernel_name == "layernorm"


def test_benchmark_registry_and_decorator() -> None:
    @register_benchmark("mock_unit_benchmark", mode="checkpoint", hardware="any", description="Mock test benchmark")
    class MockBench(BaseBenchmark):
        def run(self, model_adapter: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"accuracy": 0.99, "latency_ms": 1.2}

    assert "mock_unit_benchmark" in BENCHMARK_REGISTRY
    bench_cls = get_benchmark("mock_unit_benchmark")
    assert bench_cls is not None
    listed = list_benchmarks(mode="checkpoint")
    assert "mock_unit_benchmark" in listed

    # Duplicate registration raises ValueError
    with pytest.raises(ValueError, match="Conflicting benchmark registration"):
        @register_benchmark("mock_unit_benchmark", mode="checkpoint")
        class AnotherBench(BaseBenchmark):
            def run(self, model_adapter: Any, **kwargs: Any) -> Dict[str, Any]:
                return {}

    # Invalid mode raises ValueError
    with pytest.raises(ValueError, match="Invalid benchmark mode"):
        @register_benchmark("invalid_mode_bench", mode="nonexistent_mode")
        class InvalidModeBench(BaseBenchmark):
            def run(self, model_adapter: Any, **kwargs: Any) -> Dict[str, Any]:
                return {}

    # Verify operational modes
    modes = BenchmarkRegistry.get_operational_modes()
    assert len(modes) == 12
    for mode in modes:
        assert BenchmarkRegistry.is_valid_mode(mode)
    assert not BenchmarkRegistry.is_valid_mode("invalid_mode")

    # Benchmark Pipeline execution
    model = MockModel()
    adapter = UniversalModelAdapter.from_pretrained(model)
    pipeline = BenchmarkPipeline(model_adapter=adapter, task_keys=["mock_unit_benchmark"])
    results = pipeline.run()
    assert "mock_unit_benchmark" in results
    assert results["mock_unit_benchmark"]["accuracy"] == 0.99
    assert "wall_clock_s" in results["mock_unit_benchmark"]


def test_memory_sanitizer() -> None:
    sanitize_memory()  # Should execute cleanly without error


def test_sweep_controller_init() -> None:
    model = MockModel()
    controller = SweepController(base_model_or_adapter=model)
    assert controller.base_adapter is not None
    assert len(ATTENTION_REGISTRY) == 131
    assert len(COMPRESSOR_REGISTRY) == 72
