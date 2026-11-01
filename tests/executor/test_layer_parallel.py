#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for LayerParallelExecutor.
"""

from __future__ import annotations

import copy
import time
import pytest
import torch
import torch.nn as nn

from neuralops.distributed.context import DistributedContext
from neuralops.executor.layer_parallel import LayerParallelExecutor


class MockLlama(nn.Module):
    def __init__(self, num_layers: int = 4, dim: int = 16) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return x


class MockGPT2(nn.Module):
    def __init__(self, num_layers: int = 6, dim: int = 16) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer.h:
            x = layer(x)
        return x


class MockBERT(nn.Module):
    def __init__(self, num_layers: int = 8, dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder.layer:
            x = layer(x)
        return x


class MockMamba(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def test_find_layer_list_llama_mock():
    """Verify find_layer_list extracts layers from model.model.layers."""
    m = MockLlama(num_layers=4)
    layers = LayerParallelExecutor.find_layer_list(m)
    assert layers is not None
    assert len(layers) == 4


def test_find_layer_list_gpt2_mock():
    """Verify find_layer_list extracts layers from model.transformer.h."""
    m = MockGPT2(num_layers=6)
    layers = LayerParallelExecutor.find_layer_list(m)
    assert layers is not None
    assert len(layers) == 6


def test_find_layer_list_bert_mock():
    """Verify find_layer_list extracts layers from model.encoder.layer."""
    m = MockBERT(num_layers=8)
    layers = LayerParallelExecutor.find_layer_list(m)
    assert layers is not None
    assert len(layers) == 8


def test_find_layer_list_mamba_returns_none():
    """Verify non-layer-stackable architectures return None."""
    m = MockMamba()
    layers = LayerParallelExecutor.find_layer_list(m)
    assert layers is None


def test_transform_layers_single_gpu_matches_baseline():
    """Verify transform_layers applies transformation in single-device mode."""
    m = MockLlama(num_layers=4, dim=8)
    ctx = DistributedContext(init_if_available=False)
    lexec = LayerParallelExecutor(ctx)
    lexec.world_size = 1

    def double_weight(layer: nn.Module) -> nn.Module:
        with torch.no_grad():
            layer.weight.mul_(2.0)
        return layer

    orig_weights = [l.weight.clone() for l in m.model.layers]
    m_transformed = lexec.transform_layers(m, op_fn=double_weight)

    for idx, layer in enumerate(m_transformed.model.layers):
        assert torch.allclose(layer.weight, orig_weights[idx] * 2.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_transform_layers_2_gpu_merges_correctly():
    """Verify transform_layers splits and merges layer transformations correctly across 2 GPUs."""
    m = MockLlama(num_layers=4, dim=8)
    ctx = DistributedContext(init_if_available=False)
    lexec = LayerParallelExecutor(ctx)
    lexec.world_size = 2

    def add_constant_bias(layer: nn.Module) -> nn.Module:
        with torch.no_grad():
            layer.bias.add_(10.0)
        return layer

    orig_biases = [l.bias.clone() for l in m.model.layers]
    m_transformed = lexec.transform_layers(m, op_fn=add_constant_bias)

    for idx, layer in enumerate(m_transformed.model.layers):
        assert torch.allclose(layer.bias, orig_biases[idx] + 10.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_transplant_layer_parallel_single_vs_multi_output_identical():
    """Verify max difference between 1-GPU and 2-GPU transformed model output is < 1e-5."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(num_hidden_layers=4, hidden_size=64, num_attention_heads=2, intermediate_size=128)
    torch.manual_seed(42)
    model = LlamaForCausalLM(cfg).eval().to("cuda:0")

    inputs = torch.randint(0, 1000, (2, 16), device="cuda:0")

    ctx = DistributedContext(init_if_available=False)

    # 1. Single GPU transplant
    m_single = copy.deepcopy(model)
    lexec_single = LayerParallelExecutor(ctx)
    lexec_single.world_size = 1
    m_single_trans = lexec_single.transplant(m_single, target_attention="full_attention", lora_rank=0, seed=42)
    m_single_trans.eval()

    # 2. 2-GPU Parallel transplant
    m_parallel = copy.deepcopy(model)
    lexec_parallel = LayerParallelExecutor(ctx)
    lexec_parallel.world_size = 2
    m_parallel_trans = lexec_parallel.transplant(m_parallel, target_attention="full_attention", lora_rank=0, seed=42)
    m_parallel_trans.eval()

    with torch.no_grad():
        out_single = m_single_trans(inputs).logits
        out_parallel = m_parallel_trans(inputs).logits

    max_diff = (out_single - out_parallel).abs().max().item()
    assert max_diff < 1e-5, f"Output difference {max_diff} exceeded 1e-5 gate"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_layer_parallel_speedup_2x():
    """Verify 2-GPU layer-parallel execution achieves >= 1.5x speedup on layer transformations."""
    m = nn.Module()
    m.layers = nn.ModuleList([nn.Linear(2048, 2048) for _ in range(8)])

    ctx = DistributedContext(init_if_available=False)
    lexec_single = LayerParallelExecutor(ctx)
    lexec_single.world_size = 1

    lexec_par = LayerParallelExecutor(ctx)
    lexec_par.world_size = 2

    def svd_compress(layer: nn.Module) -> nn.Module:
        W = layer.weight
        U, S, V = torch.linalg.svd(W, full_matrices=False)
        layer.weight.data.copy_(U[:, :128] @ torch.diag(S[:128]) @ V[:128, :])
        return layer

    # Warmup
    _ = lexec_single.transform_layers(copy.deepcopy(m), layer_indices=[0], op_fn=svd_compress)
    _ = lexec_par.transform_layers(copy.deepcopy(m), layer_indices=[0, 1], op_fn=svd_compress)

    # Sequential single-GPU
    m1 = copy.deepcopy(m)
    torch.cuda.synchronize(0)
    t0 = time.perf_counter()
    lexec_single.transform_layers(m1, op_fn=svd_compress)
    torch.cuda.synchronize(0)
    single_time = time.perf_counter() - t0

    # 2-GPU Parallel
    m2 = copy.deepcopy(m)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    t0 = time.perf_counter()
    lexec_par.transform_layers(m2, op_fn=svd_compress)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    parallel_time = time.perf_counter() - t0

    speedup = single_time / max(parallel_time, 1e-9)
    assert speedup >= 1.5, f"Speedup {speedup:.2f}x was below expected 1.5x gate"
