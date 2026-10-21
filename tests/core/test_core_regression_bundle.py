#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression coverage for core attention, cache, and import contracts."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from neuralops.core import _attention_wrapper
from neuralops.core._attention_surgery import heal_replaced_model
from neuralops.core.adapters import UniversalAdapter
from neuralops.core.composable import ComposableAttention
from neuralops.core.state_cache import NeuralOpsStateCache, benchmark_autoregressive_decoding


def test_position_embeddings_are_applied_by_replacement_attention(monkeypatch) -> None:
    calls = []

    def rotary(q, k, cos, sin, **kwargs):
        calls.append((cos, sin, kwargs))
        return q + 1, k + 2

    monkeypatch.setattr(_attention_wrapper, "apply_rotary_pos_emb", rotary)
    layer = _attention_wrapper.ReplaceAttention(
        hidden_size=8, num_heads=2, dropout=0.0, is_decoder=False
    ).eval()
    hidden = torch.randn(1, 3, 8)
    cos = torch.ones(1, 3, 4)
    sin = torch.zeros_like(cos)

    output, _ = layer(hidden, position_embeddings=(cos, sin), position_ids=torch.tensor([[0, 1, 2]]))

    assert output.shape == hidden.shape
    assert len(calls) == 1
    assert calls[0][0] is cos and calls[0][1] is sin
    assert torch.equal(calls[0][2]["position_ids"], torch.tensor([[0, 1, 2]]))


def test_callable_kernel_map_is_accepted_and_used() -> None:
    calls = []

    def feature_map(value: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(value.shape))
        return torch.relu(value)

    layer = ComposableAttention(
        embed_dim=8, num_heads=2, kernel_map=feature_map, mask="none", dropout=0.0
    )
    result = layer(torch.randn(1, 4, 8))

    assert result.shape == (1, 4, 8)
    assert len(calls) == 2
    assert layer.kernel_map_type == "callable"


@pytest.mark.parametrize("mask", ["none", "sliding_window(2)"])
def test_registered_string_feature_map_affects_dense_and_sliding_paths(mask: str) -> None:
    torch.manual_seed(7)
    identity = ComposableAttention(
        embed_dim=8, num_heads=2, kernel_map="softmax", mask=mask, dropout=0.0
    ).eval()
    relu = ComposableAttention(
        embed_dim=8, num_heads=2, kernel_map="relu", mask=mask, dropout=0.0
    ).eval()
    relu.load_state_dict(identity.state_dict())
    hidden = torch.randn(2, 5, 8) - 0.8

    assert not torch.allclose(identity(hidden), relu(hidden))


def test_rectangular_linear_weight_is_not_heuristically_transposed() -> None:
    class RectangularProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(15.0).reshape(3, 5))
            self.bias = nn.Parameter(torch.zeros(3))

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = RectangularProjection()

    weight, _ = UniversalAdapter.get_weight_bias(Attention(), "q")
    assert weight.shape == (3, 5)
    assert torch.equal(weight, torch.arange(15.0).reshape(3, 5))


def test_conv1d_projection_weight_is_transposed_explicitly() -> None:
    class Conv1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(15.0).reshape(3, 5))
            self.bias = nn.Parameter(torch.zeros(5))

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = Conv1D()

    weight, _ = UniversalAdapter.get_weight_bias(Attention(), "q")
    assert weight.shape == (5, 3)
    assert torch.equal(weight, torch.arange(15.0).reshape(3, 5).t())


def test_healing_restores_original_requires_grad_flags() -> None:
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(4, 4)

        def forward(self, x):
            return self.attention(x)

    teacher, student = ToyModel(), ToyModel()
    student.attention.weight.requires_grad_(False)
    student.attention.bias.requires_grad_(True)
    before = [parameter.requires_grad for parameter in student.parameters()]

    result = heal_replaced_model(
        teacher,
        student,
        torch.randn(2, 3, 4),
        steps=1,
        also_ridge_init=False,
        log_to_ledger=False,
    )

    assert result["num_healed_layers"] == 1
    assert [parameter.requires_grad for parameter in student.parameters()] == before


def test_state_cache_reuses_preallocated_window_storage_and_preserves_order() -> None:
    cache = NeuralOpsStateCache()
    keys = torch.arange(5.0).view(1, 1, 5, 1)
    values = keys + 10

    first_k, first_v = cache.update_window_kv(0, keys[..., :3, :], values[..., :3, :], 2, 1)
    backing_ptr = cache.window_kv_states[0][0].data_ptr()
    second_k, second_v = cache.update_window_kv(0, keys[..., 3:, :], values[..., 3:, :], 2, 1)

    assert first_k.flatten().tolist() == [0.0, 1.0, 2.0]
    assert first_v.flatten().tolist() == [10.0, 11.0, 12.0]
    assert second_k.flatten().tolist() == [0.0, 3.0, 4.0]
    assert second_v.flatten().tolist() == [10.0, 13.0, 14.0]
    assert cache.window_kv_states[0][0].data_ptr() == backing_ptr


def test_decode_benchmark_refuses_to_fabricate_cpu_latency() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU-only behavior is not applicable on a CUDA host")
    with pytest.raises(NotImplementedError, match="cannot fabricate latency"):
        benchmark_autoregressive_decoding(seq_lens=(8,))


def test_core_package_does_not_eagerly_import_triton_kernels() -> None:
    code = (
        "import sys, neuralops.core; "
        "assert 'neuralops.core.triton_kernels' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
