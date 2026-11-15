#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression and contract test suite for Universal Adapters (W4-L4)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.layers.bert_adapter import BertAttentionAdapter
from neuralops.layers.qwen_adapter import QwenAttentionAdapter


class _DummyAttentionWithDimChange(nn.Module):
    """Synthetic attention returning a different dimension to test dim projection."""

    def __init__(self, in_dim: int = 64, out_dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return self.proj(x)


class _DummyAttentionWithInternalOutProj(nn.Module):
    """Synthetic attention with internal out_proj matching embed_dim."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return self.out_proj(self.q_proj(x))


def test_bert_adapter_dim_projection_registration() -> None:
    core = _DummyAttentionWithDimChange(in_dim=64, out_dim=32)
    adapter = BertAttentionAdapter(core_attention=core, embed_dim=64, num_heads=4)

    x = torch.randn(2, 8, 64)
    out = adapter(x)
    assert isinstance(out, tuple)
    assert len(out) == 2
    context, attn_weights = out
    assert context.shape == (2, 8, 64)
    assert attn_weights is None
    # Verify _dim_proj was registered as a submodule
    assert hasattr(adapter, "_dim_proj")
    assert isinstance(adapter._dim_proj, nn.Linear)
    assert any(name == "_dim_proj" for name, _ in adapter.named_children())


def test_bert_adapter_single_wo_contract() -> None:
    core = _DummyAttentionWithInternalOutProj(embed_dim=64)
    adapter = BertAttentionAdapter(core_attention=core, embed_dim=64, num_heads=4, bypass_internal_out_proj=True)
    assert isinstance(core.out_proj, nn.Identity)


def test_qwen_adapter_dim_projection_registration() -> None:
    core = _DummyAttentionWithDimChange(in_dim=64, out_dim=32)
    adapter = QwenAttentionAdapter(core_attention=core, embed_dim=64, num_heads=4, num_kv_heads=2)

    x = torch.randn(2, 8, 64)
    out = adapter(x)
    assert isinstance(out, tuple)
    assert len(out) == 3
    context, attn_weights, kv = out
    assert context.shape == (2, 8, 64)
    assert attn_weights is None
    # Verify _dim_proj was registered as a submodule
    assert hasattr(adapter, "_dim_proj")
    assert isinstance(adapter._dim_proj, nn.Linear)
    assert any(name == "_dim_proj" for name, _ in adapter.named_children())


def test_qwen_adapter_single_o_proj_contract() -> None:
    core = _DummyAttentionWithInternalOutProj(embed_dim=64)
    adapter = QwenAttentionAdapter(
        core_attention=core, embed_dim=64, num_heads=4, num_kv_heads=2, bypass_internal_out_proj=True
    )
    assert isinstance(core.out_proj, nn.Identity)
    assert isinstance(adapter.o_proj, nn.Linear)


def test_qwen_adapter_gqa_weight_expansion() -> None:
    class MockQwenSource(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(64, 64, bias=True)
            self.k_proj = nn.Linear(64, 16, bias=True)  # 2 kv heads * 8 head_dim = 16
            self.v_proj = nn.Linear(64, 16, bias=True)
            self.o_proj = nn.Linear(64, 64, bias=False)

    class MockTargetMHA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(64, 64, bias=True)
            self.k_proj = nn.Linear(64, 64, bias=True)  # 8 query heads * 8 head_dim = 64
            self.v_proj = nn.Linear(64, 64, bias=True)
            self.out_proj = nn.Linear(64, 64, bias=False)

        def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            return x

    source = MockQwenSource()
    core = MockTargetMHA()
    adapter = QwenAttentionAdapter(
        core_attention=core,
        embed_dim=64,
        num_heads=8,
        num_kv_heads=2,
        head_dim=8,
        inherit_weights_from=source,
    )
    assert adapter.num_heads == 8
    assert adapter.num_kv_heads == 2
    # Verify expanded k_proj weight has shape (64, 64)
    assert core.k_proj.weight.shape == (64, 64)
    # Check that head 0 was repeated 4 times (8 // 2 = 4)
    h0_src = source.k_proj.weight[:8, :]
    for r in range(4):
        torch.testing.assert_close(core.k_proj.weight[r * 8 : (r + 1) * 8, :], h0_src)
