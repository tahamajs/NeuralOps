#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying resolution and graceful fallbacks for core stubs and NotImplementedError
under Wave 8 Agent W8-4 remediation.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core._attention_wrapper import ReplaceAttention
from neuralops.core.composable import (
    BaseExecutionEngine,
    BaseFeatureMap,
    BaseMask,
    BaseProjection,
    ComposableAttention,
)
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.replace_attention import replace_attention_in_model
from neuralops.core.state_cache import benchmark_autoregressive_decoding
from neuralops.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    shard_model_tensor_parallel,
)


def test_state_cache_cpu_fallback_and_device_requirement() -> None:
    """Verify state cache benchmark honors allow_cpu_fallback parameter."""
    if not torch.cuda.is_available():
        # Default fails with honest NotImplementedError
        with pytest.raises(NotImplementedError, match="cannot fabricate latency"):
            benchmark_autoregressive_decoding(seq_lens=(32,))

        # Explicit CPU fallback performs calculation safely
        report = benchmark_autoregressive_decoding(seq_lens=(32,), allow_cpu_fallback=True)
        assert "results" in report
        assert len(report["results"]) == 1
        res = report["results"][0]
        assert res["seq_len"] == 32
        assert res["standard_kv_cache_mb"] > 0
        assert res["neuralops_state_cache_mb"] > 0
        assert res["standard_latency_ms"] > 0
        assert res["neuralops_latency_ms"] > 0
        assert res["speedup_x"] > 0


def test_replace_attention_cross_attention_forward() -> None:
    """Verify ReplaceAttention seamlessly supports cross-attention forward execution."""
    B, N_q, S_kv, D = 2, 8, 16, 64
    attn = ReplaceAttention(hidden_size=D, num_heads=4)
    q_x = torch.randn(B, N_q, D)
    enc_x = torch.randn(B, S_kv, D)

    # 1. Standard cross-attention
    out, weights = attn(hidden_states=q_x, encoder_hidden_states=enc_x)
    assert out.shape == (B, N_q, D)
    assert weights is None

    # 2. Output attention weights enabled
    out_w, weights_w = attn(
        hidden_states=q_x,
        encoder_hidden_states=enc_x,
        output_attentions=True,
    )
    assert out_w.shape == (B, N_q, D)
    assert weights_w is not None
    assert weights_w.shape == (B, 4, N_q, S_kv)

    # 3. Cross-attention with encoder attention mask
    mask = torch.ones(B, S_kv)
    mask[:, :4] = 0
    out_m, _ = attn(
        hidden_states=q_x,
        encoder_hidden_states=enc_x,
        encoder_attention_mask=mask,
    )
    assert out_m.shape == (B, N_q, D)


def test_composable_base_classes_graceful_fallbacks() -> None:
    """Verify BaseProjection, BaseFeatureMap, BaseMask, BaseExecutionEngine provide working fallbacks."""
    B, H, L, D, d = 2, 4, 16, 64, 16

    # BaseProjection: identity passthrough (x, x, x)
    bp = BaseProjection()
    x = torch.randn(B, L, D)
    q, k, v = bp(x)
    assert torch.equal(q, x)
    assert torch.equal(k, x)
    assert torch.equal(v, x)

    # BaseFeatureMap: identity passthrough (q, k)
    bfm = BaseFeatureMap()
    q_f, k_f = bfm(q, k)
    assert torch.equal(q_f, q)
    assert torch.equal(k_f, k)

    # BaseMask: score passthrough
    bm = BaseMask()
    scores = torch.randn(B, H, L, L)
    scores_masked = bm(scores, is_causal=True)
    assert torch.equal(scores_masked, scores)

    # BaseExecutionEngine: scaled dot product attention fallback
    bee = BaseExecutionEngine()
    q_h = torch.randn(B, H, L, d)
    k_h = torch.randn(B, H, L, d)
    v_h = torch.randn(B, H, L, d)
    ctx = bee(q_h, k_h, v_h, is_causal=False)
    assert ctx.shape == (B, H, L, d)


def test_replace_attention_in_model_encoder_decoder_handling() -> None:
    """Verify replace_attention_in_model handles encoder-decoder models gracefully without NotImplementedError."""

    class DummySelfAttn(nn.Module):
        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)

    class DummyBlock(nn.Module):
        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.self_attn = DummySelfAttn(d)

    class DummyT5Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="t5", hidden_size=32, num_attention_heads=4)
            self.encoder = SimpleNamespace(layers=[DummyBlock(32) for _ in range(2)])
            self.decoder = SimpleNamespace(layers=[DummyBlock(32) for _ in range(2)])

    first_key = list(ATTENTION_REGISTRY.keys())[0]
    model = DummyT5Model()
    replaced = replace_attention_in_model(model, first_key)
    assert replaced is model
