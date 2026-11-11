#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for attention numerics, causal masking, WY scan parity, and gradient flow.

Covers the 8 attention mechanisms across the 6 remediated modules:
  - nha_attention (NativeHybridAttention)
  - rala_attention (RALALinearAttention)
  - gdn2_attention, delta_product_attention, preconditioned_gdn_attention (flagship_fla)
  - gla_attention (GLAttention)
  - halo_regla (HALOReGLAAttention)
  - moe_deltakv_paged (MoEDeltaKVPagedAttention)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from neuralops.core.registry import create_attention
from neuralops.layers.gated.flagship_fla import (
    DeltaProductAttention,
    PreconditionedGDNAttention,
    gdn2_chunk_scan,
    gdn2_recurrent_step,
)
from neuralops.layers.qwen_adapter import QwenAttentionAdapter
from neuralops.layers.sparse.moe_deltakv_paged import simulate_paged_delta_kv

TARGET_KEYS = [
    "nha_attention",
    "rala_attention",
    "gdn2_attention",
    "delta_product_attention",
    "preconditioned_gdn_attention",
    "gla_attention",
    "halo_regla",
    "moe_deltakv_paged",
]


def _qwen_cfg():
    return SimpleNamespace(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        head_dim=64,
        model_type="qwen2",
        is_decoder=True,
    )


def _bert_cfg():
    return SimpleNamespace(
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=12,
        head_dim=64,
        model_type="bert",
        is_decoder=False,
    )


@pytest.mark.parametrize("attn_key", TARGET_KEYS)
@pytest.mark.parametrize("seq_len", [1, 64, 97])
def test_qwen_profile_shapes_and_finiteness(attn_key: str, seq_len: int):
    """Verify Qwen2.5-0.5B GQA (D=896, H=14, H_kv=2) forward+backward across sequence lengths."""
    torch.manual_seed(42 + seq_len)
    core = create_attention(attn_key, dim=896, num_heads=14, config=_qwen_cfg(), is_causal=True)
    adapter = QwenAttentionAdapter(
        core_attention=core,
        embed_dim=896,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        is_causal=True,
    )
    x = torch.randn(2, seq_len, 896, requires_grad=True)
    mask = torch.ones(2, seq_len)

    out, _, _ = adapter(x, attention_mask=mask)
    assert out.shape == (2, seq_len, 896)
    assert torch.isfinite(out).all(), f"{attn_key} produced non-finite output at L={seq_len}"

    (out**2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("attn_key", TARGET_KEYS)
def test_qwen_causal_invariance(attn_key: str):
    """Verify future token mutations do not leak into earlier positions in causal mode."""
    torch.manual_seed(123)
    core = create_attention(attn_key, dim=896, num_heads=14, config=_qwen_cfg(), is_causal=True)
    adapter = QwenAttentionAdapter(
        core_attention=core,
        embed_dim=896,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        is_causal=True,
    )
    adapter.eval()

    # Smooth sequence so delta-KV compression threshold triggers for moe_deltakv_paged
    x = torch.cumsum(torch.randn(2, 128, 896) * 0.05, dim=1)
    x_mut = x.clone()
    x_mut[:, 64:, :] += 10.0
    mask = torch.ones(2, 128)

    with torch.no_grad():
        o1, _, _ = adapter(x, attention_mask=mask)
        o2, _, _ = adapter(x_mut, attention_mask=mask)

    diff = (o1[:, :64, :] - o2[:, :64, :]).abs().max().item()
    assert diff < 1e-5, f"{attn_key} leaked future tokens into prefix: max_diff={diff}"


@pytest.mark.parametrize("attn_key", TARGET_KEYS)
def test_qwen_100pct_gradient_flow_and_10step_convergence(attn_key: str):
    """Verify 100% of active parameters get finite non-zero grads and 10-step loss drops."""
    torch.manual_seed(77)
    core = create_attention(attn_key, dim=896, num_heads=14, config=_qwen_cfg(), is_causal=True)
    adapter = QwenAttentionAdapter(
        core_attention=core,
        embed_dim=896,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        is_causal=True,
    )
    for _, p in adapter.named_parameters():
        p.requires_grad = True
    adapter.train()

    x = torch.randn(2, 64, 896)
    target = torch.randn(2, 64, 896)
    mask = torch.ones(2, 64)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

    losses = []
    for step in range(10):
        opt.zero_grad()
        out, _, _ = adapter(x, attention_mask=mask)
        loss = F.mse_loss(out, target)
        loss.backward()
        if step == 0:
            bad = [
                n
                for n, p in adapter.named_parameters()
                if p.requires_grad
                and (p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum().item() == 0.0)
            ]
            assert not bad, f"{attn_key} has zero/non-finite grads: {bad}"
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"{attn_key} failed 10-step loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f}"


@pytest.mark.parametrize("attn_key", TARGET_KEYS)
def test_bert_profile_with_all_masked_row_stability(attn_key: str):
    """Verify BERT bidirectional forward+backward stays finite even when a batch item is fully masked."""
    torch.manual_seed(99)
    mod = create_attention(attn_key, dim=768, num_heads=12, config=_bert_cfg(), is_causal=False)
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    mask[0, 48:] = 0.0
    mask[1, :] = 0.0  # Extreme edge case: fully padded sequence

    out = mod(x, attention_mask=mask)
    y = out[0] if isinstance(out, tuple) else out
    assert y.shape == (2, 64, 768)
    assert torch.isfinite(y).all(), f"{attn_key} produced non-finite output on all-masked BERT row"

    (y**2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("attn_key", ["nha_attention", "rala_attention", "halo_regla", "moe_deltakv_paged"])
def test_lora_standalone_step0_gradient_flow(attn_key: str):
    """Verify step-0 gradients flow to 100% of unfrozen LoRA/gate parameters with freeze_base=True."""
    torch.manual_seed(55)
    mod = create_attention(attn_key, dim=256, num_heads=4, config=None, is_causal=True)
    x = torch.randn(2, 32, 256, requires_grad=True)
    out = mod(x, attention_mask=torch.ones(2, 32))
    y = out[0] if isinstance(out, tuple) else out
    (y**2).mean().backward()

    bad = [
        n
        for n, p in mod.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum().item() == 0.0)
    ]
    assert not bad, f"{attn_key} standalone trainable params had zero/NaN grads: {bad}"


@pytest.mark.parametrize("attn_key", TARGET_KEYS)
def test_large_activation_stability_under_amp(attn_key: str):
    """Verify Qwen-scale outlier activations under bfloat16/float16 autocast never produce NaN/Inf."""
    torch.manual_seed(314)
    core = create_attention(attn_key, dim=256, num_heads=4, config=None, is_causal=True)
    x = torch.randn(2, 64, 256) * 50.0

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = core(x, attention_mask=torch.ones(2, 64))
        y = out[0] if isinstance(out, tuple) else out
        assert torch.isfinite(y).all(), f"{attn_key} diverged on large activations under autocast"
        loss = y.float().pow(2).mean()
    loss.backward()
    for n, p in core.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{attn_key} had non-finite grad for {n}"


def test_gdn2_wy_chunked_matches_sequential_recurrence():
    """Verify vectorized WY gdn2_chunk_scan matches step-by-step gdn2_recurrent_step."""
    torch.manual_seed(7)
    B, H, L, D = 2, 4, 97, 32
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    alpha = torch.sigmoid(torch.randn(B, H, L) + 2.0)
    be = torch.sigmoid(torch.randn(B, H, L) - 1.0)
    bw = torch.sigmoid(torch.randn(B, H, L) + 1.0)

    y_wy, s_wy = gdn2_chunk_scan(q, k, v, alpha, be, bw)

    S = torch.zeros(B, H, D, D)
    outs = []
    for t in range(L):
        yt, S = gdn2_recurrent_step(
            S,
            q[:, :, t : t + 1],
            k[:, :, t : t + 1],
            v[:, :, t : t + 1],
            alpha[:, :, t : t + 1, None],
            be[:, :, t : t + 1, None],
            bw[:, :, t : t + 1, None],
        )
        outs.append(yt)
    y_seq = torch.cat(outs, dim=2)

    assert (y_wy - y_seq).abs().max().item() < 1e-5
    assert (s_wy - S).abs().max().item() < 1e-5


def test_delta_product_and_pgdn_step_contracts():
    """Verify step() works cleanly without missing attributes for DeltaProduct and PreconditionedGDN."""
    for cls in (DeltaProductAttention, PreconditionedGDNAttention):
        mod = cls(hidden_size=128, num_heads=4)
        x = torch.randn(2, 128)
        y1, st = mod.step(x)
        y2, st = mod.step(x, st)
        assert y1.shape == (2, 128) and y2.shape == (2, 128)
        assert torch.isfinite(y1).all() and torch.isfinite(y2).all()


def test_moe_deltakv_gqa_svd_ste_zero_nan_grads():
    """Verify simulate_paged_delta_kv produces 0 NaN gradients on GQA-repeated smooth inputs."""
    torch.manual_seed(11)
    k_base = torch.randn(1, 2, 32, 64)
    k_rep = k_base.repeat(1, 7, 1, 1).detach().requires_grad_(True)
    v_rep = k_base.repeat(1, 7, 1, 1).detach().requires_grad_(True)
    k_smooth = torch.cumsum(k_rep * 0.01, dim=2)
    v_smooth = torch.cumsum(v_rep * 0.01, dim=2)

    k_hat, v_hat, ek, ev = simulate_paged_delta_kv(
        k_smooth, v_smooth, rank=32, bits=4, max_delta_error=0.9, page_size=16
    )
    assert ek <= 0.9 and ev <= 0.9
    ((k_hat**2).sum() + (v_hat**2).sum()).backward()
    assert torch.isfinite(k_rep.grad).all()
    assert torch.isfinite(v_rep.grad).all()
