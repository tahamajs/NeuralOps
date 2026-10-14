"""Unit tests for Phase 3 TopologyRouter, padding sanitation, and bidirectional wrapping."""

import torch
import torch.nn as nn
import pytest

from neuralops.surgery.topology_router import (
    extract_binary_token_mask,
    sanitize_kv_for_padding_mask,
    BiDirectionalScanWrapper,
    AsymmetricCrossAttentionRouter,
)
from neuralops.attentions.linear.cosformer import CosformerAttention


def test_extract_binary_token_mask_2d_and_4d():
    # 2D mask (B=2, T=4)
    mask_2d = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64)
    keep_2d = extract_binary_token_mask(mask_2d, seq_len=4)
    assert keep_2d.shape == (2, 1, 4, 1)
    assert keep_2d[0, 0, 3, 0] == 0.0
    assert keep_2d[0, 0, 0, 0] == 1.0

    # 4D additive float mask
    mask_4d = torch.tensor([[[[0.0, 0.0, 0.0, -10000.0]]]])
    keep_4d = extract_binary_token_mask(mask_4d, seq_len=4)
    assert keep_4d.shape == (1, 1, 4, 1)
    assert keep_4d[0, 0, 3, 0] == 0.0
    assert keep_4d[0, 0, 0, 0] == 1.0


def test_padding_sanitization_zeroing():
    B, H, T, D = 2, 2, 4, 16
    k = torch.ones(B, H, T, D)
    v = torch.ones(B, H, T, D)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64)

    k_clean, v_clean = sanitize_kv_for_padding_mask(k, v, mask)
    # Token 3 in batch 0 should be exactly 0.0
    assert torch.all(k_clean[0, :, 3, :] == 0.0)
    assert torch.all(v_clean[0, :, 3, :] == 0.0)
    # Token 0 in batch 0 should remain 1.0
    assert torch.all(k_clean[0, :, 0, :] == 1.0)
    assert torch.all(v_clean[0, :, 0, :] == 1.0)


def test_bidirectional_scan_wrapper():
    cosformer = CosformerAttention(hidden_size=64, num_heads=4)
    bi_wrapper = BiDirectionalScanWrapper(cosformer, d_model=64)

    B, T, D = 2, 8, 64
    x = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.int64)
    mask[0, 6:] = 0

    out = bi_wrapper(x, attention_mask=mask)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (B, T, D)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_asymmetric_cross_attention_router():
    router = AsymmetricCrossAttentionRouter(d_model=64, num_heads=4)
    B, T_q, T_kv, D = 2, 16, 32, 64
    q_hidden = torch.randn(B, T_q, D)
    kv_hidden = torch.randn(B, T_kv, D)

    out, _ = router(q_hidden, encoder_hidden_states=kv_hidden)
    assert out.shape == (B, T_q, D)
