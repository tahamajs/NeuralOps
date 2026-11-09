#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Robustness and Autograd Tests for Native Sparse Attention (NSA) and LearnedBlockRouter.

Verifies:
  1. Contiguity and numerical resilience of LearnedBlockRouter (cuBLAS error immunity).
  2. Dimension mismatch and dtype consistency handling.
  3. Dynamic sequence length variation across consecutive forward passes.
  4. End-to-end injection into BERT models with forward/backward autograd and optimizer step.
  5. Half-precision / mixed-precision stability.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

from neuralops.api import replace_attention
from neuralops.layers.sparse.nsa_attention import LearnedBlockRouter, NativeSparseAttention


def test_nsa_router_contiguity_and_guards() -> None:
    """Verify LearnedBlockRouter handles non-contiguous, sliced, and unaligned tensors safely."""
    router = LearnedBlockRouter(d=768, num_blocks=4, router_hidden=64)

    # 1. Non-contiguous sliced 3D tensor
    q_orig = torch.randn(4, 32, 1536)
    q_noncontig = q_orig[:, :, :768].permute(1, 0, 2).permute(1, 0, 2)
    assert not q_noncontig.is_contiguous()

    logits, selected = router(q_noncontig, top_k=2)
    assert logits.shape == (4, 4)
    assert selected.shape == (4, 2)
    assert torch.isfinite(logits).all()

    # 2. 2D tensor
    q_2d = torch.randn(4, 768)
    logits_2d, selected_2d = router(q_2d, top_k=3)
    assert logits_2d.shape == (4, 4)
    assert selected_2d.shape == (4, 3)

    # 3. Dimension mismatch (smaller or larger feature dim)
    q_small = torch.randn(2, 16, 512)
    logits_small, selected_small = router(q_small, top_k=2)
    assert logits_small.shape == (2, 4)
    assert selected_small.shape == (2, 2)

    # 4. NaNs / Infs in input
    q_nan = torch.randn(2, 16, 768)
    q_nan[0, 0, 0] = float("nan")
    q_nan[1, 1, 1] = float("inf")
    logits_nan, selected_nan = router(q_nan, top_k=2)
    assert torch.isfinite(logits_nan).all()
    assert (selected_nan >= 0).all() and (selected_nan < 4).all()

    # 5. Dynamic num_blocks slicing/padding
    logits_sliced, selected_sliced = router(q_2d, top_k=2, num_blocks=2)
    assert logits_sliced.shape == (4, 2)
    assert selected_sliced.shape == (4, 2)
    assert (selected_sliced >= 0).all() and (selected_sliced < 2).all()


def test_nsa_standalone_forward_backward() -> None:
    """Verify NativeSparseAttention executes forward and backward passes with valid gradients."""
    attn = NativeSparseAttention(
        hidden_size=768,
        num_heads=12,
        window_size=32,
        top_k=8,
        block_size=64,
        lora_rank=16,
    )

    B, L, D = 2, 128, 768
    x = torch.randn(B, L, D, requires_grad=True)

    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert torch.isfinite(out).all()

    # Backward autograd check
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert attn.q_proj.lora_A.grad is not None
    assert torch.isfinite(attn.q_proj.lora_A.grad).all()


def test_nsa_variable_sequence_lengths() -> None:
    """Verify NativeSparseAttention dynamically handles variable sequence lengths without error."""
    attn = NativeSparseAttention(
        hidden_size=768,
        num_heads=12,
        window_size=32,
        top_k=4,
        block_size=64,
    )

    # Test variable sequence lengths spanning 1 to 4 blocks
    for seq_len in [32, 64, 128, 256]:
        x = torch.randn(2, seq_len, 768)
        out, _ = attn(x)
        assert out.shape == (2, seq_len, 768)
        assert torch.isfinite(out).all()


def test_nsa_bert_injection_forward_backward_distill() -> None:
    """Verify BERT model injection, distillation forward/backward pass, and optimizer step."""
    cfg = BertConfig(
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        num_hidden_layers=2,
        num_labels=2,
        max_position_embeddings=512,
    )
    model = BertForSequenceClassification(cfg)

    # Inject NSA into all attention layers
    model = replace_attention(model, attention="nsa_attention", block_size=32, top_k=4)

    # Verify that layers are replaced
    assert hasattr(model.bert.encoder.layer[0].attention.self, "is_nested_encoder")

    # Set up optimizer similar to bert_zoo_train_cached.py
    for name, p in model.named_parameters():
        p.requires_grad = ("attention" in name) or ("LayerNorm" in name)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    # Run multiple training steps simulating distillation
    model.train()
    for step in range(3):
        b_in = torch.randint(0, 1000, (4, 64))
        b_att = torch.ones(4, 64, dtype=torch.long)
        b_lbl = torch.randint(0, 2, (4,))

        out = model(input_ids=b_in, attention_mask=b_att)
        loss = nn.functional.cross_entropy(out.logits, b_lbl)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        assert torch.isfinite(loss)
        assert torch.isfinite(out.logits).all()


def test_nsa_bfloat16_precision() -> None:
    """Verify NativeSparseAttention functions correctly under bfloat16 / half precision."""
    attn = NativeSparseAttention(
        hidden_size=256,
        num_heads=4,
        window_size=16,
        top_k=2,
        block_size=32,
    ).to(dtype=torch.bfloat16)

    x = torch.randn(2, 64, 256, dtype=torch.bfloat16, requires_grad=True)
    out, _ = attn(x)

    assert out.shape == (2, 64, 256)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

    loss = out.float().sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
