# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for autograd backward fallbacks across CUDA kernels (Wave 8 W8-5).

Verifies that custom CUDA autograd functions provide analytical PyTorch/CPU fallbacks
in backward passes instead of raising NotImplementedError.
"""

from __future__ import annotations

import pytest
import torch

from neuralops.ops.cuda.based_taylor_attention.autograd import based_taylor_attention
from neuralops.ops.cuda.gated_deltanet.autograd import gated_deltanet
from neuralops.ops.cuda.gla_chunk_attn.autograd import gla_chunk_attn
from neuralops.ops.cuda.lightning_attention.autograd import lightning_attention
from neuralops.ops.cuda.retnet_chunk_retention.autograd import retnet_chunk_retention
from neuralops.ops.cuda.rwkv6_state_scan.autograd import rwkv6_state_scan


def test_gla_chunk_attn_autograd_backward() -> None:
    """Test GLA Chunk Attention backward pass produces valid gradients without NotImplementedError."""
    q = torch.randn(2, 4, 8, 16, requires_grad=True)
    k = torch.randn(2, 4, 8, 16, requires_grad=True)
    v = torch.randn(2, 4, 8, 16, requires_grad=True)
    g = torch.randn(2, 4, 8, 8, requires_grad=True)

    out = gla_chunk_attn(q, k, v, g)
    assert out.shape == v.shape
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert g.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()
    assert not torch.isnan(g.grad).any()


def test_based_taylor_attention_autograd_backward() -> None:
    """Test Based Taylor Attention backward pass produces valid gradients without NotImplementedError."""
    q = torch.randn(2, 4, 8, 16, requires_grad=True)
    k = torch.randn(2, 4, 8, 16, requires_grad=True)
    v = torch.randn(2, 4, 8, 16, requires_grad=True)

    out = based_taylor_attention(q, k, v)
    assert out.shape == v.shape
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()


def test_gated_deltanet_autograd_backward() -> None:
    """Test Gated DeltaNet backward pass produces valid gradients without NotImplementedError."""
    q = torch.randn(2, 2, 4, 8, requires_grad=True)
    k = torch.randn(2, 2, 4, 8, requires_grad=True)
    v = torch.randn(2, 2, 4, 8, requires_grad=True)
    beta = torch.randn(2, 2, 4, requires_grad=True)
    g = torch.randn(2, 2, 4, requires_grad=True)

    out = gated_deltanet(q, k, v, beta, g)
    assert out.shape == v.shape
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert beta.grad is not None
    assert g.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()
    assert not torch.isnan(beta.grad).any()
    assert not torch.isnan(g.grad).any()


def test_lightning_attention_autograd_backward() -> None:
    """Test Lightning Attention backward pass produces valid gradients without NotImplementedError."""
    q = torch.randn(2, 4, 8, 16, requires_grad=True)
    k = torch.randn(2, 4, 8, 16, requires_grad=True)
    v = torch.randn(2, 4, 8, 16, requires_grad=True)

    out = lightning_attention(q, k, v)
    assert out.shape == v.shape
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()


def test_rwkv6_state_scan_autograd_backward() -> None:
    """Test RWKV-6 State Scan backward pass produces valid gradients without NotImplementedError."""
    r = torch.randn(2, 2, 4, 8, requires_grad=True)
    k = torch.randn(2, 2, 4, 8, requires_grad=True)
    v = torch.randn(2, 2, 4, 8, requires_grad=True)
    w = torch.randn(2, 2, 4, 8, requires_grad=True)
    u = torch.randn(2, 2, 8, requires_grad=True)

    out = rwkv6_state_scan(r, k, v, w, u)
    assert out.shape == r.shape
    loss = out.sum()
    loss.backward()

    assert r.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert w.grad is not None
    assert u.grad is not None
    assert not torch.isnan(r.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()
    assert not torch.isnan(w.grad).any()
    assert not torch.isnan(u.grad).any()


def test_retnet_chunk_retention_autograd_backward() -> None:
    """Test RetNet Chunk Retention backward pass produces valid gradients without NotImplementedError."""
    q = torch.randn(2, 4, 8, 16, requires_grad=True)
    k = torch.randn(2, 4, 8, 16, requires_grad=True)
    v = torch.randn(2, 4, 8, 16, requires_grad=True)

    out = retnet_chunk_retention(q, k, v)
    assert out.shape == v.shape
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()
