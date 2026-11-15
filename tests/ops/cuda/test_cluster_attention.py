#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for cluster_attention CUDA/CPU operator."""

import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.cluster_attention import (
    cluster_attention,
    cluster_attention_ref,
)


def test_cluster_attention_cpu_fallback_shape():
    """Verify cluster_attention runs on CPU fallback producing expected shape."""
    b, h, n, d = 2, 4, 16, 64
    k = 8
    q = torch.randn(b, h, n, d)
    centroids = torch.randn(b, h, k, d)
    v = torch.randn(b, h, k, d)
    indices = torch.randint(0, k, (b, h, n), dtype=torch.int32)

    out = cluster_attention(q, centroids, v, indices)
    assert out.shape == (b, h, n, d)
    assert torch.isfinite(out).all()


def test_cluster_attention_cpu_parity_with_ref():
    """Verify cluster_attention matches cluster_attention_ref within tolerance on CPU."""
    b, h, n, d = 2, 4, 16, 64
    k = 8
    q = torch.randn(b, h, n, d)
    centroids = torch.randn(b, h, k, d)
    v = torch.randn(b, h, k, d)
    indices = torch.randint(0, k, (b, h, n), dtype=torch.int32)

    out = cluster_attention(q, centroids, v, indices)
    ref = cluster_attention_ref(q, centroids, v, indices)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_cluster_attention_autograd_gradients():
    """Verify backward gradient propagation through cluster_attention on CPU."""
    b, h, n, d = 2, 2, 8, 32
    k = 4
    q = torch.randn(b, h, n, d, requires_grad=True)
    centroids = torch.randn(b, h, k, d, requires_grad=True)
    v = torch.randn(b, h, k, d, requires_grad=True)
    indices = torch.randint(0, k, (b, h, n), dtype=torch.int32)

    out = cluster_attention(q, centroids, v, indices)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert centroids.grad is not None and torch.isfinite(centroids.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_cluster_attention_torch_amp_autocast_cpu():
    """Verify compatibility with torch.amp.autocast for CPU execution."""
    b, h, n, d = 2, 2, 8, 32
    k = 4
    q = torch.randn(b, h, n, d)
    centroids = torch.randn(b, h, k, d)
    v = torch.randn(b, h, k, d)
    indices = torch.randint(0, k, (b, h, n), dtype=torch.int32)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = cluster_attention(q, centroids, v, indices)

    assert out.shape == (b, h, n, d)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_cluster_attention_head_dim_exceeding_128_fallback():
    """Verify head dimension D > 128 falls back cleanly to reference without error."""
    b, h, n, d = 2, 2, 8, 160
    k = 4
    q = torch.randn(b, h, n, d)
    centroids = torch.randn(b, h, k, d)
    v = torch.randn(b, h, k, d)
    indices = torch.randint(0, k, (b, h, n), dtype=torch.int32)

    out = cluster_attention(q, centroids, v, indices)
    ref = cluster_attention_ref(q, centroids, v, indices)

    assert out.shape == (b, h, n, d)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_cluster_attention_top_level_noc_export():
    """Verify neuralops.ops.cuda exports cluster_attention properly."""
    assert hasattr(noc, "cluster_attention")
    assert callable(noc.cluster_attention)


@pytest.mark.gpu
def test_cluster_attention_cuda():
    """Verify CUDA execution when GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    b, h, n, d = 2, 4, 16, 64
    k = 8
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    centroids = torch.randn(b, h, k, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, k, d, device="cuda", dtype=torch.float16)
    indices = torch.randint(0, k, (b, h, n), device="cuda", dtype=torch.int32)

    out = cluster_attention(q, centroids, v, indices)
    ref = cluster_attention_ref(q, centroids, v, indices)

    assert out.shape == (b, h, n, d)
    assert (out - ref).abs().max() < 1e-2
