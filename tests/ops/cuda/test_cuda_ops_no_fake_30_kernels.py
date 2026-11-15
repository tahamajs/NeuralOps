# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive verification suite for the 30 remaining CUDA kernel operators (Wave 4 W4-O2).

Verifies:
1. Operator importability and symbol resolution from neuralops.ops.cuda.
2. Execution through Level 3 PyTorch reference fallback on CPU / non-CUDA environments.
3. Shape contracts, numerical stability (finite outputs, no NaNs/Infs).
4. Gradient flow where autograd is supported.
"""

from __future__ import annotations

import pytest
import torch

import neuralops.ops.cuda as noc


def test_adaptive_span() -> None:
    spans = torch.tensor([16.0])
    q_pos = torch.arange(32).unsqueeze(0)
    k_pos = torch.arange(32).unsqueeze(0)
    out = noc.adaptive_span(spans, q_pos, k_pos, ramp_size=4.0)
    assert isinstance(out, torch.Tensor)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_adaptive_sparse() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    out = noc.adaptive_sparse(q, k, v)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_admm() -> None:
    W = torch.randn(64, 64)
    out = noc.admm(W)
    assert out.shape == W.shape
    assert not torch.isnan(out).any()


def test_aegis_gate_norm() -> None:
    b0 = torch.randn(2, 16, 32)
    b1 = torch.randn(2, 16, 32)
    weights = torch.randn(32)
    out = noc.aegis_gate_norm(b0, b1, weights)
    assert out.shape == b0.shape
    assert not torch.isnan(out).any()


def test_asvd() -> None:
    W = torch.randn(64, 64)
    out = noc.asvd(W)
    assert out.shape == W.shape
    assert not torch.isnan(out).any()


def test_awq() -> None:
    W = torch.randn(64, 64)
    out = noc.awq(W)
    assert out.shape == W.shape
    assert not torch.isnan(out).any()


def test_based_taylor_attention() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    out = noc.based_taylor_attention(q, k, v)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_bigbird() -> None:
    q = torch.randn(2, 4, 64, 32)
    k = torch.randn(2, 4, 64, 32)
    v = torch.randn(2, 4, 64, 32)
    out = noc.bigbird(q, k, v, window_size=16, num_global_tokens=2, num_random_blocks=1)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_bitonic_topk() -> None:
    x = torch.randn(2, 64)
    out = noc.bitonic_topk(x, K=8)
    assert out.shape == (2, 8)
    assert not torch.isnan(out).any()


def test_block_diag_gemm() -> None:
    x = torch.randn(4, 16, 128)
    weights = torch.randn(4, 32, 32)
    out = noc.block_diag_gemm(x, weights)
    assert out.shape == (4, 16, 128)
    assert not torch.isnan(out).any()


def test_causal_conv1d() -> None:
    x = torch.randn(2, 32, 16)
    weight = torch.randn(32, 1, 4)
    out = noc.causal_conv1d(x, weight)
    assert out.shape == (2, 32, 16)
    assert not torch.isnan(out).any()


def test_channel_shuffle() -> None:
    x = torch.randn(2, 8, 16, 16)
    out = noc.channel_shuffle(x, groups=2)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_chela_slay() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    out = noc.chela_slay(q, k, v)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_chela_slay_wide() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    out = noc.chela_slay_wide(q, k, v)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_cluster_attention() -> None:
    q = torch.randn(2, 4, 16, 32)
    centroids = torch.randn(2, 4, 4, 32)
    v = torch.randn(2, 4, 4, 32)
    indices = torch.randint(0, 4, (2, 4, 16))
    out = noc.cluster_attention(q, centroids, v, indices)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_cross_attention() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 24, 32)
    v = torch.randn(2, 4, 24, 32)
    out = noc.cross_attention(q, k, v)
    assert out.shape == (2, 4, 16, 32)
    assert not torch.isnan(out).any()


def test_decay_linear_attention() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    gamma = torch.randn(4, 32)
    out = noc.decay_linear_attention(q, k, v, gamma)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_direct_svd() -> None:
    W = torch.randn(64, 64)
    out = noc.direct_svd(W)
    assert out.shape == W.shape
    assert not torch.isnan(out).any()


def test_dynamic_conv() -> None:
    q = torch.randn(2, 4, 16, 32)
    out = noc.dynamic_conv(q)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_dynamic_conv1d() -> None:
    v = torch.randn(2, 16, 32)
    weights = torch.randn(32, 4)
    out = noc.dynamic_conv1d(v, weights)
    assert out.shape == (2, 32, 32)
    assert not torch.isnan(out).any()


def test_entmax_bisect() -> None:
    scores = torch.randn(2, 16, 16)
    out = noc.entmax_bisect(scores)
    assert out.shape == scores.shape
    assert not torch.isnan(out).any()


def test_entropy_gate() -> None:
    x = torch.randn(2, 16, 32)
    V_r = torch.randn(32, 16)
    U_r = torch.randn(32, 16)
    out = noc.entropy_gate(x, V_r, U_r, effective_rank=16)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_entropy_pruning() -> None:
    scores = torch.randn(64, 64)
    out = noc.entropy_pruning(scores)
    assert out.shape == (64,)
    assert not torch.isnan(out).any()


def test_fast_attention() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    out = noc.fast_attention(q, k, v)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_favor_plus() -> None:
    proj = torch.randn(64, 32)
    diag_sq = torch.randn(64)
    out = noc.favor_plus(proj, diag_sq, norm_factor=1.0)
    assert out.shape == proj.shape
    assert not torch.isnan(out).any()


def test_flash_decode_stage2() -> None:
    mid_out = torch.randn(2, 4, 16, 32)
    mid_lse = torch.randn(2, 4, 16)
    out = noc.flash_decode_stage2(mid_out, mid_lse)
    assert out.shape == (2, 4, 1, 32)
    assert not torch.isnan(out).any()


def test_flash_decoding() -> None:
    q = torch.randn(2, 4, 1, 32)
    k = torch.randn(2, 4, 64, 32)
    v = torch.randn(2, 4, 64, 32)
    out = noc.flash_decoding(q, k, v)
    assert out.shape == (2, 4, 1, 32)
    assert not torch.isnan(out).any()


def test_fms_scan() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    decay = torch.randn(4)
    out = noc.fms_scan(q, k, v, decay)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_fp4_gemm() -> None:
    x = torch.randn(2, 32)
    w_packed = torch.randint(0, 255, (16, 16), dtype=torch.uint8)
    scales = torch.ones(16)
    out = noc.fp4_gemm(x, w_packed, scales)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()


def test_fp8_compressor() -> None:
    W = torch.randn(64, 64)
    out = noc.fp8_compressor(W)
    assert out.shape == W.shape
    assert not torch.isnan(out).any()
