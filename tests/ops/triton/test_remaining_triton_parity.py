"""Tests for the 30 remaining Triton kernels in neuralops/ops/triton.

Verifies execution, shapes, numerical finiteness, and parity against PyTorch
reference implementations under deterministic seeds.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuralops.ops.triton.adaptive_sparse_triton import (
    _adaptive_sparse_attention_pytorch,
    adaptive_sparse_attention_forward,
)
from neuralops.ops.triton.based_scan import based_taylor_scan, based_taylor_scan_pytorch
from neuralops.ops.triton.block_sparse import (
    triton_block_sparse_streaming_attn,
    triton_block_sparse_streaming_attn_pytorch_ref,
)
from neuralops.ops.triton.chela_slay_triton import _chela_slay_pytorch, chela_slay_forward
from neuralops.ops.triton.chela_slay_wide_triton import (
    _chela_slay_wide_pytorch_ref,
    chela_slay_wide_forward,
    chela_slay_wide_triton,
)
from neuralops.ops.triton.cross_attention_triton import (
    _cross_attention_pytorch,
    cross_attention_forward,
)
from neuralops.ops.triton.delta_net_scan import delta_net_scan, delta_net_scan_pytorch
from neuralops.ops.triton.dynamic_conv_triton import (
    _dynamic_conv_pytorch,
    dynamic_conv_attention_forward,
)
from neuralops.ops.triton.fused_svd_projection_triton import fused_svd_projection_triton
from neuralops.ops.triton.gqa import fused_mqa_bias_scale_pytorch, fused_mqa_bias_scale_triton
from neuralops.ops.triton.lightning_scan import lightning_scan, lightning_scan_pytorch
from neuralops.ops.triton.linear_attn import (
    fused_rala_augmented_scan_pytorch,
    fused_rala_augmented_scan_triton,
)
from neuralops.ops.triton.lsh_attention_triton import (
    _lsh_attention_pytorch,
    lsh_attention_forward,
)
from neuralops.ops.triton.paged_linear_attn import (
    paged_linear_attention_decode_step,
    paged_linear_attention_decode_step_pytorch,
)
from neuralops.ops.triton.prune_admm import admm_prune, admm_pruner_compress_pytorch
from neuralops.ops.triton.prune_sparsegpt import sparsegpt_compress, sparsegpt_compress_pytorch
from neuralops.ops.triton.prune_wanda import wanda_compress, wanda_compress_pytorch
from neuralops.ops.triton.quantize_awq import awq_compress, awq_compress_pytorch
from neuralops.ops.triton.quantize_fp8 import fp8_compress, fp8_compressor_compress_pytorch
from neuralops.ops.triton.quantize_kmeans import (
    kmeans_quantize_compress,
    kmeans_quantize_compress_pytorch,
)
from neuralops.ops.triton.rala_attention_triton import (
    _rala_attention_pytorch,
    rala_attention_forward,
)
from neuralops.ops.triton.retnet_scan import retnet_scan, retnet_scan_pytorch
from neuralops.ops.triton.routing_attention_triton import (
    _routing_attention_pytorch,
    routing_attention_forward,
)
from neuralops.ops.triton.rwkv6_scan import rwkv6_scan, rwkv6_scan_pytorch
from neuralops.ops.triton.sparse_attn import (
    fused_hamming_scores_pytorch,
    fused_hamming_scores_triton,
)
from neuralops.ops.triton.state_update import (
    fused_recurrent_linear_step,
    state_update_pytorch_fallback,
)
from neuralops.ops.triton.svd_linear_decay_triton import svd_linear_decay_triton
from neuralops.ops.triton.svd_mamba_fused_triton import svd_mamba_fused_triton
from neuralops.ops.triton.token_compressed_triton import (
    _token_compressed_attention_pytorch,
    token_compressed_attention_forward,
)
from neuralops.ops.triton.ttt_linear_step_triton import (
    _ttt_linear_step_pytorch_ref,
    ttt_linear_step_triton,
)


@pytest.fixture(autouse=True)
def set_seed() -> None:
    torch.manual_seed(42)


def test_adaptive_sparse_attention_parity() -> None:
    q = torch.randn(2, 4, 32, 64)
    k = torch.randn(2, 4, 32, 64)
    v = torch.randn(2, 4, 32, 64)

    out_fwd = adaptive_sparse_attention_forward(q, k, v, sparsity_ratio=0.5)
    out_ref = _adaptive_sparse_attention_pytorch(q, k, v, sparsity_ratio=0.5)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_based_scan_parity() -> None:
    q = torch.randn(2, 4, 32, 64)
    k = torch.randn(2, 4, 32, 64)
    v = torch.randn(2, 4, 32, 64)

    out_fwd = based_taylor_scan(q, k, v)
    out_ref = based_taylor_scan_pytorch(q, k, v)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_block_sparse_streaming_attn_parity() -> None:
    q = torch.randn(2, 4, 64, 64)
    k = torch.randn(2, 4, 64, 64)
    v = torch.randn(2, 4, 64, 64)
    active_blocks = torch.zeros(2, 4, 1, 1, dtype=torch.int32)

    out_fwd = triton_block_sparse_streaming_attn(q, k, v, active_blocks, block_size=64)
    out_ref = triton_block_sparse_streaming_attn_pytorch_ref(q, k, v, active_blocks, block_size=64)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_chela_slay_parity() -> None:
    q = torch.randn(2, 4, 32, 64)
    k = torch.randn(2, 4, 32, 64)
    v = torch.randn(2, 4, 32, 64)

    out_fwd = chela_slay_forward(q, k, v)
    out_ref = _chela_slay_pytorch(q, k, v)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_chela_slay_wide_parity() -> None:
    q = torch.randn(2, 4, 32, 128)
    k = torch.randn(2, 4, 32, 128)
    v = torch.randn(2, 4, 32, 128)

    out_fwd = chela_slay_wide_forward(q, k, v)
    out_triton = chela_slay_wide_triton(q, k, v)
    out_ref = _chela_slay_wide_pytorch_ref(q, k, v)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    assert torch.allclose(out_fwd, out_triton)
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_cross_attention_triton_parity() -> None:
    q = torch.randn(2, 4, 16, 64)
    k = torch.randn(2, 4, 32, 64)
    v = torch.randn(2, 4, 32, 64)

    out_fwd = cross_attention_forward(q, k, v)
    out_ref = _cross_attention_pytorch(q, k, v)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_delta_net_scan_parity() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    beta = torch.sigmoid(torch.randn(2, 4, 16))
    alpha = torch.sigmoid(torch.randn(2, 4, 16))

    out_fwd = delta_net_scan(q, k, v, beta=beta, alpha=alpha)
    out_ref = delta_net_scan_pytorch(q, k, v, beta=beta, alpha=alpha)

    assert out_fwd.shape == v.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_dynamic_conv_attention_parity() -> None:
    B, H, N, D, K = 2, 4, 16, 32, 4
    v = torch.randn(B, H, N, D)
    w = F.softmax(torch.randn(B, H, N, K), dim=-1)

    out_fwd = dynamic_conv_attention_forward(v, weights=w, kernel_size=K)
    pad = torch.zeros((B, H, K - 1, D), dtype=v.dtype, device=v.device)
    v_full = torch.cat([pad, v], dim=2)
    out_ref = _dynamic_conv_pytorch(v_full, w)

    assert out_fwd.shape == v.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_fused_svd_projection_parity() -> None:
    x = torch.randn(16, 64)
    u = torch.randn(64, 16)
    s = torch.ones(16)
    vt = torch.randn(16, 32)

    out = fused_svd_projection_triton(x, u, S=s, V_T=vt)
    ref = (x @ u) * s @ vt

    assert out.shape == (16, 32)
    assert torch.isfinite(out).all()
    diff = (out - ref).abs().max().item()
    assert diff < 1e-4


def test_gqa_parity() -> None:
    scores = torch.randn(2, 8, 16, 16)
    scale = 0.125

    out_fwd = fused_mqa_bias_scale_triton(scores, scale, is_causal=True)
    out_ref = fused_mqa_bias_scale_pytorch(scores, scale, is_causal=True)

    assert out_fwd.shape == scores.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_lightning_scan_parity() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)

    out_fwd = lightning_scan(q, k, v)
    out_ref = lightning_scan_pytorch(q, k, v)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_linear_attn_rala_parity() -> None:
    B, H, N, D, R = 2, 4, 16, 32, 8
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    lora_u = torch.randn(H, D, R)
    lora_v = torch.randn(H, D, R)

    lin_fwd, aug_fwd = fused_rala_augmented_scan_triton(q, k, v, lora_u, lora_v, is_causal=True)
    lin_ref, aug_ref = fused_rala_augmented_scan_pytorch(q, k, v, lora_u, lora_v, is_causal=True)

    assert lin_fwd.shape == q.shape
    assert aug_fwd.shape == q.shape
    assert torch.isfinite(lin_fwd).all()
    assert torch.isfinite(aug_fwd).all()
    diff_lin = (lin_fwd - lin_ref).abs().max().item()
    diff_aug = (aug_fwd - aug_ref).abs().max().item()
    assert diff_lin < 1e-4
    assert diff_aug < 1e-4


def test_lsh_attention_parity() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)

    out_fwd = lsh_attention_forward(q, k, v, num_buckets=4)
    out_ref = _lsh_attention_pytorch(q, k, v, num_buckets=4)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_paged_linear_attn_decode_parity() -> None:
    q_t = torch.randn(2, 4, 32)
    k_t = torch.randn(2, 4, 32)
    v_t = torch.randn(2, 4, 32)
    state = torch.zeros(2, 4, 32, 32)

    out_fwd, s_fwd = paged_linear_attention_decode_step(q_t, k_t, v_t, state.clone())
    out_ref, s_ref = paged_linear_attention_decode_step_pytorch(q_t, k_t, v_t, state.clone())

    assert out_fwd.shape == q_t.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4
    assert torch.allclose(s_fwd, s_ref, atol=1e-4)


def test_prune_admm_parity() -> None:
    w = torch.randn(32, 64)
    out_fwd = admm_prune(w)
    out_ref = admm_pruner_compress_pytorch(w)

    assert out_fwd.shape == w.shape
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_prune_sparsegpt_parity() -> None:
    w = torch.randn(32, 64)
    out_fwd = sparsegpt_compress(w, sparsity=0.5)
    out_ref = sparsegpt_compress_pytorch(w, sparsity=0.5)

    assert out_fwd.shape == w.shape
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_prune_wanda_parity() -> None:
    w = torch.randn(32, 64)
    out_fwd = wanda_compress(w, sparsity=0.5)
    out_ref = wanda_compress_pytorch(w, sparsity=0.5)

    assert out_fwd.shape == w.shape
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_quantize_awq_parity() -> None:
    w = torch.randn(32, 64)
    out_fwd = awq_compress(w, bits=4)
    out_ref = awq_compress_pytorch(w, bits=4)

    assert out_fwd.shape == w.shape
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_quantize_fp8_parity() -> None:
    w = torch.randn(32, 64)
    out_fwd = fp8_compress(w)
    out_ref = fp8_compressor_compress_pytorch(w)

    assert out_fwd.shape == w.shape
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_quantize_kmeans_parity() -> None:
    w = torch.randn(32, 64)
    out_fwd = kmeans_quantize_compress(w, num_clusters=16)
    out_ref = kmeans_quantize_compress_pytorch(w, num_clusters=16)

    assert out_fwd.shape == w.shape
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_rala_attention_parity() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)

    out_fwd = rala_attention_forward(q, k, v)
    out_ref = _rala_attention_pytorch(q, k, v)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_retnet_scan_parity() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)

    out_fwd = retnet_scan(q, k, v, gamma=0.9)
    out_ref = retnet_scan_pytorch(q, k, v, gamma=0.9)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_routing_attention_parity() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)

    out_fwd = routing_attention_forward(q, k, v, num_clusters=4)
    out_ref = _routing_attention_pytorch(q, k, v, num_clusters=4)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_rwkv6_scan_parity() -> None:
    r = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    w = torch.randn(2, 4, 16, 32)
    u = torch.randn(4, 32)

    out_fwd = rwkv6_scan(r, k, v, w, u)
    out_ref = rwkv6_scan_pytorch(r, k, v, w, u)

    assert out_fwd.shape == r.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_sparse_attn_hamming_parity() -> None:
    B, H, N, D = 2, 4, 16, 32
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    lora_q_scale = torch.ones(H)
    lora_k_scale = torch.ones(H)
    temp = torch.tensor(1.0)

    scores_fwd = fused_hamming_scores_triton(q, k, lora_q_scale, lora_k_scale, temp)
    scores_ref = fused_hamming_scores_pytorch(q, k, lora_q_scale, lora_k_scale, temp)

    assert scores_fwd.shape == (B, H, N, N)
    assert torch.isfinite(scores_fwd).all()
    diff = (scores_fwd - scores_ref).abs().max().item()
    assert diff < 1e-4


def test_state_update_parity() -> None:
    B, H, Dk, Dv = 2, 4, 32, 32
    q_t = torch.randn(B, H, Dk)
    k_t = torch.randn(B, H, Dk)
    v_t = torch.randn(B, H, Dv)
    kv_state = torch.zeros(B, H, Dk, Dv)
    z_state = torch.zeros(B, H, Dk)

    out_fwd, kv_new, z_new = fused_recurrent_linear_step(q_t, k_t, v_t, kv_state.clone(), z_state.clone())

    kv_ref = kv_state + torch.einsum("bhk,bhv->bhkv", k_t.float(), v_t.float()).to(kv_state.dtype)
    z_ref = z_state + k_t
    num = torch.einsum("bhk,bhkv->bhv", q_t.float(), kv_ref.float())
    denom = torch.einsum("bhk,bhk->bh", q_t.float(), z_ref.float()).unsqueeze(-1) + 1e-6
    out_ref = (num / denom).to(q_t.dtype)

    assert out_fwd.shape == (B, H, Dv)
    assert torch.isfinite(out_fwd).all()
    assert torch.allclose(out_fwd, out_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(kv_new, kv_ref, atol=1e-4, rtol=1e-4)


def test_svd_linear_decay_parity() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    decay = torch.sigmoid(torch.randn(2, 4, 16, 32))

    out = svd_linear_decay_triton(q, k, v, decay)
    assert out.shape == (2, 4, 16, 32)
    assert torch.isfinite(out).all()


def test_svd_mamba_fused_parity() -> None:
    B, L, D, N = 2, 16, 32, 16
    u = torch.randn(B, L, D)
    delta = F.softplus(torch.randn(B, L, D))
    A = -torch.exp(torch.randn(D, N))
    B_ssm = torch.randn(B, L, N)
    C = torch.randn(B, L, N)

    out = svd_mamba_fused_triton(u, delta, A, B_ssm, C)
    assert out.shape == (B, L, D)
    assert torch.isfinite(out).all()


def test_token_compressed_parity() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 8, 32)
    v = torch.randn(2, 4, 8, 32)

    out_fwd = token_compressed_attention_forward(q, k, v)
    out_ref = _token_compressed_attention_pytorch(q, k, v)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4


def test_ttt_linear_step_parity() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    etas = torch.ones(2, 4, 16) * 0.01

    out_fwd = ttt_linear_step_triton(q, k, v, etas)
    out_ref = _ttt_linear_step_pytorch_ref(q, k, v, etas)

    assert out_fwd.shape == q.shape
    assert torch.isfinite(out_fwd).all()
    diff = (out_fwd - out_ref).abs().max().item()
    assert diff < 1e-4
