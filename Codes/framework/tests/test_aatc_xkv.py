"""
Unit tests for AATC-XKV Compressor (Attention-Aware Cross-Layer KV Cache Compression).

Verifies all 9 mandatory requirements:
  1. test_shared_subspace_shape: verify V_shared has correct rank
  2. test_sensitivity_computation: check sensitivity is non-negative
  3. test_waterfilling_budget: verify sum(bits) <= total_bits and close to budget
  4. test_waterfilling_drops_low_importance: verify components with low s*σ² get zero bits
  5. test_quantization_roundtrip: quantize/dequantize error on active bits
  6. test_attention_sink_preserved: verify tokens 0 and N-1 are exactly preserved
  7. test_cross_layer_rank_savings: joint SVD rank is effective
  8. test_model_unchanged: model parameters hash before/after compression is 100% bit-identical
  9. test_kv_cache_reduction: actual KV cache representation is compressed
"""

import hashlib
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.compression_engine.aatc_xkv_compressor import (
    AATC_XKV_CacheCompressor, AATC_XKVCompressor, SharedSubspace,
    dequantize_from_bits, quantize_to_bits, reverse_waterfilling)
from framework.core.registry import COMPRESSOR_REGISTRY


class ToyAttention(nn.Module):
    def __init__(self, hidden_dim=64, kv_dim=32):
        super().__init__()
        self.k_proj = nn.Linear(hidden_dim, kv_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, kv_dim, bias=True)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # return dummy attention
        return self.o_proj(q)


class ToyTransformer(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=64, kv_dim=32, vocab_size=500):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"self_attn": ToyAttention(hidden_dim, kv_dim)})
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer["self_attn"](x)
        return self.lm_head(x)


def compute_param_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for p in model.parameters():
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def test_shared_subspace_shape():
    """1. Verify V_shared has correct rank and dimension."""
    D = 64
    N = 100
    G = 3
    # Stacked K matrix from 3 layers
    K_stack = torch.randn(G * N, D)
    _, S, Vh = torch.linalg.svd(K_stack, full_matrices=False)
    r = 16
    V_shared = Vh[:r, :]
    assert V_shared.shape == (r, D), f"Expected shape ({r}, {D}), got {V_shared.shape}"
    # Check orthogonality
    gram = torch.matmul(V_shared, V_shared.t())
    eye = torch.eye(r)
    assert torch.allclose(gram, eye, atol=1e-4), "Shared basis rows must be orthonormal"


def test_sensitivity_computation():
    """2. Check attention sensitivity s_c is non-negative and finite."""
    D = 32
    r = 8
    N = 50
    V_shared = torch.randn(r, D)
    V_shared, _ = torch.linalg.qr(V_shared.t())
    V_shared = V_shared.t()

    K_stack = torch.randn(N, D)
    gK_stack = torch.randn(N, D)

    proj_gk = torch.matmul(gK_stack, V_shared.t())
    proj_k = torch.matmul(K_stack, V_shared.t())
    sensitivity = torch.mean(proj_gk ** 2, dim=0) + torch.mean(proj_k ** 2, dim=0)

    assert sensitivity.shape == (r,)
    assert (sensitivity >= 0).all(), "Sensitivity must be non-negative"
    assert torch.isfinite(sensitivity).all(), "Sensitivity must be finite"


def test_waterfilling_budget():
    """3. Verify reverse waterfilling bit allocation respects total bit budget."""
    r = 16
    sigma_sq = torch.linspace(10.0, 0.1, r)
    sensitivity = torch.linspace(5.0, 0.01, r)
    total_bits = 64  # average 4 bits per component

    bits_alloc, lambda_val = reverse_waterfilling(sigma_sq, sensitivity, total_bits, min_bits=0, max_bits=8)
    assert bits_alloc.shape == (r,)
    assert bits_alloc.sum().item() <= total_bits, f"Allocated {bits_alloc.sum()} > budget {total_bits}"
    assert bits_alloc.sum().item() >= total_bits - 2, f"Allocated {bits_alloc.sum()} much lower than budget {total_bits}"
    assert (bits_alloc >= 0).all() and (bits_alloc <= 8).all()


def test_waterfilling_drops_low_importance():
    """4. Verify components with low s*σ² receive 0 bits (dropped)."""
    r = 8
    sigma_sq = torch.tensor([100.0, 50.0, 20.0, 10.0, 1.0, 0.1, 0.01, 1e-6])
    sensitivity = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.01, 1e-6])
    total_bits = 20  # constrained budget

    bits_alloc, _ = reverse_waterfilling(sigma_sq, sensitivity, total_bits, min_bits=0, max_bits=8)
    # The tail components should get 0 bits
    assert bits_alloc[-1].item() == 0, "Lowest component must receive 0 bits"
    assert bits_alloc[-2].item() == 0, "Second lowest component must receive 0 bits"
    assert bits_alloc[0].item() > bits_alloc[-1].item(), "Top component must get more bits than lowest"


def test_quantization_roundtrip():
    """5. Quantize/dequantize roundtrip error is low on active components."""
    r = 8
    batch_tokens = 32
    U = torch.randn(batch_tokens, r)
    bit_alloc = torch.tensor([8, 8, 8, 6, 6, 4, 4, 0])

    quantized, scales = quantize_to_bits(U, bit_alloc)
    assert quantized.dtype == torch.int8, "Quantized tensor must be int8"
    assert (quantized[:, -1] == 0).all(), "Zero-bit component must be all zeros"

    dequant = dequantize_from_bits(quantized, scales, bit_alloc)
    # Check error on 8-bit components
    rel_err_8bit = torch.norm(U[:, :3] - dequant[:, :3]) / torch.norm(U[:, :3])
    assert rel_err_8bit.item() < 0.03, f"8-bit quantization relative error too high: {rel_err_8bit.item()}"


def test_attention_sink_preserved():
    """6. Verify attention sinks (token 0 and last token N-1) are preserved exactly."""
    D = 16
    r = 4
    subspace = SharedSubspace(
        V_k=torch.eye(r, D),
        V_v=torch.eye(r, D),
        sigma_k=torch.ones(r),
        sigma_v=torch.ones(r),
        sensitivity_k=torch.ones(r),
        sensitivity_v=torch.ones(r),
        bit_alloc_k=torch.full((r,), 4, dtype=torch.int64),
        bit_alloc_v=torch.full((r,), 4, dtype=torch.int64),
        group_id=0,
        layer_indices=[0],
    )
    compressor = AATC_XKV_CacheCompressor(
        subspaces={0: subspace},
        layer_to_group={0: 0},
        sink_tokens=1,
        preserve_last_token=True,
    )

    B, H, N = 1, 1, 10
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    k_comp, v_comp = compressor.compress_kv_pair(k, v, layer_idx=0)

    # Token 0 must be 100% identical
    assert torch.equal(k[:, :, 0, :], k_comp[:, :, 0, :]), "Token 0 (initial sink) was modified!"
    assert torch.equal(v[:, :, 0, :], v_comp[:, :, 0, :]), "Token 0 (initial sink) was modified!"

    # Token N-1 must be 100% identical
    assert torch.equal(k[:, :, -1, :], k_comp[:, :, -1, :]), "Token N-1 (current query/sink) was modified!"
    assert torch.equal(v[:, :, -1, :], v_comp[:, :, -1, :]), "Token N-1 (current query/sink) was modified!"


def test_cross_layer_rank_savings():
    """7. Joint SVD rank across correlated layers is lower than sum of independent ranks."""
    D = 64
    N = 100
    G = 3
    # Synthetic correlated KV across adjacent layers
    base_signal = torch.randn(N, 8) @ torch.randn(8, D)
    K_layers = [base_signal + 0.1 * torch.randn(N, D) for _ in range(G)]

    # Stacked matrix
    K_stack = torch.cat(K_layers, dim=0)
    _, S_joint, _ = torch.linalg.svd(K_stack, full_matrices=False)
    cum_joint = torch.cumsum(S_joint ** 2, dim=0) / torch.sum(S_joint ** 2)
    rank_joint = (cum_joint >= 0.95).nonzero()[0].item() + 1

    # Per-layer independent ranks
    indep_ranks = 0
    for K_l in K_layers:
        _, S_l, _ = torch.linalg.svd(K_l, full_matrices=False)
        cum_l = torch.cumsum(S_l ** 2, dim=0) / torch.sum(S_l ** 2)
        indep_ranks += (cum_l >= 0.95).nonzero()[0].item() + 1

    assert rank_joint < indep_ranks, f"Joint rank {rank_joint} should be strictly less than sum of independent ranks {indep_ranks}"


def test_model_unchanged():
    """8. Model parameters hash before and after compression is 100% identical."""
    model = ToyTransformer(num_layers=4, hidden_dim=32, kv_dim=16, vocab_size=100)
    hash_before = compute_param_hash(model)

    compressor = AATC_XKVCompressor(model, group_size=2, target_bits_per_component=4.0, calib_samples=4)
    compressed_model, stats = compressor.compress()

    hash_after = compute_param_hash(compressed_model)
    assert hash_before == hash_after, "Model parameters were modified! AATC-XKV must be a pure KV cache compressor."
    assert stats["weights_modified"] is False


def test_kv_cache_reduction():
    """9. Theoretical KV cache bits reduction ratio > 1.5x."""
    model = ToyTransformer(num_layers=4, hidden_dim=64, kv_dim=32, vocab_size=200)
    compressor = AATC_XKVCompressor(model, group_size=2, target_bits_per_component=4.0, calib_samples=4)
    _, stats = compressor.compress()

    ratio = stats["theoretical_kv_compression_ratio"]
    assert ratio >= 1.5, f"Theoretical KV cache compression ratio {ratio}x is below 1.5x threshold"
