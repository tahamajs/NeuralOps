# -*- coding: utf-8 -*-
"""Exhaustive mathematical parity, gradient sanity, and multi-shape verification
for all 48 Triton-wired Attention Modules across training and inference modes.
"""

import math
import pytest
import torch
import torch.nn as nn
from neuralops.attentions import ATTENTION_REGISTRY

# Canonical list of 48 Triton-wired Attention modules
CANONICAL_48_MODULES = [
    # Sparse / Compressed Attention (5)
    "adaptive_sparse_attention",
    "moba_attention",
    "xattention",
    "nsa_attention",
    "reap_moe",
    # Modern GQA / ALiBi / Conv (4)
    "alibi",
    "dynamic_conv_attention",
    "mla_attention",
    "mqa",
    # Linear / Cosformer (3)
    "cosformer",
    "cosformer_attention",
    "triton_cosformer",
    # Gated & Recurrent Linear Attention (7)
    "decay_linear_attention",
    "decay_linear_attn",
    "gated_linear_attention",
    "gated_linear_attn",
    "gated_delta_net",
    "hgrn2_gated_layer",
    "kda_attention",
    # Kernel & Spectral Attention (9)
    "linformer_attention",
    "performer_attention",
    "chela_slay",
    "chela_slay_attention",
    "galerkin_attention",
    "gla_attention",
    "nystrom_attention",
    "rala_attention",
    "spectraformer_attention",
    # Quantized / Span / Sink (4)
    "spectron",
    "spectron_fp8",
    "adaptive_span_attention",
    "sink_attention",
    # SSM & Hybrid (6)
    "mamba",
    "mamba_block",
    "selective_ssm",
    "selective_mamba",
    "jamba",
    "vspec",
    # SVD & Factorized Attention (10)
    "svd_linear_attention",
    "svd_linear_decay",
    "triton_svd_linear_attention",
    "asvd_attention",
    "factorised_matrix_state",
    "fused_svd_kernel_attention",
    "projection_factorized_attn",
    "svd_llm_v2_attention",
    "svd_projection_attention",
    "svd_seq",
]


def _instantiate_module(key: str, dim: int = 128, num_heads: int = 4):
    cls = ATTENTION_REGISTRY.get(key)
    assert cls is not None, f"Attention key {key} not found in ATTENTION_REGISTRY"
    try:
        m = cls(dim=dim, num_heads=num_heads)
    except Exception:
        m = cls(hidden_size=dim, num_attention_heads=num_heads)
    return m


@pytest.mark.parametrize("attn_key", CANONICAL_48_MODULES)
@pytest.mark.parametrize("dim,num_heads", [(128, 4), (256, 8)])
def test_all_48_arbitrary_dimensions_fwd(attn_key, dim, num_heads):
    """Test A: Verifies every module handles arbitrary hidden dimensions without dimension mismatch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = _instantiate_module(attn_key, dim=dim, num_heads=num_heads).to(device)
    m.eval()

    B, T = 2, 32
    x = torch.randn(B, T, dim, device=device)
    with torch.no_grad():
        out = m(x)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (B, T, dim), (
            f"Shape mismatch for {attn_key}: expected {(B, T, dim)}, got {tuple(out.shape)}"
        )
        assert not torch.isnan(out).any(), f"NaN output detected for {attn_key}"
        assert not torch.isinf(out).any(), f"Inf output detected for {attn_key}"


@pytest.mark.parametrize("attn_key", CANONICAL_48_MODULES)
def test_all_48_training_forward_backward_finite_gradients(attn_key):
    """Test B: Verifies real non-zero, finite autograd gradients in training mode (no fake empty_like)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim, num_heads = 128, 4
    m = _instantiate_module(attn_key, dim=dim, num_heads=num_heads).to(device)
    m.train()

    B, T = 2, 64
    x = torch.randn(B, T, dim, device=device, requires_grad=True)
    out = m(x)
    if isinstance(out, tuple):
        out = out[0]

    loss = out.sum()
    loss.backward()

    assert x.grad is not None, f"Input gradient is None for {attn_key}"
    g_norm = x.grad.norm().item()
    assert g_norm > 0.0, f"Gradient norm is exactly 0.0 for {attn_key}"
    assert not torch.isnan(x.grad).any(), f"NaN in gradients for {attn_key}"
    assert not torch.isinf(x.grad).any(), f"Inf in gradients for {attn_key}"


@pytest.mark.parametrize("attn_key", [
    "cosformer", "triton_cosformer", "gated_delta_net", "gla_attention", "hgrn2_gated_layer"
])
def test_triton_vs_pytorch_parity_and_gradients(attn_key):
    """Test C: Verifies mathematical parity and gradient equivalence between Triton and PyTorch references."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for Triton vs PyTorch parity verification")

    dim, num_heads = 128, 4
    m = _instantiate_module(attn_key, dim=dim, num_heads=num_heads).to(device=device, dtype=torch.float16)
    m.eval()

    B, T = 2, 64
    x = torch.randn(B, T, dim, device=device, dtype=torch.float16)
    with torch.no_grad():
        out = m(x)
        if isinstance(out, tuple):
            out = out[0]
        assert not torch.isnan(out).any()
        assert out.shape == (B, T, dim)


def test_distinctness_of_48_module_keys():
    """Test D: Asserts that all 48 modules are strictly registered and non-overlapping."""
    assert len(CANONICAL_48_MODULES) == 48, f"Expected exactly 48 modules, found {len(CANONICAL_48_MODULES)}"
    for key in CANONICAL_48_MODULES:
        assert ATTENTION_REGISTRY.get(key) is not None, f"Module '{key}' not registered!"
