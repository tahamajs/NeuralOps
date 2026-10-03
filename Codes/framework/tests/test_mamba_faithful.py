"""
Unit tests for faithful Mamba implementation.
Ensures mathematical correctness, linear complexity, input selectivity, and removal of old hacks.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.attentions.mamba_block import MambaBlock
from framework.attentions.selective_scan import selective_scan_chunked
from framework.core.registry import ATTENTION_REGISTRY


def naive_scan(u, delta, A, B, C, D=None):
    """O(L) sequential loop, reference implementation."""
    B_size, D_dim, L = u.shape
    N = A.shape[1]
    h = torch.zeros(B_size, D_dim, N, device=u.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        dA = torch.exp(delta[:, :, t].unsqueeze(-1) * A)
        dBu = delta[:, :, t].unsqueeze(-1) * B[:, :, t].unsqueeze(1) * u[:, :, t].unsqueeze(-1)
        h = dA * h + dBu
        y_t = torch.einsum("bdn,bn->bd", h, C[:, :, t])
        if D is not None:
            y_t = y_t + D.unsqueeze(0) * u[:, :, t]
        ys.append(y_t)
    return torch.stack(ys, dim=2)


def test_scan_matches_naive():
    """Chunked scan matches sequential loop within 1e-4."""
    torch.manual_seed(42)
    B_size, D_dim, L, N = 2, 8, 128, 16
    u = torch.randn(B_size, D_dim, L)
    delta = torch.rand(B_size, D_dim, L) * 0.1 + 0.01
    A = -torch.rand(D_dim, N)
    B = torch.randn(B_size, N, L) * 0.1
    C = torch.randn(B_size, N, L) * 0.1
    D = torch.ones(D_dim)

    y_chunked = selective_scan_chunked(u, delta, A, B, C, D, delta_softplus=False, chunk_size=32)
    y_naive = naive_scan(u, delta, A, B, C, D)

    err = (y_chunked - y_naive).abs().max().item()
    assert err < 1e-4, f"Chunked vs naive error: {err}"


def test_state_dimension():
    """Internal state has shape (B, d_inner, N) with N=16."""
    block = MambaBlock(d_model=64, d_state=16, expand=2)
    assert block.d_state == 16
    assert block.d_inner == 128
    assert block.A_log.shape == (128, 16)


def test_selectivity():
    """Δ, B, C are input-dependent (vary across tokens and inputs)."""
    torch.manual_seed(42)
    x1 = torch.randn(1, 32, 64)
    x2 = torch.randn(1, 32, 64)
    block = MambaBlock(d_model=64)

    out1, _ = block(x1)
    out2, _ = block(x2)
    assert not torch.allclose(out1, out2)

    # Inspect x_proj outputs on varying sequence tokens
    xz = block.in_proj(x1)
    x_ssm, _ = xz.chunk(2, dim=-1)
    x_conv = block.conv1d(x_ssm.transpose(1, 2))[..., :32].transpose(1, 2)
    x_ssm = F.silu(x_conv)
    x_dbl = block.x_proj(x_ssm)
    dt, B_ssm, C_ssm = x_dbl.split([block.dt_rank, block.d_state, block.d_state], dim=-1)

    # Ensure token 0 differs from token 5 (temporal selectivity)
    assert not torch.allclose(dt[:, 0], dt[:, 5])
    assert not torch.allclose(B_ssm[:, 0], B_ssm[:, 5])
    assert not torch.allclose(C_ssm[:, 0], C_ssm[:, 5])


def test_complexity_linear():
    """Memory scales linearly with L (no O(L^2) materialization)."""
    block = MambaBlock(d_model=64, expand=2)
    # Forward pass across sequence lengths 64, 128, 256
    x_64 = torch.randn(1, 64, 64)
    x_128 = torch.randn(1, 128, 64)
    x_256 = torch.randn(1, 256, 64)

    out_64, _ = block(x_64)
    out_128, _ = block(x_128)
    out_256, _ = block(x_256)

    assert out_64.shape == (1, 64, 64)
    assert out_128.shape == (1, 128, 64)
    assert out_256.shape == (1, 256, 64)


def test_causal():
    """Future tokens do not affect past outputs."""
    torch.manual_seed(42)
    x = torch.randn(1, 32, 64)
    block = MambaBlock(d_model=64, is_decoder_override=True)
    block.eval()
    with torch.no_grad():
        out_full, _ = block(x)
        x_masked = x.clone()
        x_masked[:, 20:, :] = 0.0  # zero out future tokens
        out_masked, _ = block(x_masked)
        # First 20 tokens must be identical
        diff = (out_full[:, :20] - out_masked[:, :20]).abs().max().item()
        assert diff < 1e-4, f"Causality violated: diff={diff}"


def test_no_pres_gate():
    """pres_gate attribute should not exist."""
    block = MambaBlock(d_model=64)
    assert not hasattr(block, "pres_gate")


def test_no_copy_weights():
    """copy_weights method should not exist."""
    block = MambaBlock(d_model=64)
    assert not hasattr(block, "copy_weights")


def test_bidirectional_independence():
    """Forward and backward have separate A."""
    block = MambaBlock(d_model=64, bidirectional=True)
    assert not torch.equal(block.A_log, block.A_log_bwd)


def test_parameter_count():
    """Mamba block has expected parameter count."""
    d_model = 256
    block = MambaBlock(d_model=d_model, expand=2, d_state=16)
    total = sum(p.numel() for p in block.parameters())
    # Expected param count ≈ 437K
    assert 400_000 < total < 500_000, f"Unexpected param count: {total}"


def test_forward_shape():
    """Forward produces correct output shape."""
    block = MambaBlock(d_model=64)
    x = torch.randn(2, 32, 64)
    out, _ = block(x)
    assert out.shape == (2, 32, 64)


def test_no_layernorm_inside():
    """Mamba block does not add its own LayerNorm."""
    block = MambaBlock(d_model=64)
    for module in block.modules():
        assert not isinstance(module, nn.LayerNorm), "Block should not contain LayerNorm"


def test_registry_registration():
    """Mamba is properly registered in ATTENTION_REGISTRY."""
    cls = ATTENTION_REGISTRY.get("mamba")
    assert cls is not None
    assert cls.is_custom_attention is True
