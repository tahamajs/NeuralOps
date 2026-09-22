"""Unit test suite for faithful SelectiveSSM and selective scan kernel.

Tests:
1. test_scan_naive_loop
2. test_state_dimension
3. test_selectivity
4. test_bidirectional_independence
5. test_no_layernorm_duplication
6. test_gradient_flow
7. test_forward_shape
8. test_causal_mask
9. test_numerical_stability
10. test_parameter_count
11. test_registry_registration
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.attentions.ssm_scan import selective_scan_chunked, selective_scan
from framework.attentions.selective_ssm import SelectiveSSM
from framework.core.registry import ATTENTION_REGISTRY


def _naive_selective_scan(u, delta, A, B, C, D=None):
    """Naive element-by-element recurrence for ground-truth reference."""
    B_sz, D_sz, L_sz = u.shape
    N_sz = A.shape[-1]
    y = torch.zeros_like(u)
    h = torch.zeros(B_sz, D_sz, N_sz, device=u.device, dtype=u.dtype)

    for t in range(L_sz):
        # u_t: (B, D), delta_t: (B, D), B_t: (B, N), C_t: (B, N)
        u_t = u[:, :, t]
        d_t = delta[:, :, t]
        b_t = B[:, :, t]
        c_t = C[:, :, t]

        # dA_t: (B, D, N) = exp(delta * A)
        dA_t = torch.exp(d_t.unsqueeze(-1) * A.unsqueeze(0))
        # dB_t: (B, D, N) = delta * u * B
        dB_t = d_t.unsqueeze(-1) * u_t.unsqueeze(-1) * b_t.unsqueeze(1)

        h = dA_t * h + dB_t
        # y_t = sum_n(h * C)
        y_t = (h * c_t.unsqueeze(1)).sum(dim=-1)
        if D is not None:
            y_t = y_t + u_t * D.unsqueeze(0)
        y[:, :, t] = y_t

    return y


def test_scan_naive_loop():
    """Verify chunked scan matches naive step-by-step recurrence within 1e-4."""
    torch.manual_seed(42)
    B, D, L, N = 2, 8, 32, 4
    u = torch.randn(B, D, L)
    delta = F.softplus(torch.randn(B, D, L))
    A = -torch.exp(torch.randn(D, N))
    B_mat = torch.randn(B, N, L)
    C_mat = torch.randn(B, N, L)
    D_vec = torch.randn(D)

    y_naive = _naive_selective_scan(u, delta, A, B_mat, C_mat, D_vec)
    y_chunked = selective_scan_chunked(u, delta, A, B_mat, C_mat, D_vec, chunk_size=8)

    max_diff = (y_naive - y_chunked).abs().max().item()
    assert max_diff < 1e-4, f"Chunked scan diverged from naive recurrence: {max_diff}"


def test_state_dimension():
    """Verify that state parameterization maintains full (D_inner, d_state) structured state."""
    model = SelectiveSSM(d_model=64, d_state=16, expand=2)
    assert model.d_inner == 128
    assert model.A_log.shape == (128, 16)
    assert model.D.shape == (128,)


def test_selectivity():
    """Verify that delta, B, C projections are strictly input-dependent."""
    model = SelectiveSSM(d_model=32, d_state=8, expand=2)
    x1 = torch.randn(1, 10, 32)
    x2 = torch.randn(1, 10, 32)

    x_proj1 = model.x_proj(model.conv1d(model.in_proj(x1).chunk(2, dim=-1)[0].transpose(1, 2))[..., :10].transpose(1, 2))
    x_proj2 = model.x_proj(model.conv1d(model.in_proj(x2).chunk(2, dim=-1)[0].transpose(1, 2))[..., :10].transpose(1, 2))

    assert not torch.allclose(x_proj1, x_proj2)


def test_bidirectional_independence():
    """Verify forward and backward directions use independent parameter sets."""
    model = SelectiveSSM(d_model=32, d_state=8, expand=2, bidirectional=True)
    assert hasattr(model, "A_log_bwd")
    assert hasattr(model, "x_proj_bwd")
    assert hasattr(model, "dt_proj_bwd")
    # Verify forward and backward A are independent tensors
    assert model.A_log.data_ptr() != model.A_log_bwd.data_ptr()


def test_no_layernorm_duplication():
    """Verify SelectiveSSM has no internal LayerNorm modules."""
    model = SelectiveSSM(d_model=32, d_state=8)
    for mod in model.modules():
        assert not isinstance(mod, nn.LayerNorm)


def test_gradient_flow():
    """Verify gradients propagate cleanly to A_log, D, and dt_proj."""
    model = SelectiveSSM(d_model=32, d_state=8, expand=2)
    x = torch.randn(2, 16, 32, requires_grad=True)
    out, _ = model(x)
    loss = out.sum()
    loss.backward()

    assert model.A_log.grad is not None and model.A_log.grad.abs().sum() > 0
    assert model.D.grad is not None and model.D.grad.abs().sum() > 0
    assert model.dt_proj.weight.grad is not None and model.dt_proj.weight.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_forward_shape():
    """Verify (B, L, D) -> (B, L, D)."""
    model = SelectiveSSM(d_model=64, d_state=16, expand=2)
    x = torch.randn(2, 20, 64)
    out, attn = model(x)
    assert out.shape == (2, 20, 64)
    assert attn is None


def test_causal_mask():
    """Verify causality: altering token at step t does not alter outputs at step < t."""
    model = SelectiveSSM(d_model=32, d_state=8, expand=2, bidirectional=False)
    model.eval()

    x1 = torch.randn(1, 16, 32)
    x2 = x1.clone()
    # Change future token at index 10
    x2[:, 10:, :] = torch.randn(1, 6, 32)

    with torch.no_grad():
        out1, _ = model(x1)
        out2, _ = model(x2)

    # Outputs before index 10 must be identical
    diff_past = (out1[:, :10, :] - out2[:, :10, :]).abs().max().item()
    assert diff_past < 1e-5, f"Future altered past output! max diff: {diff_past}"


def test_numerical_stability():
    """Verify no NaN or Inf with extreme input values."""
    model = SelectiveSSM(d_model=32, d_state=8, expand=2)
    x_extreme = torch.randn(2, 64, 32) * 100.0
    out, _ = model(x_extreme)
    assert not torch.isnan(out).any(), "NaN found in output with extreme input"
    assert not torch.isinf(out).any(), "Inf found in output with extreme input"


def test_parameter_count():
    """Verify parameter count is comparable to a transformer with same d_model."""
    d_model = 256
    ssm = SelectiveSSM(d_model=d_model, d_state=16, expand=2)
    ssm_params = sum(p.numel() for p in ssm.parameters())

    # Reference attention: 4 projections of (d_model, d_model)
    attn_params = 4 * (d_model * d_model)
    # SSM params should be on the same order of magnitude (within 3x)
    ratio = ssm_params / attn_params
    assert 0.5 <= ratio <= 3.0, f"SSM parameter count ratio unexpected: {ratio:.2f}"


def test_registry_registration():
    """Verify registration in ATTENTION_REGISTRY."""
    assert "selective_ssm" in ATTENTION_REGISTRY
    assert "mamba_ssm" in ATTENTION_REGISTRY
    assert "simple_ssm" in ATTENTION_REGISTRY
