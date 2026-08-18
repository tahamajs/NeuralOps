"""
Unit tests for Fused SVD-Linear Layer with in-SRAM RMSNorm and Autograd backward pass.
"""

import pytest
import torch
import torch.nn as nn
from framework.attentions.fused_svd_kernel import FusedSVDLinear, fused_svd_linear


def test_fused_svd_linear_forward_backward():
    """Verify forward and backward gradients of FusedSVDLinear."""
    torch.manual_seed(42)
    B, L, in_dim, out_dim, rank = 2, 16, 64, 128, 16
    
    layer = FusedSVDLinear(in_features=in_dim, out_features=out_dim, rank=rank, bias=True, use_rms_norm=True)
    x = torch.randn(B, L, in_dim, requires_grad=True)

    out = layer(x)
    assert out.shape == (B, L, out_dim)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.U.grad is not None and torch.isfinite(layer.U.grad).all()
    assert layer.V.grad is not None and torch.isfinite(layer.V.grad).all()
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()
    assert layer.norm_weight.grad is not None and torch.isfinite(layer.norm_weight.grad).all()


def test_fused_svd_linear_gradcheck():
    """Verify analytical gradients match numerical gradients via torch.autograd.gradcheck."""
    torch.manual_seed(42)
    in_dim, out_dim, rank = 8, 8, 4
    x = torch.randn(2, 4, in_dim, dtype=torch.float64, requires_grad=True)
    V = torch.randn(rank, in_dim, dtype=torch.float64, requires_grad=True)
    U = torch.randn(out_dim, rank, dtype=torch.float64, requires_grad=True)
    norm_w = torch.ones(rank, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(out_dim, dtype=torch.float64, requires_grad=True)

    test_passed = torch.autograd.gradcheck(
        fused_svd_linear,
        (x, V, U, norm_w, bias, 1e-5),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    assert test_passed, "Gradcheck failed for fused_svd_linear"
