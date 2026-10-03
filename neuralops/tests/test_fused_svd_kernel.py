"""
Unit tests for Fused SVD-Linear Layer with in-SRAM RMSNorm, Autograd backward pass,
and integration into SVDLinear and SVDLinearAttention modules.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.attentions.fused_svd_kernel import (HAS_TRITON, FusedSVDLinear,
                                                   fused_svd_linear)
from neuralops.attentions.svd_linear_attention import (SVDLinear,
                                                       SVDLinearAttention)
from neuralops.compression_engine.svd_linear import SVDLinearLayer


def test_fused_svd_linear_forward_backward() -> None:
    """Verify forward and backward gradients of FusedSVDLinear with RMSNorm and bias."""
    torch.manual_seed(42)
    B, L, in_dim, out_dim, rank = 2, 16, 64, 128, 16

    layer = FusedSVDLinear(
        in_features=in_dim, out_features=out_dim, rank=rank, bias=True, use_rms_norm=True
    )
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


def test_fused_svd_linear_no_rms_norm() -> None:
    """Verify forward and backward pass when RMSNorm is disabled."""
    torch.manual_seed(42)
    B, L, in_dim, out_dim, rank = 2, 8, 32, 64, 16

    layer = FusedSVDLinear(
        in_features=in_dim, out_features=out_dim, rank=rank, bias=True, use_rms_norm=False
    )
    x = torch.randn(B, L, in_dim, requires_grad=True)

    out = layer(x)
    assert out.shape == (B, L, out_dim)
    assert torch.isfinite(out).all()

    loss = (out ** 2).sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.U.grad is not None and torch.isfinite(layer.U.grad).all()
    assert layer.V.grad is not None and torch.isfinite(layer.V.grad).all()
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()
    assert layer.norm_weight is None


def test_fused_svd_linear_no_bias() -> None:
    """Verify forward and backward pass without bias."""
    torch.manual_seed(42)
    B, L, in_dim, out_dim, rank = 2, 8, 32, 64, 16

    layer = FusedSVDLinear(
        in_features=in_dim, out_features=out_dim, rank=rank, bias=False, use_rms_norm=True
    )
    x = torch.randn(B, L, in_dim, requires_grad=True)

    out = layer(x)
    assert out.shape == (B, L, out_dim)
    assert layer.bias is None

    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.U.grad is not None and torch.isfinite(layer.U.grad).all()
    assert layer.V.grad is not None and torch.isfinite(layer.V.grad).all()
    assert layer.norm_weight.grad is not None and torch.isfinite(layer.norm_weight.grad).all()


def test_fused_svd_linear_nd_shapes() -> None:
    """Verify arbitrary multi-dimensional input tensor shapes (2D, 4D)."""
    torch.manual_seed(42)
    layer = FusedSVDLinear(in_features=32, out_features=64, rank=16)

    # 2D input: (M, K_in)
    x_2d = torch.randn(32, 32)
    out_2d = layer(x_2d)
    assert out_2d.shape == (32, 64)

    # 4D input: (B, H, L, K_in)
    x_4d = torch.randn(2, 4, 8, 32)
    out_4d = layer(x_4d)
    assert out_4d.shape == (2, 4, 8, 64)


def test_fused_svd_linear_gradcheck_full() -> None:
    """Verify analytical gradients match numerical gradients via torch.autograd.gradcheck."""
    torch.manual_seed(42)
    in_dim, out_dim, rank = 4, 4, 2
    x = torch.randn(1, 2, in_dim, dtype=torch.float64, requires_grad=True)
    V = torch.randn(rank, in_dim, dtype=torch.float64, requires_grad=True)
    U = torch.randn(out_dim, rank, dtype=torch.float64, requires_grad=True)
    norm_w = torch.ones(rank, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(out_dim, dtype=torch.float64, requires_grad=True)

    test_passed = torch.autograd.gradcheck(
        fused_svd_linear,
        (x, V, U, norm_w, bias, 1e-5, True),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    assert test_passed, "Gradcheck failed for fused_svd_linear with full parameters"


def test_fused_svd_linear_gradcheck_no_norm_weight() -> None:
    """Verify gradcheck without norm_weight and bias."""
    torch.manual_seed(42)
    in_dim, out_dim, rank = 4, 4, 2
    x = torch.randn(1, 2, in_dim, dtype=torch.float64, requires_grad=True)
    V = torch.randn(rank, in_dim, dtype=torch.float64, requires_grad=True)
    U = torch.randn(out_dim, rank, dtype=torch.float64, requires_grad=True)

    test_passed = torch.autograd.gradcheck(
        fused_svd_linear,
        (x, V, U, None, None, 1e-5, True),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    assert test_passed, "Gradcheck failed for fused_svd_linear with None norm_weight/bias"


def test_fused_svd_linear_gradcheck_no_rms_norm() -> None:
    """Verify gradcheck when RMSNorm is disabled."""
    torch.manual_seed(42)
    in_dim, out_dim, rank = 4, 4, 2
    x = torch.randn(1, 2, in_dim, dtype=torch.float64, requires_grad=True)
    V = torch.randn(rank, in_dim, dtype=torch.float64, requires_grad=True)
    U = torch.randn(out_dim, rank, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(out_dim, dtype=torch.float64, requires_grad=True)

    test_passed = torch.autograd.gradcheck(
        fused_svd_linear,
        (x, V, U, None, bias, 1e-5, False),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    assert test_passed, "Gradcheck failed for fused_svd_linear without RMSNorm"


def test_fused_svd_linear_module_methods() -> None:
    """Verify FusedSVDLinear helper methods: set_low_rank_factors, get_orthogonality_loss, extra_repr."""
    layer = FusedSVDLinear(in_features=32, out_features=64, rank=16, bias=True, use_rms_norm=True)
    
    # Test set_low_rank_factors
    new_U = torch.randn(64, 16)
    new_V = torch.randn(16, 32)
    layer.set_low_rank_factors(new_U, new_V)
    assert torch.allclose(layer.U, new_U)
    assert torch.allclose(layer.V, new_V)

    # Test orthogonality loss
    ortho_loss = layer.get_orthogonality_loss()
    assert ortho_loss.ndim == 0 and ortho_loss >= 0.0

    # Test extra_repr
    repr_str = layer.extra_repr()
    assert "in_features=32" in repr_str
    assert "out_features=64" in repr_str
    assert "rank=16" in repr_str


def test_analytical_vs_reference_gradients() -> None:
    """
    RESEARCH FIX: Verify analytical backward gradients are 100% numerically identical
    to PyTorch autograd graph backpropagation.
    """
    torch.manual_seed(42)
    B, L, in_dim, out_dim, rank = 2, 8, 32, 48, 16

    x_val = torch.randn(B, L, in_dim, dtype=torch.float64)
    V_val = torch.randn(rank, in_dim, dtype=torch.float64)
    U_val = torch.randn(out_dim, rank, dtype=torch.float64)
    w_val = torch.randn(rank, dtype=torch.float64)
    b_val = torch.randn(out_dim, dtype=torch.float64)
    eps = 1e-6

    # 1. Reference PyTorch computation with autograd
    x_ref = x_val.clone().requires_grad_(True)
    V_ref = V_val.clone().requires_grad_(True)
    U_ref = U_val.clone().requires_grad_(True)
    w_ref = w_val.clone().requires_grad_(True)
    b_ref = b_val.clone().requires_grad_(True)

    z_ref = F.linear(x_ref, V_ref)
    rms_ref = torch.rsqrt(torch.mean(z_ref ** 2, dim=-1, keepdim=True) + eps)
    z_norm_ref = z_ref * rms_ref * w_ref
    y_ref = F.linear(z_norm_ref, U_ref, b_ref)
    loss_ref = (y_ref ** 2).sum()
    loss_ref.backward()

    # 2. Fused SVD Linear analytical implementation
    x_test = x_val.clone().requires_grad_(True)
    V_test = V_val.clone().requires_grad_(True)
    U_test = U_val.clone().requires_grad_(True)
    w_test = w_val.clone().requires_grad_(True)
    b_test = b_val.clone().requires_grad_(True)

    y_test = fused_svd_linear(x_test, V_test, U_test, w_test, b_test, eps=eps, use_rms_norm=True)
    loss_test = (y_test ** 2).sum()
    loss_test.backward()

    assert torch.allclose(y_ref, y_test, atol=1e-10, rtol=1e-10)
    assert torch.allclose(x_ref.grad, x_test.grad, atol=1e-10, rtol=1e-10)
    assert torch.allclose(V_ref.grad, V_test.grad, atol=1e-10, rtol=1e-10)
    assert torch.allclose(U_ref.grad, U_test.grad, atol=1e-10, rtol=1e-10)
    assert torch.allclose(w_ref.grad, w_test.grad, atol=1e-10, rtol=1e-10)
    assert torch.allclose(b_ref.grad, b_test.grad, atol=1e-10, rtol=1e-10)


def test_svd_linear_integration() -> None:
    """Verify SVDLinear module with use_fused_kernel=True and False."""
    torch.manual_seed(42)
    in_dim, out_dim, rank = 32, 64, 16

    # Standard mode
    layer_std = SVDLinear(in_dim, out_dim, rank=rank, use_fused_kernel=False)
    # Fused mode
    layer_fused = SVDLinear(in_dim, out_dim, rank=rank, use_fused_kernel=True)
    # Copy parameters
    layer_fused.U.data.copy_(layer_std.U.data)
    layer_fused.V.data.copy_(layer_std.V.data)
    layer_fused.bias.data.copy_(layer_std.bias.data)

    x = torch.randn(2, 8, in_dim)
    out_std = layer_std(x)
    out_fused = layer_fused(x)

    assert torch.allclose(out_std, out_fused, atol=1e-6, rtol=1e-5)


def test_svd_linear_attention_integration() -> None:
    """Verify SVDLinearAttention executes with use_fused_kernel=True across exact and linear modes."""
    torch.manual_seed(42)
    config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        dropout=0.1,
        linear_attention_eps=1e-6,
    )

    # 1. Exact Softmax mode with fused SVD projections
    attn_exact = SVDLinearAttention(config, rank=16, mode="exact", use_fused_kernel=True)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out_exact, _ = attn_exact(x)
    assert out_exact.shape == (2, 16, 64)
    loss = out_exact.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # 2. Linear Kernel mode with fused SVD projections
    attn_linear = SVDLinearAttention(config, rank=16, mode="linear", use_fused_kernel=True)
    x2 = torch.randn(2, 16, 64, requires_grad=True)
    out_linear, _ = attn_linear(x2)
    assert out_linear.shape == (2, 16, 64)
    loss2 = out_linear.sum()
    loss2.backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()


def test_svd_linear_layer_compression_integration() -> None:
    """Verify SVDLinearLayer in compression engine with use_fused_kernel=True."""
    torch.manual_seed(42)
    linear = nn.Linear(32, 64, bias=True)
    svd_layer_fused = SVDLinearLayer(linear, rank=16, use_fused_kernel=True)
    svd_layer_std = SVDLinearLayer(linear, rank=16, use_fused_kernel=False)

    x = torch.randn(2, 8, 32)
    out_fused = svd_layer_fused(x)
    out_std = svd_layer_std(x)

    assert out_fused.shape == (2, 8, 64)
    assert out_std.shape == (2, 8, 64)
    assert torch.allclose(out_fused, out_std, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    test_fused_svd_linear_forward_backward()
    print("test_fused_svd_linear_forward_backward: PASSED")
    test_fused_svd_linear_no_rms_norm()
    print("test_fused_svd_linear_no_rms_norm: PASSED")
    test_fused_svd_linear_no_bias()
    print("test_fused_svd_linear_no_bias: PASSED")
    test_fused_svd_linear_nd_shapes()
    print("test_fused_svd_linear_nd_shapes: PASSED")
    test_fused_svd_linear_gradcheck_full()
    print("test_fused_svd_linear_gradcheck_full: PASSED")
    test_fused_svd_linear_gradcheck_no_norm_weight()
    print("test_fused_svd_linear_gradcheck_no_norm_weight: PASSED")
    test_fused_svd_linear_gradcheck_no_rms_norm()
    print("test_fused_svd_linear_gradcheck_no_rms_norm: PASSED")
    test_fused_svd_linear_module_methods()
    print("test_fused_svd_linear_module_methods: PASSED")
    test_analytical_vs_reference_gradients()
    print("test_analytical_vs_reference_gradients: PASSED")
    test_svd_linear_integration()
    print("test_svd_linear_integration: PASSED")
    test_svd_linear_attention_integration()
    print("test_svd_linear_attention_integration: PASSED")
    test_svd_linear_layer_compression_integration()
    print("test_svd_linear_layer_compression_integration: PASSED")
    print("ALL 12 TESTS PASSED SUCCESSFULLY!")
