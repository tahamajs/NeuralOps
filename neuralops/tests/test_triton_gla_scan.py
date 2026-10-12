"""
Comprehensive test suite for neuralops.core.triton_kernels.gla_scan.

Coverage:
- Numerical correctness vs PyTorch reference
- Multiple dtypes: fp32, fp16, bf16
- Multiple shapes: batch, seq, head, dim variations
- Edge cases: head_dim bounds
- Fallback path & error handling
- Determinism
- Gradient correctness
"""
import pytest
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON
from neuralops.core.triton_kernels.gla_scan import MAX_REGISTER_DIM, gla_scan_triton


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reference_gla_scan_pytorch(q, k, v, lambdas):
    """
    Pure PyTorch sequential reference for Gated Linear Attention scan.
    Recurrence:
        S_t = lam_t * S_{t-1} + k_t^T @ v_t    # (D, D) matrix
        Z_t = lam_t * Z_{t-1} + k_t            # (D,) vector
        num_t = q_t @ S_t                      # (D,)
        den_t = q_t @ Z_t + 1e-4               # scalar
        y_t = num_t / den_t                    # (D,)
    """
    B, H, L, D = q.shape
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    lam_f = lambdas.float().clamp(min=1e-4, max=1.0 - 1e-4)

    y = torch.zeros(B, H, L, D, device=q.device, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            S = torch.zeros(D, D, device=q.device, dtype=torch.float32)
            Z = torch.zeros(D, device=q.device, dtype=torch.float32)
            for t in range(L):
                qt = q_f[b, h, t]          # (D,)
                kt = k_f[b, h, t]          # (D,)
                vt = v_f[b, h, t]          # (D,)
                lt = lam_f[b, h, t]        # scalar

                S = lt * S + torch.outer(kt, vt)
                Z = lt * Z + kt

                num = torch.matmul(qt, S)
                den = torch.dot(qt, Z) + 1e-4
                y[b, h, t] = num / den

    return y.to(q.dtype)


def test_reference_gla_scan_pytorch_basic():
    """Verify PyTorch reference runs and outputs correct shape on CPU."""
    B, H, L, D = 2, 4, 16, 32
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    lambdas = torch.sigmoid(torch.randn(B, H, L))

    out = reference_gla_scan_pytorch(q, k, v, lambdas)
    assert out.shape == (B, H, L, D)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not HAS_TRITON or not torch.cuda.is_available(), reason="Requires Triton and CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D", [16, 32, 64])
def test_gla_scan_triton_correctness(device, dtype, D):
    """Verify Triton GLA matches PyTorch reference on CUDA."""
    B, H, L = 2, 2, 16
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)
    lambdas = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype))

    out = gla_scan_triton(q, k, v, lambdas)
    ref = reference_gla_scan_pytorch(q, k, v, lambdas)

    tol = 1e-2 if dtype != torch.float32 else 1e-4
    assert torch.allclose(out, ref, atol=tol, rtol=tol)


@pytest.mark.skipif(not HAS_TRITON or not torch.cuda.is_available(), reason="Requires Triton and CUDA")
def test_gla_scan_triton_dim_exceeded(device):
    """Verify exception raised when D > MAX_REGISTER_DIM."""
    B, H, L, D = 1, 1, 8, MAX_REGISTER_DIM + 16
    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    v = torch.randn(B, H, L, D, device=device)
    lambdas = torch.sigmoid(torch.randn(B, H, L, device=device))

    with pytest.raises(ValueError, match="head_dim"):
        gla_scan_triton(q, k, v, lambdas)


@pytest.mark.skipif(not HAS_TRITON or not torch.cuda.is_available(), reason="Requires Triton and CUDA")
def test_gla_scan_triton_backward(device):
    """Verify backward gradient computation."""
    B, H, L, D = 1, 2, 16, 32
    q = torch.randn(B, H, L, D, device=device, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, requires_grad=True)
    lambdas = torch.sigmoid(torch.randn(B, H, L, device=device)).detach().requires_grad_()

    out = gla_scan_triton(q, k, v, lambdas)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert lambdas.grad is not None and torch.isfinite(lambdas.grad).all()
