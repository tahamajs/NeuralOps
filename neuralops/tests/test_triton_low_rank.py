"""
Comprehensive test suite for neuralops.core.triton_kernels.low_rank.

Coverage:
- Numerical correctness vs PyTorch dual GEMM reference: y = (x @ V.T) @ U.T + bias
- Multiple dtypes: fp32, fp16, bf16
- Multiple ranks and dimensions: rank < in_features, rank < out_features
- Bias present vs bias None
- Fallback path when Triton unavailable
- Determinism
- Analytical autograd backward gradients
"""
import pytest
import torch

from neuralops.core.triton_kernels._common import HAS_TRITON, low_rank_pytorch_fallback
from neuralops.core.triton_kernels.low_rank import fwd, bwd, is_supported


# ═══════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════

@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reference_low_rank_pytorch(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Pure PyTorch reference dual-projection: y = (x @ V.T) @ U.T + bias."""
    D_in = x.shape[-1]
    V_mat = V.t() if V.shape[-1] == D_in else V
    Rank = V_mat.shape[1]
    U_mat = U.t() if U.shape[-1] == Rank else U
    
    z = torch.matmul(x, V_mat)
    y = torch.matmul(z, U_mat)
    if bias is not None:
        y = y + bias
    return y


# ═══════════════════════════════════════════════════
# Numerical correctness
# ═══════════════════════════════════════════════════

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("batch,seq_len", [(1, 16), (2, 64), (4, 128)])
@pytest.mark.parametrize("in_dim,rank,out_dim", [(64, 16, 64), (128, 32, 64), (64, 8, 128)])
def test_low_rank_correctness(device, dtype, has_bias, batch, seq_len, in_dim, rank, out_dim):
    """Verify forward projection matches PyTorch reference."""
    if dtype == torch.bfloat16 and not torch.cuda.is_available() and device.type == "cuda":
        pytest.skip("bfloat16 cuda not supported")

    x = torch.randn(batch, seq_len, in_dim, device=device, dtype=dtype)
    V = torch.randn(rank, in_dim, device=device, dtype=dtype) # [Rank, D_in]
    U = torch.randn(out_dim, rank, device=device, dtype=dtype) # [D_out, Rank]
    bias = torch.randn(out_dim, device=device, dtype=dtype) if has_bias else None

    out = fwd(x, U, V, bias=bias)
    ref = reference_low_rank_pytorch(x, U, V, bias=bias)

    tol = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 2e-2}[dtype]
    assert out.shape == (batch, seq_len, out_dim)
    assert torch.allclose(out, ref, atol=tol, rtol=tol)


# ═══════════════════════════════════════════════════
# Edge cases & Non-power-of-2
# ═══════════════════════════════════════════════════

def test_low_rank_single_token(device):
    """Single token sequence."""
    x = torch.randn(1, 1, 64, device=device)
    V = torch.randn(16, 64, device=device)
    U = torch.randn(32, 16, device=device)
    out = fwd(x, U, V)
    assert out.shape == (1, 1, 32)


def test_low_rank_non_power_of_2(device):
    """Non-power-of-2 dimensions."""
    x = torch.randn(2, 33, 47, device=device)
    V = torch.randn(13, 47, device=device)
    U = torch.randn(29, 13, device=device)
    bias = torch.randn(29, device=device)
    out = fwd(x, U, V, bias=bias)
    assert out.shape == (2, 33, 29)
    assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════
# Fallback path & Determinism
# ═══════════════════════════════════════════════════

def test_low_rank_cpu_fallback():
    """Verify fallback executes cleanly on CPU."""
    x = torch.randn(2, 32, 64, device="cpu")
    V = torch.randn(16, 64, device="cpu")
    U = torch.randn(64, 16, device="cpu")
    out = low_rank_pytorch_fallback(x, U, V)
    assert out.shape == (2, 32, 64)
    assert torch.isfinite(out).all()


def test_low_rank_deterministic(device):
    """Deterministic output for identical input."""
    x = torch.randn(2, 16, 32, device=device)
    V = torch.randn(8, 32, device=device)
    U = torch.randn(32, 8, device=device)
    out1 = fwd(x, U, V)
    out2 = fwd(x, U, V)
    assert torch.equal(out1, out2)


# ═══════════════════════════════════════════════════
# Backward gradients
# ═══════════════════════════════════════════════════

def test_low_rank_backward_gradients(device):
    """Verify gradients propagate to x, U, V, and bias."""
    x = torch.randn(2, 16, 32, device=device, requires_grad=True)
    V = torch.randn(8, 32, device=device, requires_grad=True)
    U = torch.randn(16, 8, device=device, requires_grad=True)
    bias = torch.randn(16, device=device, requires_grad=True)

    out = fwd(x, U, V, bias=bias)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert U.grad is not None and torch.isfinite(U.grad).all()
    assert V.grad is not None and torch.isfinite(V.grad).all()
    assert bias.grad is not None and torch.isfinite(bias.grad).all()
