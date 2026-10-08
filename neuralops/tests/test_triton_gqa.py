"""
Comprehensive test suite for neuralops.core.triton_kernels.gqa.

Coverage:
- Numerical correctness vs PyTorch repeat_kv reference
- Multiple dtypes: fp32, fp16, bf16
- Multiple shapes: batch, seq, head ratios
- Edge cases: num_heads == num_kv_heads, invalid ratios
- Fallback path when Triton unavailable
- Determinism
- Backward gradient reduction
- Shape preservation
"""
import pytest
import torch

from neuralops.core.triton_kernels._common import HAS_TRITON, gqa_pytorch_fallback
from neuralops.core.triton_kernels.gqa import fwd, bwd, is_supported


# ═══════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════

@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reference_gqa_pytorch(k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int):
    """Pure PyTorch reference expansion."""
    if num_kv_heads == num_heads:
        return k, v
    n_rep = num_heads // num_kv_heads
    B, H_kv, N, D = k.shape
    k_exp = k[:, :, None, :, :].expand(B, H_kv, n_rep, N, D).reshape(B, num_heads, N, D).contiguous()
    v_exp = v[:, :, None, :, :].expand(B, H_kv, n_rep, N, D).reshape(B, num_heads, N, D).contiguous()
    return k_exp, v_exp


# ═══════════════════════════════════════════════════
# Numerical correctness
# ═══════════════════════════════════════════════════

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 64, 128])
@pytest.mark.parametrize("num_kv_heads,num_heads", [(1, 4), (2, 8), (4, 16), (4, 4)])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_gqa_correctness(device, dtype, batch, seq_len, num_kv_heads, num_heads, head_dim):
    """Verify forward expansion matches PyTorch reference across dtypes and shapes."""
    if dtype == torch.bfloat16 and not torch.cuda.is_available() and device.type == "cuda":
        pytest.skip("bfloat16 cuda not supported")

    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

    k_out, v_out = fwd(k, v, num_kv_heads, num_heads)
    k_ref, v_ref = reference_gqa_pytorch(k, v, num_kv_heads, num_heads)

    assert k_out.shape == (batch, num_heads, seq_len, head_dim)
    assert v_out.shape == (batch, num_heads, seq_len, head_dim)
    assert torch.allclose(k_out, k_ref, atol=1e-3, rtol=1e-3)
    assert torch.allclose(v_out, v_ref, atol=1e-3, rtol=1e-3)


# ═══════════════════════════════════════════════════
# Edge cases & Unsupported checks
# ═══════════════════════════════════════════════════

def test_gqa_equal_heads(device):
    """When num_heads == num_kv_heads, return inputs directly without copy overhead."""
    k = torch.randn(2, 4, 32, 64, device=device)
    v = torch.randn(2, 4, 32, 64, device=device)
    k_out, v_out = fwd(k, v, 4, 4)
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)


def test_gqa_single_token(device):
    """Single token sequence (seq_len = 1)."""
    k = torch.randn(1, 2, 1, 64, device=device)
    v = torch.randn(1, 2, 1, 64, device=device)
    k_out, v_out = fwd(k, v, 2, 8)
    assert k_out.shape == (1, 8, 1, 64)
    assert v_out.shape == (1, 8, 1, 64)


def test_gqa_non_power_of_2_dim(device):
    """Head dimension that is not a power of 2."""
    k = torch.randn(2, 2, 32, 48, device=device)
    v = torch.randn(2, 2, 32, 48, device=device)
    k_out, v_out = fwd(k, v, 2, 6)
    assert k_out.shape == (2, 6, 32, 48)
    assert v_out.shape == (2, 6, 32, 48)


def test_gqa_unsupported_ratio_fallback(device):
    """num_heads not divisible by num_kv_heads should fallback or be handled safely."""
    k = torch.randn(2, 3, 32, 64, device=device)
    v = torch.randn(2, 3, 32, 64, device=device)
    # 7 is not divisible by 3 -> is_supported returns False
    assert not is_supported(k, v, 3, 7)


# ═══════════════════════════════════════════════════
# Fallback path & Determinism
# ═══════════════════════════════════════════════════

def test_gqa_pytorch_fallback_cpu():
    """Verify PyTorch fallback works seamlessly on CPU."""
    k = torch.randn(2, 2, 32, 64, device="cpu")
    v = torch.randn(2, 2, 32, 64, device="cpu")
    k_out, v_out = gqa_pytorch_fallback(k, v, 2, 8)
    assert k_out.shape == (2, 8, 32, 64)
    assert v_out.shape == (2, 8, 32, 64)
    assert torch.isfinite(k_out).all()
    assert torch.isfinite(v_out).all()


def test_gqa_deterministic(device):
    """Deterministic output for identical input."""
    k = torch.randn(2, 2, 32, 64, device=device)
    v = torch.randn(2, 2, 32, 64, device=device)
    k1, v1 = fwd(k, v, 2, 8)
    k2, v2 = fwd(k, v, 2, 8)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)


# ═══════════════════════════════════════════════════
# Backward gradient reduction
# ═══════════════════════════════════════════════════

def test_gqa_backward_reduction(device):
    """Verify gradient reduction correctly reduces expanded gradients."""
    grad_k = torch.ones(2, 8, 16, 64, device=device)
    grad_v = torch.ones(2, 8, 16, 64, device=device)
    gk, gv = bwd(grad_k, grad_v, num_kv_heads=2, num_heads=8)
    # Each KV head was repeated 4 times, so sum should be 4.0
    assert gk.shape == (2, 2, 16, 64)
    assert gv.shape == (2, 2, 16, 64)
    assert torch.allclose(gk, torch.full_like(gk, 4.0))
    assert torch.allclose(gv, torch.full_like(gv, 4.0))
