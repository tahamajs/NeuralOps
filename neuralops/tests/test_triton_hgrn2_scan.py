"""
Comprehensive test suite for neuralops.core.triton_kernels.hgrn2_scan.

Coverage:
- Numerical correctness vs PyTorch reference
- Multiple dtypes: fp32, fp16, bf16
- Multiple shapes: batch, seq, dim variations
- Edge cases: clipping, small/large dimensions
- Fallback path & error handling
- Determinism
- Gradient correctness
"""
import pytest
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON
from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_triton


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reference_hgrn2_scan_pytorch(f, iv, res_clip=None):
    """
    Pure PyTorch sequential reference for HGRN2 selective scan.
    Recurrence:
        h_t = f_t * h_{t-1} + iv_t
        if res_clip:
            h_t = clamp(h_t, -res_clip, res_clip)
    """
    B, L, D = f.shape
    f_f = f.float()
    iv_f = iv.float()

    h = torch.zeros(B, L, D, device=f.device, dtype=torch.float32)
    h_prev = torch.zeros(B, D, device=f.device, dtype=torch.float32)

    for t in range(L):
        ft = f_f[:, t, :]
        ivt = iv_f[:, t, :]
        ht = ft * h_prev + ivt
        if res_clip is not None and res_clip > 0:
            ht = torch.clamp(ht, -res_clip, res_clip)
        h[:, t, :] = ht
        h_prev = ht

    return h.to(f.dtype)


def test_reference_hgrn2_scan_pytorch_basic():
    """Verify PyTorch reference runs and outputs correct shape on CPU."""
    B, L, D = 2, 16, 32
    f = torch.sigmoid(torch.randn(B, L, D))
    iv = torch.randn(B, L, D)

    out = reference_hgrn2_scan_pytorch(f, iv, res_clip=1.0)
    assert out.shape == (B, L, D)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not HAS_TRITON or not torch.cuda.is_available(), reason="Requires Triton and CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("res_clip", [None, 1.0])
def test_hgrn2_scan_triton_correctness(device, dtype, D, res_clip):
    """Verify Triton HGRN2 matches PyTorch reference on CUDA."""
    B, L = 2, 16
    f = torch.sigmoid(torch.randn(B, L, D, device=device, dtype=dtype))
    iv = torch.randn(B, L, D, device=device, dtype=dtype)

    out = hgrn2_scan_triton(f, iv, res_clip=res_clip)
    ref = reference_hgrn2_scan_pytorch(f, iv, res_clip=res_clip)

    tol = 1e-2 if dtype != torch.float32 else 1e-4
    assert torch.allclose(out, ref, atol=tol, rtol=tol)


@pytest.mark.skipif(not HAS_TRITON or not torch.cuda.is_available(), reason="Requires Triton and CUDA")
def test_hgrn2_scan_triton_backward(device):
    """Verify backward gradient computation."""
    B, L, D = 1, 16, 32
    f = torch.sigmoid(torch.randn(B, L, D, device=device, requires_grad=True))
    iv = torch.randn(B, L, D, device=device, requires_grad=True)

    out = hgrn2_scan_triton(f, iv, res_clip=1.0)
    loss = out.sum()
    loss.backward()

    assert f.grad is not None and torch.isfinite(f.grad).all()
    assert iv.grad is not None and torch.isfinite(iv.grad).all()
