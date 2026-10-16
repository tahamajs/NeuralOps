"""
Comprehensive GPU tests for Flash GQA (Module 4).
Validates forward numerical parity, N_Q != N_KV (decode / chunked prefill),
multi-dtype (FP32, FP16, BF16), non-contiguous tensors, unsupported head_dim fallback,
and analytical autograd backward pass.
"""

import pytest
import torch

from neuralops.core.cuda_kernels.flash_gqa import (
    flash_gqa,
    flash_gqa_pytorch_ref,
)
from neuralops.core.cuda_kernels.flash_gqa.flash_gqa_autograd import _get_module


@pytest.fixture(autouse=True)
def check_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("h_q,h_kv,d", [(8, 8, 64), (8, 2, 64), (8, 1, 128), (4, 2, 32)])
def test_flash_gqa_forward_equal_lengths(dtype, is_causal, h_q, h_kv, d):
    """Test equal sequence lengths across MHA, GQA, and MQA configurations."""
    torch.manual_seed(42)
    B, N = 2, 64

    q = torch.randn(B, h_q, N, d, device="cuda", dtype=dtype)
    k = torch.randn(B, h_kv, N, d, device="cuda", dtype=dtype)
    v = torch.randn(B, h_kv, N, d, device="cuda", dtype=dtype)

    out_cuda = flash_gqa(q, k, v, is_causal=is_causal)
    out_ref = flash_gqa_pytorch_ref(q, k, v, is_causal=is_causal)

    tol = 1e-3 if dtype == torch.float32 else (5e-2 if dtype == torch.bfloat16 else 1e-2)
    torch.testing.assert_close(out_cuda, out_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("n_q,n_kv", [(1, 64), (16, 64), (32, 128)])
def test_flash_gqa_unequal_sequence_lengths(is_causal, n_q, n_kv):
    """Test decode step (N_Q = 1) and chunked prefill (N_Q < N_KV)."""
    torch.manual_seed(42)
    B, h_q, h_kv, d = 2, 8, 2, 64

    q = torch.randn(B, h_q, n_q, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, h_kv, n_kv, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, h_kv, n_kv, d, device="cuda", dtype=torch.float32)

    out_cuda = flash_gqa(q, k, v, is_causal=is_causal)
    out_ref = flash_gqa_pytorch_ref(q, k, v, is_causal=is_causal)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)


def test_flash_gqa_non_contiguous():
    """Test strided non-contiguous inputs."""
    torch.manual_seed(42)
    q = torch.randn(2, 4, 32, 64, device="cuda")[:, :, ::2, :]  # shape: [2, 4, 16, 64], non-contiguous
    k = torch.randn(2, 2, 32, 64, device="cuda")[:, :, ::2, :]
    v = torch.randn(2, 2, 32, 64, device="cuda")[:, :, ::2, :]

    assert not q.is_contiguous()
    out_cuda = flash_gqa(q, k, v, is_causal=False)
    out_ref = flash_gqa_pytorch_ref(q, k, v, is_causal=False)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)


def test_flash_gqa_unsupported_head_dim_fallback():
    """Verify head_dim=48 falls back gracefully to PyTorch without crashing."""
    torch.manual_seed(42)
    q = torch.randn(2, 4, 16, 48, device="cuda")
    k = torch.randn(2, 2, 16, 48, device="cuda")
    v = torch.randn(2, 2, 16, 48, device="cuda")

    out = flash_gqa(q, k, v, is_causal=False)
    ref = flash_gqa_pytorch_ref(q, k, v, is_causal=False)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_gqa_autograd_backward(is_causal):
    """Verify loss.backward() gradients on Q, K, V match analytical PyTorch autograd."""
    torch.manual_seed(42)
    B, h_q, h_kv, N, d = 2, 4, 2, 16, 32

    q1 = torch.randn(B, h_q, N, d, device="cuda", dtype=torch.float32, requires_grad=True)
    k1 = torch.randn(B, h_kv, N, d, device="cuda", dtype=torch.float32, requires_grad=True)
    v1 = torch.randn(B, h_kv, N, d, device="cuda", dtype=torch.float32, requires_grad=True)

    q2 = q1.clone().detach().requires_grad_(True)
    k2 = k1.clone().detach().requires_grad_(True)
    v2 = v1.clone().detach().requires_grad_(True)

    out1 = flash_gqa(q1, k1, v1, is_causal=is_causal)
    loss1 = (out1 ** 2).sum()
    loss1.backward()

    out2 = flash_gqa_pytorch_ref(q2, k2, v2, is_causal=is_causal)
    loss2 = (out2 ** 2).sum()
    loss2.backward()

    torch.testing.assert_close(out1, out2, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(q1.grad, q2.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k1.grad, k2.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v1.grad, v2.grad, atol=1e-2, rtol=1e-2)
