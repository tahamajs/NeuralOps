"""
Parity tests for Flash GQA (Module 4).
"""

import pytest
import torch

from neuralops.core.cuda_kernels.flash_gqa import (
    flash_gqa,
    flash_gqa_pytorch_ref,
)


def test_flash_gqa_forward_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Flash GQA test")

    torch.manual_seed(42)
    B, H, H_KV, N, D = 2, 8, 2, 64, 64

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H_KV, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H_KV, N, D, device="cuda", dtype=torch.float32)

    out_cuda = flash_gqa(q, k, v, is_causal=False)
    out_ref = flash_gqa_pytorch_ref(q, k, v, is_causal=False)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)


def test_flash_gqa_causal_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Flash GQA test")

    torch.manual_seed(42)
    B, H, H_KV, N, D = 2, 4, 1, 32, 64

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H_KV, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H_KV, N, D, device="cuda", dtype=torch.float32)

    out_cuda = flash_gqa(q, k, v, is_causal=True)
    out_ref = flash_gqa_pytorch_ref(q, k, v, is_causal=True)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)
