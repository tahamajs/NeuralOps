"""
Parity and backward tests for Selective Scan (Module 2).
"""

import pytest
import torch

from neuralops.core.cuda_kernels.selective_scan import (
    selective_scan,
    selective_scan_pytorch_ref,
)


def _setup_scan_inputs(B=2, D=32, L=64, D_STATE=16, device="cuda", dtype=torch.float32):
    torch.manual_seed(42)
    u = torch.randn(B, D, L, device=device, dtype=dtype)
    delta = torch.randn(B, D, L, device=device, dtype=dtype).sigmoid() * 0.1
    A = -torch.rand(D, D_STATE, device=device, dtype=dtype) - 0.5
    B_sel = torch.randn(B, D_STATE, L, device=device, dtype=dtype)
    C_sel = torch.randn(B, D_STATE, L, device=device, dtype=dtype)
    D_val = torch.randn(D, device=device, dtype=dtype)
    return u, delta, A, B_sel, C_sel, D_val


def test_selective_scan_forward_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Selective Scan test")

    u, delta, A, B_sel, C_sel, D_val = _setup_scan_inputs()

    out_cuda, state_cuda = selective_scan(u, delta, A, B_sel, C_sel, D_val)
    out_ref, state_ref = selective_scan_pytorch_ref(u, delta, A, B_sel, C_sel, D_val)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(state_cuda, state_ref, atol=1e-3, rtol=1e-3)


def test_selective_scan_numerical_stability():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Selective Scan test")

    # Extreme scale test
    u, delta, A, B_sel, C_sel, D_val = _setup_scan_inputs()
    delta = delta * 100.0  # Test clamping to prevent exp overflow
    out_cuda, _ = selective_scan(u, delta, A, B_sel, C_sel, D_val)

    assert not torch.isnan(out_cuda).any()
    assert not torch.isinf(out_cuda).any()
