# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Aegis Gated Normalization CUDA kernel & CPU fallback (Wave 9 / W9-K5).
"""

from __future__ import annotations

import inspect
import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.aegis_gate_norm import (
    aegis_gate_norm,
    aegis_gate_norm_pytorch_ref,
    aegis_gate_norm_triton_ref,
)
import neuralops.ops.cuda.aegis_gate_norm as agn_pkg
from neuralops.ops.cuda.aegis_gate_norm import aegis_gate_norm_autograd as agn_autograd
from neuralops.ops.cuda._fallback import force_kernel_backend


def test_aegis_gate_norm_cpu_fallback_basic() -> None:
    """Verify basic forward execution and tensor attributes on CPU fallback."""
    b0 = torch.randn(2, 4, 16, dtype=torch.float32)
    b1 = torch.randn(2, 4, 16, dtype=torch.float32)
    weights = torch.randn(2, 4, 2, dtype=torch.float32)

    out = aegis_gate_norm(b0, b1, weights)

    assert isinstance(out, torch.Tensor)
    assert not isinstance(out, tuple)
    assert out.shape == b0.shape
    assert out.dtype == b0.dtype
    assert out.device == b0.device


def test_aegis_gate_norm_numerical_parity() -> None:
    """Verify numerical parity against the PyTorch reference implementation."""
    torch.manual_seed(42)
    b0 = torch.randn(4, 8, 32, dtype=torch.float32)
    b1 = torch.randn(4, 8, 32, dtype=torch.float32)
    weights = torch.randn(4, 8, 2, dtype=torch.float32)

    out = aegis_gate_norm(b0, b1, weights, eps=1e-5)
    ref = aegis_gate_norm_pytorch_ref(b0, b1, weights, eps=1e-5)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_aegis_gate_norm_autograd_backward() -> None:
    """Verify analytical gradients propagate back through b0, b1, and weights."""
    b0 = torch.randn(2, 4, 16, dtype=torch.float32, requires_grad=True)
    b1 = torch.randn(2, 4, 16, dtype=torch.float32, requires_grad=True)
    weights = torch.randn(2, 4, 2, dtype=torch.float32, requires_grad=True)

    out = aegis_gate_norm(b0, b1, weights)
    loss = (out * 2.5).sum()
    loss.backward()

    assert b0.grad is not None
    assert b0.grad.shape == b0.shape
    assert b1.grad is not None
    assert b1.grad.shape == b1.shape
    assert weights.grad is not None
    assert weights.grad.shape == weights.shape


def test_aegis_gate_norm_amp_and_dtypes() -> None:
    """Verify torch.amp compliance (Rule 2.1) and multiple floating-point datatypes."""
    # FP16
    b0_fp16 = torch.randn(2, 4, 16, dtype=torch.float16)
    b1_fp16 = torch.randn(2, 4, 16, dtype=torch.float16)
    w_fp16 = torch.randn(2, 4, 2, dtype=torch.float16)
    out_fp16 = aegis_gate_norm(b0_fp16, b1_fp16, w_fp16)
    assert out_fp16.dtype == torch.float16

    # BF16
    b0_bf16 = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    b1_bf16 = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    w_bf16 = torch.randn(2, 4, 2, dtype=torch.bfloat16)
    out_bf16 = aegis_gate_norm(b0_bf16, b1_bf16, w_bf16)
    assert out_bf16.dtype == torch.bfloat16

    # Context manager torch.amp.autocast
    with autocast("cpu", dtype=torch.bfloat16):
        b0 = torch.randn(2, 4, 16)
        b1 = torch.randn(2, 4, 16)
        w = torch.randn(2, 4, 2)
        out_amp = aegis_gate_norm(b0, b1, w)
        assert isinstance(out_amp, torch.Tensor)


def test_aegis_gate_norm_shape_validation() -> None:
    """Verify shape validation catches mismatches gracefully."""
    b0 = torch.randn(2, 4, 16)
    b1_wrong_hidden = torch.randn(2, 4, 32)
    b1_wrong_batch = torch.randn(3, 4, 16)
    w_correct = torch.randn(2, 4, 2)
    w_wrong_last = torch.randn(2, 4, 3)
    w_wrong_batch = torch.randn(3, 4, 2)

    with pytest.raises(ValueError, match="Shape mismatch"):
        aegis_gate_norm(b0, b1_wrong_hidden, w_correct)

    with pytest.raises(ValueError, match="Shape mismatch"):
        aegis_gate_norm(b0, b1_wrong_batch, w_correct)

    with pytest.raises(ValueError, match="weights last dimension to be 2"):
        aegis_gate_norm(b0, b0, w_wrong_last)

    with pytest.raises(ValueError, match="Token batch dimensions mismatch"):
        aegis_gate_norm(b0, b0, w_wrong_batch)


def test_aegis_gate_norm_empty_tensor() -> None:
    """Verify handling of empty (zero-token) tensors."""
    b0 = torch.empty(0, 16)
    b1 = torch.empty(0, 16)
    w = torch.empty(0, 2)
    out = aegis_gate_norm(b0, b1, w)
    assert out.shape == b0.shape


def test_aegis_gate_norm_top_level_dispatch() -> None:
    """Verify top-level export and dispatch via neuralops.ops.cuda works."""
    b0 = torch.randn(2, 4, 16)
    b1 = torch.randn(2, 4, 16)
    weights = torch.randn(2, 4, 2)

    out = noc.aegis_gate_norm(b0, b1, weights)
    assert isinstance(out, torch.Tensor)
    assert out.shape == b0.shape


def test_aegis_gate_norm_force_pytorch_backend() -> None:
    """Verify explicit backend forcing to pytorch fallback."""
    b0 = torch.randn(2, 4, 16)
    b1 = torch.randn(2, 4, 16)
    weights = torch.randn(2, 4, 2)

    with force_kernel_backend("pytorch"):
        out = aegis_gate_norm(b0, b1, weights)
        assert isinstance(out, torch.Tensor)
        assert out.shape == b0.shape


def test_aegis_gate_norm_cuda_parity() -> None:
    """Verify CUDA execution matches reference when CUDA hardware is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on host")

    b0 = torch.randn(2, 8, 32, device="cuda", dtype=torch.float32)
    b1 = torch.randn(2, 8, 32, device="cuda", dtype=torch.float32)
    weights = torch.randn(2, 8, 2, device="cuda", dtype=torch.float32)

    out_cuda = aegis_gate_norm(b0, b1, weights)
    out_ref = aegis_gate_norm_pytorch_ref(b0, b1, weights)
    torch.testing.assert_close(out_cuda, out_ref, atol=1e-4, rtol=1e-4)


def test_no_forbidden_patterns() -> None:
    """Verify absence of forbidden torch.cuda.amp or unsynchronized time.time()."""
    for mod in [agn_pkg, agn_autograd]:
        src = inspect.getsource(mod)
        assert "torch.cuda.amp" not in src, f"Deprecated torch.cuda.amp found in {mod.__name__}"
        assert "time.time()" not in src, f"Unsynchronized time.time() found in {mod.__name__}"
