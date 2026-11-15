# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for ADMM 2:4 structured pruning CUDA kernel & CPU fallback (Wave 9 / W9-K4).
"""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch
from torch.amp import autocast

from neuralops.core.pruning_utils import apply_24_structured_mask
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.admm import ADMM24Function, admm_pruner_compress


def test_admm_cpu_fallback_basic() -> None:
    """Verify ADMM 2:4 structured pruning runs on CPU and returns tensor."""
    w = torch.randn(8, 16, dtype=torch.float32)
    out = admm_pruner_compress(w)

    assert isinstance(out, torch.Tensor)
    assert not isinstance(out, tuple)
    assert out.shape == w.shape
    assert out.dtype == w.dtype
    assert out.device == w.device


def test_admm_cpu_numerical_parity() -> None:
    """Verify ADMM output matches PyTorch reference 2:4 structured pruning."""
    w = torch.randn(12, 16, dtype=torch.float32)
    out = admm_pruner_compress(w, dim=1)
    ref = apply_24_structured_mask(w, dim=1)
    ref_tensor = ref[0] if isinstance(ref, tuple) else ref

    torch.testing.assert_close(out, ref_tensor)


def test_admm_sparsity_ratio() -> None:
    """Verify that exactly 50% sparsity (2 zeros per group of 4) is produced."""
    w = torch.randn(4, 16, dtype=torch.float32)
    out = admm_pruner_compress(w)

    # For each group of 4 along dim=1, exactly 2 elements must be zero
    out_groups = out.view(-1, 4)
    zeros_per_group = (out_groups == 0.0).sum(dim=1)
    assert (zeros_per_group == 2).all().item()


def test_admm_autograd_ste() -> None:
    """Verify ADMM straight-through estimator backward pass."""
    w = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
    out = admm_pruner_compress(w)
    loss = (out * 2.0).sum()
    loss.backward()

    assert w.grad is not None
    assert w.grad.shape == w.shape
    # Under STE, grad_output flows directly back
    expected_grad = torch.full_like(w, 2.0)
    torch.testing.assert_close(w.grad, expected_grad)


def test_admm_amp_compliance() -> None:
    """Verify torch.amp compliance (Rule 2.1) across float16 and bfloat16."""
    w_fp16 = torch.randn(8, 16, dtype=torch.float16)
    out_fp16 = admm_pruner_compress(w_fp16)
    assert out_fp16.dtype == torch.float16

    w_bf16 = torch.randn(8, 16, dtype=torch.bfloat16)
    out_bf16 = admm_pruner_compress(w_bf16)
    assert out_bf16.dtype == torch.bfloat16

    # Test within torch.amp.autocast context
    with autocast("cpu", dtype=torch.bfloat16):
        w = torch.randn(8, 16)
        out = admm_pruner_compress(w)
        assert isinstance(out, torch.Tensor)


def test_admm_arbitrary_dim_and_odd_shapes() -> None:
    """Verify graceful handling when shape or dim is not divisible by 4 or dim!=1."""
    # dim=0 with 2D tensor
    w0 = torch.randn(16, 8)
    out0 = admm_pruner_compress(w0, dim=0)
    assert out0.shape == w0.shape

    # Not divisible by 4 along dim=1
    w_odd = torch.randn(8, 15)
    out_odd = admm_pruner_compress(w_odd)
    assert out_odd.shape == w_odd.shape


def test_admm_top_level_dispatch() -> None:
    """Verify top-level dispatch noc.admm works seamlessly."""
    w = torch.randn(8, 16, dtype=torch.float32)
    out = noc.admm(w)
    assert isinstance(out, torch.Tensor)
    assert out.shape == w.shape


def test_no_forbidden_patterns() -> None:
    """Verify absence of torch.cuda.amp, raw time.time(), etc."""
    import neuralops.ops.cuda.admm as admm_pkg
    autograd_mod = importlib.import_module("neuralops.ops.cuda.admm.autograd")

    for mod in [admm_pkg, autograd_mod]:
        src = inspect.getsource(mod)
        assert "torch.cuda.amp" not in src, f"Deprecated torch.cuda.amp found in {mod.__name__}"
        assert "time.time()" not in src, f"Unsynchronized time.time() found in {mod.__name__}"
