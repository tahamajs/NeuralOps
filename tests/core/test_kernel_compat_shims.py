#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for neuralops.core.triton_kernels backward compatibility shims."""

from __future__ import annotations

import importlib
import pytest
import torch


KERNEL_SHIM_PAIRS = [
    ("neuralops.core.triton_kernels.selective_scan", "neuralops.ops.triton.selective_scan"),
    ("neuralops.core.triton_kernels.fused_rope", "neuralops.ops.triton.fused_rope"),
    ("neuralops.core.triton_kernels.gla_scan", "neuralops.ops.triton.gla_scan"),
    ("neuralops.core.triton_kernels.sliding_window", "neuralops.ops.triton.sliding_window"),
    ("neuralops.core.triton_kernels.gdn_scan", "neuralops.ops.triton.gdn_scan"),
    ("neuralops.core.triton_kernels.hgrn2_scan", "neuralops.ops.triton.hgrn2_scan"),
    ("neuralops.core.triton_kernels.fused_kd", "neuralops.ops.triton.fused_kd"),
    ("neuralops.core.triton_kernels.fused_ce", "neuralops.ops.triton.fused_ce"),
    ("neuralops.core.triton_kernels.block_sparse", "neuralops.ops.triton.block_sparse"),
    ("neuralops.core.triton_kernels.standalone.hgrn2_fused_bwd_scan", "neuralops.ops.triton.standalone.hgrn2_fused_bwd_scan"),
]


@pytest.mark.parametrize(
    "shim_mod_name,target_mod_name",
    KERNEL_SHIM_PAIRS,
    ids=[f"k_{i}" for i in range(len(KERNEL_SHIM_PAIRS))],
)
def test_shim_exports_identical_symbols(shim_mod_name: str, target_mod_name: str) -> None:
    shim = importlib.import_module(shim_mod_name)
    target = importlib.import_module(target_mod_name)

    target_symbols = getattr(target, "__all__", [name for name in dir(target) if not name.startswith("_")])
    assert len(target_symbols) > 0, f"{target_mod_name} has no exported symbols"

    for symbol in target_symbols:
        assert hasattr(shim, symbol), f"Shim {shim_mod_name} missing symbol '{symbol}' from {target_mod_name}"
        assert getattr(shim, symbol) is getattr(target, symbol), f"Symbol mismatch in {shim_mod_name}.{symbol}"


def test_top_level_shim_package() -> None:
    import neuralops.core.triton_kernels as tk
    import neuralops.ops.triton as target

    assert hasattr(tk, "HAS_TRITON")
    assert hasattr(tk, "triton_dispatch")
    assert tk.HAS_TRITON == target._base.HAS_TRITON


def test_sliding_window_pytorch_reference() -> None:
    from neuralops.core.triton_kernels.sliding_window import sliding_window_pytorch

    # Test small synthetic tensor through PyTorch reference
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    out = sliding_window_pytorch(q, k, v, window_size=4)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_fused_kd_pytorch_reference() -> None:
    from neuralops.core.triton_kernels.fused_kd import fused_kd_loss_pytorch

    student_logits = torch.randn(4, 10)
    teacher_logits = torch.randn(4, 10)
    loss = fused_kd_loss_pytorch(student_logits, teacher_logits, temperature=2.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
