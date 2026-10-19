# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Shared test verification utilities for NeuralOps CUDA C++ Kernel Verification Suite.

Designed for NVIDIA Ada Lovelace SM 8.9 (RTX 4070 Laptop GPU, 8GB VRAM).
Strict numerical parity, NaN/Inf checks, multiple input shapes, memory layouts, and fallbacks.
"""

from __future__ import annotations

import pytest
import torch
from typing import Callable, Any, Tuple, Optional


def check_tensor_validity(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Asserts that a tensor contains zero NaNs and zero Infs."""
    assert not torch.isnan(tensor).any(), f"AssertionError: {name} contains NaN values!"
    assert not torch.isinf(tensor).any(), f"AssertionError: {name} contains Inf values!"


def assert_numerical_parity(
    cuda_out: torch.Tensor,
    ref_out: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    atol: Optional[float] = None,
    name: str = "kernel",
) -> None:
    """Verifies numerical parity between CUDA kernel and PyTorch reference implementation.

    Operating Law #3:
    - FP16 numerical parity requires max_diff < 0.01
    - FP32 numerical parity requires max_diff < 1e-4
    - Zero NaNs and zero Infs
    """
    check_tensor_validity(cuda_out, f"{name}_cuda_out")
    check_tensor_validity(ref_out, f"{name}_ref_out")

    if atol is None:
        if dtype == torch.float16 or dtype == torch.bfloat16:
            atol = 0.01
        else:
            atol = 1e-4

    if cuda_out.dtype == torch.bool:
        assert (cuda_out == ref_out).all(), f"AssertionError: Boolean mask mismatch in {name}!"
        return

    if torch.is_complex(cuda_out):
        diff_re = (cuda_out.real.float() - ref_out.real.float()).abs().max().item()
        diff_im = (cuda_out.imag.float() - ref_out.imag.float()).abs().max().item()
        max_diff = max(diff_re, diff_im)
    else:
        max_diff = (cuda_out.float() - ref_out.float()).abs().max().item()
    assert max_diff <= atol, (
        f"AssertionError: Numerical parity failure in {name} ({dtype})! "
        f"max_diff = {max_diff:.6e} > tolerance = {atol:.6e}"
    )


def assert_tuple_numerical_parity(
    cuda_outs: Tuple[torch.Tensor, ...],
    ref_outs: Tuple[torch.Tensor, ...],
    dtype: torch.dtype = torch.float16,
    atol: Optional[float] = None,
    name: str = "kernel",
) -> None:
    """Asserts parity across multiple returned tensors."""
    assert len(cuda_outs) == len(ref_outs), (
        f"Length mismatch: cuda returned {len(cuda_outs)} tensors, ref returned {len(ref_outs)}"
    )
    for i, (c_o, r_o) in enumerate(zip(cuda_outs, ref_outs)):
        if isinstance(c_o, torch.Tensor) and isinstance(r_o, torch.Tensor):
            assert_numerical_parity(c_o, r_o, dtype=dtype, atol=atol, name=f"{name}_output_{i}")


def create_non_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Creates a strictly non-contiguous layout of the same shape and values."""
    if tensor.dim() >= 2:
        return tensor.clone().transpose(0, 1).contiguous().transpose(0, 1)
    return tensor.clone()[::1]
