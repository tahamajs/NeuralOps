# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Speculative Verification Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "spec_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "speculative_verify.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_speculative_verify",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["speculative_verify_fwd"],
        )
    return _MODULE


def speculative_verify_pytorch_ref(
    draft_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation."""
    match = (draft_tokens == target_tokens)
    accept_mask = match.cumprod(dim=-1).bool()
    accept_count = accept_mask.sum(dim=-1).to(torch.int32)
    return accept_mask, accept_count


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.elementwise import speculative_verify_triton
        return speculative_verify_triton
    except (ImportError, AttributeError):
        return None


def speculative_verify(
    draft_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3-Level Dispatched Speculative Verification."""
    def cuda_fn(draft, target, **kwargs):
        mod = _get_module()
        res = mod.speculative_verify_fwd(draft.contiguous(), target.contiguous())
        return res[0], res[1]

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        speculative_verify_pytorch_ref,
        draft_tokens,
        target_tokens,
        kernel_name="speculative_verify",
    )
