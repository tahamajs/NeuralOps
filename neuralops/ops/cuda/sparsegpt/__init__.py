#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SparseGPT Pruning CUDA Kernel Operator.
"""

from __future__ import annotations

from typing import Optional

import torch
from neuralops.ops.cuda.sparsegpt.autograd import SparseGPTFunction


def sparsegpt_compress(
    W: torch.Tensor,
    sparsity: float = 0.5,
    diag_h_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SparseGPT second-order pruning using CUDA C++ kernel."""
    if W.dim() != 2:
        raise ValueError(f"sparsegpt_compress requires 2D weight matrix, got shape {W.shape}")

    W_fp32 = W.contiguous().to(torch.float32)
    diag_fp32 = diag_h_inv.contiguous().to(torch.float32) if diag_h_inv is not None else None

    out = SparseGPTFunction.apply(W_fp32, float(sparsity), diag_fp32)
    return out.to(dtype=W.dtype)


__all__ = ["sparsegpt_compress", "SparseGPTFunction"]
