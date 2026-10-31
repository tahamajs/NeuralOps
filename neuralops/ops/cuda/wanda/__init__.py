#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Wanda Pruning CUDA Kernel Operator.
"""

from __future__ import annotations

from typing import Optional

import torch
from neuralops.ops.cuda.wanda.autograd import WandaFunction


def wanda_compress(
    W: torch.Tensor,
    sparsity: float = 0.5,
    act_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wanda pruning using CUDA C++ kernel."""
    if W.dim() != 2:
        raise ValueError(f"wanda_compress requires 2D weight matrix, got shape {W.shape}")

    W_fp32 = W.contiguous().to(torch.float32)
    act_fp32 = act_norm.contiguous().to(torch.float32) if act_norm is not None else None

    out = WandaFunction.apply(W_fp32, float(sparsity), act_fp32)
    return out.to(dtype=W.dtype)


__all__ = ["wanda_compress", "WandaFunction"]
