#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
AWQ CUDA Kernel Operator.
"""

from __future__ import annotations

from typing import Optional

import torch
from neuralops.ops.cuda.awq.autograd import AWQFunction
from neuralops.ops.triton.quantize_awq import _compute_awq_scales


def awq_compress(
    W: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
    act_scales: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """AWQ compression using CUDA C++ kernel."""
    effective_scale = _compute_awq_scales(
        W, group_size=group_size, bits=bits, act_scales=act_scales, alpha=alpha
    )

    W_fp32 = W.contiguous().to(torch.float32)
    scales_fp32 = effective_scale.contiguous().to(torch.float32)

    out = AWQFunction.apply(W_fp32, scales_fp32, group_size)
    return out.reshape(W.shape).to(dtype=W.dtype)


__all__ = ["awq_compress", "AWQFunction"]
