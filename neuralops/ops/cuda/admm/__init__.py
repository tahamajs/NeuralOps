#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
ADMM 2:4 Structured Pruning CUDA Kernel Operator.
"""

from __future__ import annotations

import torch

from neuralops.core.pruning_utils import apply_24_structured_mask
from neuralops.ops.cuda.admm.autograd import ADMM24Function


def admm_pruner_compress(
    W: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """ADMM 2:4 structured pruning using CUDA C++ kernel."""
    if W.dim() != 2 or dim != 1 or (W.shape[1] % 4 != 0):
        return apply_24_structured_mask(W, dim=dim)

    W_fp32 = W.contiguous().to(torch.float32)
    out = ADMM24Function.apply(W_fp32)
    return out.to(dtype=W.dtype)


__all__ = ["admm_pruner_compress", "ADMM24Function"]
