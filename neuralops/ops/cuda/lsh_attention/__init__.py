#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
LSH Attention CUDA Kernel Operator.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from neuralops.ops.cuda.lsh_attention.autograd import LSHAttentionFunction
from neuralops.ops.triton.lsh_attention_triton import _compute_lsh_buckets


def lsh_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    rounds: int = 4,
    num_buckets: int = 16,
    proj_matrices: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """LSH Attention forward pass using CUDA C++ kernel."""
    B, H, N_q, d = q.shape
    device = q.device
    dtype = q.dtype
    scale = 1.0 / math.sqrt(int(d))

    num_planes = max(1, int(math.ceil(math.log2(max(2, num_buckets)))))
    if proj_matrices is None:
        generator = torch.Generator(device=device).manual_seed(0xCAFE)
        proj_matrices = torch.randn(rounds, d, num_planes, device=device, dtype=dtype, generator=generator)

    rounds_eff = min(rounds, proj_matrices.size(0))
    accum = torch.zeros_like(q)

    for r in range(rounds_eff):
        proj_r = proj_matrices[r].to(device=device, dtype=dtype)
        q_hash = _compute_lsh_buckets(q, proj_r, num_buckets)
        k_hash = _compute_lsh_buckets(k, proj_r, num_buckets)

        round_out = LSHAttentionFunction.apply(q, k, v, q_hash, k_hash, scale)
        accum = accum + round_out

    return (accum / float(rounds_eff)).to(dtype=dtype)


__all__ = ["lsh_attention_forward", "LSHAttentionFunction"]
