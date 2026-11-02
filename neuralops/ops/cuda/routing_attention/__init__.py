#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Routing Attention CUDA Kernel Operator.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from neuralops.ops.cuda.routing_attention.autograd import RoutingAttentionFunction
from neuralops.ops.triton.routing_attention_triton import _kmeans_assign, _routing_attention_pytorch

routing_attention_ref = _routing_attention_pytorch


def routing_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_clusters: int = 16,
    n_iters: int = 3,
    is_causal: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Routing Attention forward pass using CUDA C++ kernel."""
    B, H, L, d = q.shape
    scale = 1.0 / math.sqrt(int(d))

    K_num = min(max(1, num_clusters), L)
    joint_feat = F.normalize(q + k, p=2, dim=-1)
    cluster_ids = _kmeans_assign(joint_feat, K_num, n_iters)

    out = RoutingAttentionFunction.apply(q, k, v, cluster_ids, scale)
    return out.to(dtype=q.dtype)


__all__ = [
    "routing_attention_forward",
    "RoutingAttentionFunction",
    "routing_attention_ref",
]
