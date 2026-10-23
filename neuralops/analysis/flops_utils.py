#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Canonical FLOPs Estimation Utilities.

Provides standard analytical floating-point operation (FLOP) estimation for
attention mechanisms, linear projections, and transformer blocks.
"""

from __future__ import annotations

from typing import Any
import torch.nn as nn


def _compute_flops_estimate(
    attention_type: str = "full",
    batch_size: int = 1,
    seq_len: int = 512,
    hidden_dim: int = 768,
    num_heads: int = 12,
    head_dim: int | None = None,
    is_causal: bool = False,
    module: nn.Module | None = None,
    **kwargs: Any,
) -> int:
    """Analytical FLOP estimate for an attention or module forward pass."""
    b = max(1, int(batch_size))
    s = max(1, int(seq_len))
    h_dim = max(1, int(hidden_dim))
    n_heads = max(1, int(num_heads))
    d_head = int(head_dim) if head_dim is not None and head_dim > 0 else max(1, h_dim // n_heads)

    if module is not None:
        n_heads = getattr(module, "num_heads", getattr(module, "n_heads", n_heads))
        h_dim = getattr(module, "hidden_size", getattr(module, "d_model", h_dim))
        d_head = getattr(module, "head_dim", d_head)
        is_causal = getattr(module, "is_causal", getattr(module, "causal", is_causal))

    kind = str(attention_type).lower()
    causal = bool(is_causal or kind == "causal")

    if causal:
        token_pairs = s * (s + 1) // 2
    elif kind == "local":
        window = int(kwargs.get("window", 128))
        token_pairs = sum(min(s, i + window + 1) for i in range(s))
    elif kind == "linear":
        token_pairs = 2 * s
    else:
        token_pairs = s * s

    # Q, K, V projections (3 x GEMM)
    qkv_flops = 3 * (2 * b * s * h_dim * (n_heads * d_head))
    # Score matrix computation (Q @ K.T)
    score_flops = 2 * b * n_heads * token_pairs * d_head
    # Attention output computation (Softmax(Score) @ V)
    context_flops = 2 * b * n_heads * token_pairs * d_head
    # Final output projection
    out_proj_flops = 2 * b * s * (n_heads * d_head) * h_dim

    total_flops = qkv_flops + score_flops + context_flops + out_proj_flops
    return int(total_flops)


__all__ = ["_compute_flops_estimate"]
