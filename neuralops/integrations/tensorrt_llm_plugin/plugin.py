#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
TensorRT-LLM Python Plugin implementation for NeuralOps Attention.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = ["neuralops_attention", "python_plugin"]

# 1. Resolve python_plugin decorator from tensorrt_llm or provide robust fallback
try:
    from tensorrt_llm.plugin import python_plugin  # type: ignore[import-not-found]
except (ImportError, AttributeError):

    def python_plugin(name: str, outputs: Optional[list[str]] = None) -> Callable[..., Any]:
        """Fallback python_plugin decorator when tensorrt_llm is not installed."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func._trt_plugin_name = name  # type: ignore[attr-defined]
            func._trt_outputs = outputs or ["attention_output"]  # type: ignore[attr-defined]
            return func

        return decorator


# 2. Define NeuralOps attention plugin operator
@python_plugin("NeuralOpsAttention", outputs=["attention_output"])
def neuralops_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    heads_first: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """
    TensorRT-LLM Python Plugin for NeuralOps Attention.

    Dispatches to subquadratic recurrent attention (e.g., GLA scan) or vectorized
    attention, computing:
        S_t = S_{t-1} + K_t^T @ V_t
        O_t = Q_t @ S_t

    Args:
        query: Query tensor of shape (batch, heads, seq, dim) or (batch, seq, dim).
        key: Key tensor of matching shape.
        value: Value tensor of matching shape.
        mask: Optional attention mask tensor (e.g. causal mask of shape (seq, seq)).
        scale: Optional softmax / attention scaling factor (defaults to dim**-0.5).
        heads_first: Whether the head dimension precedes sequence dimension (default: True).

    Returns:
        attention_output: Output tensor matching query shape.
    """
    orig_shape = query.shape
    head_dim = query.shape[-1]
    attn_scale = scale if scale is not None else (head_dim**-0.5)

    # Standardize to 4D: (batch, heads, seq, dim)
    if query.ndim == 3:
        # (B, T, D) -> (B, 1, T, D)
        q = query.unsqueeze(1)
        k = key.unsqueeze(1)
        v = value.unsqueeze(1)
        is_3d = True
    elif query.ndim == 4 and not heads_first:
        # (B, T, H, D) -> (B, H, T, D)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        is_3d = False
    else:
        # Standard (B, H, T, D)
        q, k, v = query, key, value
        is_3d = False

    q_scaled = q * attn_scale

    if mask is not None:
        # Masked attention path: (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        scores = torch.matmul(q_scaled, k.transpose(-1, -2))
        scores = scores + mask
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
    else:
        # Recurrent linear attention path:
        # State: S = K^T @ V -> (B, H, D, D)
        delta_s = torch.matmul(k.transpose(-1, -2), v)
        # Output: O = Q @ S -> (B, H, T, D)
        out = torch.matmul(q_scaled, delta_s)

    if is_3d:
        out = out.squeeze(1)
    elif query.ndim == 4 and not heads_first:
        out = out.transpose(1, 2).contiguous()

    return out
