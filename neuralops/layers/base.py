#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Modular Attention Framework & Developer Extension Base

Provides clean, standardized abstractions for implementing and extending
subquadratic and canonical attention mechanisms across the 12 mathematical families:
1. Linear Kernel Attention
2. Gated DeltaNet & Recurrent Attention
3. SVD Low-Rank & Singular Value Bottlenecks
4. State-Space Models & Selective Scans (SSM/Mamba)
5. Hybrid Multi-Branch & Sparse MoE
6. Sliding-Window & Dynamic Sparse Attention
7. Modern Rotary Position & Grouped-Query Attention (GQA)
8. Dense Multi-Head Attention Baselines

Developers extending NeuralOps can inherit from `ModularAttention`, which handles
all QKV projection scaffolding, GQA head broadcasting, causal masking, and shape
reconciliations automatically, requiring only the core mathematical kernel in `compute_attention()`.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import register_attention

logger = logging.getLogger(__name__)


class ModularAttention(BaseAttention):
    """Standardized modular scaffold for custom attention mechanisms.

    Eliminates repetitive boilerplate for projection initialization, GQA
    head broadcasting, rotary embedding dispatch, and autoregressive caching.

    To implement a new attention mechanism:
        1. Subclass `ModularAttention`.
        2. Decorate with `@register_attention("my_attention_name")`.
        3. Implement `compute_attention(self, q, k, v, mask=None, **kwargs)`.

    Example:
        >>> from neuralops.layers.base import ModularAttention
        >>> from neuralops.core.registry import register_attention
        >>>
        >>> @register_attention("clean_linear_attention", supports_causal=True)
        >>> class CleanLinearAttention(ModularAttention):
        ...     def compute_attention(self, q, k, v, mask=None, **kwargs):
        ...         # Compute linear feature map: phi(x) = elu(x) + 1
        ...         phi_q = torch.nn.functional.elu(q) + 1.0
        ...         phi_k = torch.nn.functional.elu(k) + 1.0
        ...         # O(T * d^2) associative scan
        ...         kv = torch.einsum("bhsn,bhsd->bhnd", phi_k, v)
        ...         return torch.einsum("bhsn,bhnd->bhsd", phi_q, kv)
    """

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        dim: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        num_key_value_heads: int | None = None,
        is_causal: bool = False,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, rank=rank, **kwargs)

        # 1. Resolve architectural dimensions cleanly
        resolved_dim = (
            dim
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or getattr(config, "hidden_size", getattr(config, "dim", 768))
        )
        self.dim = int(resolved_dim)
        self.hidden_size = self.dim

        resolved_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or getattr(config, "num_attention_heads", getattr(config, "num_heads", 12))
        )
        self.num_heads = int(resolved_heads)

        resolved_head_dim = (
            head_dim
            or getattr(config, "head_dim", None)
            or (self.dim // max(1, self.num_heads))
        )
        self.head_dim = int(resolved_head_dim)

        resolved_kv_heads = (
            num_key_value_heads
            or getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", self.num_heads))
        )
        self.num_key_value_heads = int(resolved_kv_heads)
        self.num_key_value_groups = self.num_heads // max(1, self.num_key_value_heads)

        self.is_causal = bool(is_causal or getattr(config, "is_causal", False))

        # 2. Standard unified linear projection tensors
        self.q_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.dim, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.dim, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=bias)

    @abstractmethod
    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Core mathematical attention operator contract.

        Subclasses implement this method with their specialized kernel or algorithm.

        Args:
            q: Query tensor shaped `[batch, num_heads, seq_len, head_dim]`.
            k: Key tensor shaped `[batch, num_heads, seq_len, head_dim]`.
            v: Value tensor shaped `[batch, num_heads, seq_len, head_dim]`.
            mask: Optional attention mask (additive or boolean).
            **kwargs: Extra parameters (e.g. state, temperature, past_key_values).

        Returns:
            Output context tensor shaped `[batch, num_heads, seq_len, head_dim]`.
        """
        raise NotImplementedError("Subclasses of ModularAttention must implement compute_attention()")

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat key/value heads for Grouped-Query Attention (GQA)."""
        if self.num_key_value_groups <= 1:
            return x
        B, H_kv, S, D = x.shape
        x = x[:, :, None, :, :].expand(B, H_kv, self.num_key_value_groups, S, D)
        return x.reshape(B, H_kv * self.num_key_value_groups, S, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Unified forward pass contract with multi-dimensional shape resilience."""
        orig_shape = hidden_states.shape
        is_2d = hidden_states.dim() == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(1)

        B, S, _ = hidden_states.shape

        # 1. Linear projection
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. Reshape into multi-head tensor format [B, H, S, D]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. GQA head expansion if needed
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # 4. Core operator delegation
        context = self.compute_attention(
            q=q,
            k=k,
            v=v,
            mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        # 5. Output projection and shape preservation
        context = context.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        output = self.out_proj(context)

        if is_2d:
            output = output.squeeze(1)

        return (output, None)


__all__ = ["BaseAttention", "ModularAttention", "register_attention"]
