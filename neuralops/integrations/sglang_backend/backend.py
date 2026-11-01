#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SGLang Linear Attention Backend Implementation for NeuralOps.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["NeuralOpsLinearAttnBackend"]


class NeuralOpsLinearAttnBackend(nn.Module):
    """
    SGLang Linear Attention Backend for NeuralOps.

    Implements the SGLang linear attention interface (forward accepting mixed_qkv,
    a, b gates) with support for Mamba/Radix recurrent cache integration.
    """

    needs_cpu_seq_lens: bool = False
    supports_mis: bool = True
    requires_contiguous_prefill_state: bool = True
    uses_mamba_radix_cache: bool = True

    def __init__(
        self,
        model_runner: Any = None,
        num_heads: int = 12,
        head_dim: int = 64,
        state_dim: Optional[int] = None,
        mechanism: str = "gla_attention",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_runner = model_runner
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_dim = state_dim if state_dim is not None else head_dim
        self.mechanism = mechanism
        self.scale = head_dim**-0.5
        self._state_cache: dict[str, torch.Tensor] = {}

    def forward(
        self,
        mixed_qkv: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        forward_batch: Any = None,
        kv_cache: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Execute linear attention forward pass.

        Accepts:
            mixed_qkv: Tensor of shape (batch, seq, 3 * num_heads * head_dim) or (tokens, 3 * hidden),
                       or a tuple of (q, k, v).
            a: Optional gate / recurrent decay tensor (e.g. log-space decay for GLA / GDN).
            b: Optional input gate tensor for key/value injection.
            forward_batch: SGLang ForwardBatch metadata.
            kv_cache: Radix linear cache or pre-allocated recurrent state tensor.
        """
        # 1. Unpack Q, K, V
        if isinstance(mixed_qkv, (tuple, list)) and len(mixed_qkv) == 3:
            q, k, v = mixed_qkv
        else:
            assert isinstance(mixed_qkv, torch.Tensor), "mixed_qkv must be a Tensor or 3-tuple"
            # Split along last dimension into 3 chunks
            q, k, v = torch.chunk(mixed_qkv, 3, dim=-1)

        # Ensure consistent shapes: (B, H, T, D)
        orig_shape = q.shape
        if q.ndim == 2:
            # Flattened token representation (total_tokens, hidden_dim)
            total_tokens = q.shape[0]
            q = q.view(1, total_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(1, total_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, total_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        elif q.ndim == 3:
            # (B, T, hidden_dim)
            batch_size, seq_len, _ = q.shape
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        batch_size = q.shape[0]

        # 2. Retrieve or allocate recurrent state
        state: Optional[torch.Tensor] = None
        if isinstance(kv_cache, torch.Tensor):
            state = kv_cache
        elif hasattr(kv_cache, "state"):
            state = kv_cache.state
        elif "recurrent_state" in self._state_cache:
            state = self._state_cache["recurrent_state"]

        if state is None or state.shape[0] != batch_size:
            state = torch.zeros(
                (batch_size, self.num_heads, self.head_dim, self.state_dim),
                dtype=q.dtype,
                device=q.device,
            )

        # 3. Apply optional gating (a = decay, b = input modulation)
        if a is not None:
            # Decay existing state
            if a.ndim == 1:
                state = state * a.view(1, 1, 1, -1)
            elif a.ndim >= 2:
                state = state * a.view(batch_size, -1, 1, 1)

        if b is not None:
            # Modulate value or key
            if b.ndim == 1:
                v = v * b.view(1, 1, 1, -1)
            elif b.ndim >= 2:
                v = v * b.view(batch_size, -1, 1, 1)

        # 4. Recurrent linear update:
        # S_t = S_{t-1} + K_t^T @ V_t
        # O_t = Q_t @ S_t
        q_scaled = q * self.scale
        delta_s = torch.matmul(k.transpose(-1, -2), v)
        updated_state = state + delta_s

        # Persist updated state
        self._state_cache["recurrent_state"] = updated_state
        if kv_cache is not None and hasattr(kv_cache, "state"):
            object.__setattr__(kv_cache, "state", updated_state)

        # Compute output
        out = torch.matmul(q_scaled, updated_state)
        # Reshape to match original input
        out = out.transpose(1, 2).contiguous()
        if len(orig_shape) == 2:
            out = out.view(orig_shape[0], -1)
        elif len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], -1)

        return out

    def forward_decode(
        self,
        mixed_qkv: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        forward_batch: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode step execution for SGLang scheduler."""
        return self.forward(mixed_qkv, a=a, b=b, forward_batch=forward_batch, **kwargs)

    def forward_extend(
        self,
        mixed_qkv: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        forward_batch: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Prefill / extend step execution for SGLang scheduler."""
        return self.forward(mixed_qkv, a=a, b=b, forward_batch=forward_batch, **kwargs)

    def init_forward_metadata(self, forward_batch: Any) -> None:
        """Initialize per-batch metadata for radix cache."""
