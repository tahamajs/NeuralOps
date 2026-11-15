#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Scaled Dot-Product Attention (SDPA) Backend Dispatch & Utilities.

Provides rigorous backend selection (flash attention vs memory-efficient vs math)
based on tensor hardware device, floating-point precision, head dimensions, and masking layout.
"""

from __future__ import annotations

import contextlib
import logging
from enum import Enum
from typing import Any, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SDPABackend(str, Enum):
    """Enumeration of supported Scaled Dot-Product Attention (SDPA) execution backends."""

    FLASH = "flash"
    MEM_EFFICIENT = "mem_efficient"
    MATH = "math"
    AUTO = "auto"


def select_sdpa_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> SDPABackend:
    """Determines the mathematically and architecturally optimal SDPA backend.

    Selection Criteria:
    - FlashAttention: Requires CUDA, fp16/bf16, head_dim <= 256 (multiple of 8), and no arbitrary mask.
    - Memory-Efficient: Requires CUDA, supports fp16/bf16/fp32 and arbitrary mask tensors.
    - Math: Fallback for CPU, MPS, double precision, or unsupported head dimensions.

    Args:
        q: Query tensor (batch, heads, seq_len_q, head_dim).
        k: Key tensor (batch, heads, seq_len_k, head_dim).
        v: Value tensor (batch, heads, seq_len_k, head_dim).
        attn_mask: Optional attention mask tensor.
        is_causal: Whether strict autoregressive causal masking is requested.

    Returns:
        The selected SDPABackend enum.
    """
    if not q.is_cuda:
        return SDPABackend.MATH

    dtype = q.dtype
    head_dim = q.shape[-1]
    has_arbitrary_mask = attn_mask is not None

    # Check FlashAttention eligibility
    is_half_precision = dtype in (torch.float16, torch.bfloat16)
    valid_flash_dim = head_dim <= 256 and (head_dim % 8 == 0)
    flash_eligible = is_half_precision and valid_flash_dim and not has_arbitrary_mask

    if flash_eligible:
        return SDPABackend.FLASH

    # Check Memory-Efficient eligibility
    mem_eff_eligible = dtype in (torch.float16, torch.bfloat16, torch.float32)
    if mem_eff_eligible:
        return SDPABackend.MEM_EFFICIENT

    return SDPABackend.MATH


@contextlib.contextmanager
def sdpa_backend_context(
    backend: SDPABackend | str,
) -> Generator[None, None, None]:
    """Context manager configuring PyTorch SDPA kernel backends.

    Args:
        backend: The target SDPABackend enum or string name.
    """
    backend_val = SDPABackend(backend) if isinstance(backend, str) else backend

    if backend_val == SDPABackend.AUTO:
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        enable_flash = backend_val == SDPABackend.FLASH
        enable_mem_eff = backend_val == SDPABackend.MEM_EFFICIENT
        enable_math = backend_val == SDPABackend.MATH

        backends: list[Any] = []
        if enable_flash:
            backends.append(SDPBackend.FLASH_ATTENTION)
        if enable_mem_eff:
            backends.append(SDPBackend.EFFICIENT_ATTENTION)
        if enable_math:
            backends.append(SDPBackend.MATH)

        if backends:
            with sdpa_kernel(backends):
                yield
        else:
            yield
    except (ImportError, AttributeError):
        # Fallback for environments lacking torch.nn.attention.sdpa_kernel
        yield


def sdpa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    backend: SDPABackend | str | None = None,
) -> torch.Tensor:
    """Executes scaled dot-product attention under the chosen or auto-selected backend.

    Args:
        q: Query tensor of shape (batch, heads, seq_len_q, head_dim).
        k: Key tensor of shape (batch, heads, seq_len_k, head_dim).
        v: Value tensor of shape (batch, heads, seq_len_k, head_dim).
        attn_mask: Optional attention mask tensor.
        dropout_p: Dropout probability.
        is_causal: Whether causal masking is applied.
        backend: Optional backend override; if None, automatically selected.

    Returns:
        Attention output tensor of shape (batch, heads, seq_len_q, head_dim).
    """
    chosen_backend = (
        SDPABackend(backend)
        if backend is not None and backend != "auto"
        else select_sdpa_backend(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    )

    with sdpa_backend_context(chosen_backend):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )


class SDPAContext:
    """Utility wrapper for configuring SDPA backend execution parameters."""

    def __init__(self, preferred_backend: SDPABackend = SDPABackend.AUTO) -> None:
        """Initializes SDPAContext with a preferred default backend."""
        self.preferred_backend: SDPABackend = preferred_backend

    def resolve(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> SDPABackend:
        """Resolves the active backend given input shapes and properties."""
        if self.preferred_backend != SDPABackend.AUTO:
            return self.preferred_backend
        return select_sdpa_backend(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


__all__ = [
    "SDPABackend",
    "SDPAContext",
    "sdpa_backend_context",
    "sdpa_forward",
    "select_sdpa_backend",
]
