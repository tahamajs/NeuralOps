#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Adaptation Utilities

Centralized helpers for attention module instantiation, tensor transformation, and weight adaptation.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core._attention_injectors import _inject_attention_loop
from neuralops.core._attention_surgery import (
    FUSED_QKV_REGEX,
    PROJ_REGEX_PATTERNS,
    _copy_projection_weights,
    _find_submodule_by_regex,
    _slice_fused_qkv_weights,
    _svd_factorize_weights,
    compute_reconstruction_error,
    slice_fused_qkv,
    split_and_set_qkv_weights,
)
from neuralops.core.common import softpick
from neuralops.core.utils.gqa import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.rope import _rotate_half, apply_rope_after_up_projection

logger = logging.getLogger(__name__)


def inject_attention(
    model: nn.Module,
    attention_type: str,
    rank: int | None = None,
    copy_weights: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Universal attention layer replacement engine for HuggingFace Transformers."""
    return _inject_attention_loop(
        model=model,
        attention_type=attention_type,
        rank=rank,
        copy_weights=copy_weights,
        **kwargs,
    )


def available_attention_types() -> list[str]:
    """Return all registered attention mechanism identifiers."""
    return sorted(list(ATTENTION_REGISTRY.keys()))


def softcap(scores: torch.Tensor, cap: float = 50.0) -> torch.Tensor:
    """Apply softcapping to attention scores (Gemma-2, Grok)."""
    return cap * torch.tanh(scores / cap)


_DETERMINISTIC_ATTENTION: bool = False


def enable_deterministic_attention(enabled: bool = True) -> bool:
    """Toggle deterministic backward pass and execution for attention operations.

    Also sets torch.use_deterministic_algorithms if available.
    Returns previous deterministic state.
    """
    global _DETERMINISTIC_ATTENTION
    prev = _DETERMINISTIC_ATTENTION
    _DETERMINISTIC_ATTENTION = enabled
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=True)
    except Exception:
        pass
    return prev


def is_deterministic_attention() -> bool:
    """Check whether deterministic attention is active."""
    return _DETERMINISTIC_ATTENTION


def safe_softmax(logits: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable softmax preventing underflow/overflow and NaN gradients."""
    m = torch.amax(logits, dim=dim, keepdim=True)
    exp = torch.exp(logits - m)
    denom = torch.sum(exp, dim=dim, keepdim=True).clamp(min=eps)
    return exp / denom


def safe_log(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """Numerically stable logarithm preventing log(0) -inf."""
    return torch.log(x.clamp(min=eps))


def safe_mask_value(dtype: torch.dtype) -> float:
    """Dtype-aware mask sentinel value."""
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


def normalize_attention_mask(
    mask: torch.Tensor | None,
    B: int,
    H: int,
    N: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Normalize attention mask to (B, 1, N, 1) float mask where 1.0 = valid, 0.0 = masked."""
    if mask is None:
        return None

    if mask.dim() == 2:
        if mask.dtype == torch.bool:
            m = mask.float()
        elif mask.dtype.is_floating_point:
            m = (mask > -1e4).float()
        else:
            m = (mask > 0).float()
        m = m.view(B, 1, N, 1)

    elif mask.dim() == 3:
        if mask.shape[1] == H:
            m = mask
            if m.dtype == torch.bool:
                m = m.float()
            elif m.dtype.is_floating_point:
                m = (m > -1e4).float()
            else:
                m = (m > 0).float()
            m = m.unsqueeze(-1)
        else:
            m = mask
            if m.dtype == torch.bool:
                m = m.float()
            elif m.dtype.is_floating_point:
                m = (m > -1e4).float()
            else:
                m = (m > 0).float()
            m = m.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
            m = m.unsqueeze(1)

    elif mask.dim() == 4:
        m = mask
        if m.dtype == torch.bool:
            m = m.float()
        elif m.dtype.is_floating_point:
            m = (m > -1e4).float()
        else:
            m = (m > 0).float()
        if m.shape[-2] == 1:
            m = m.permute(0, 1, 3, 2)
        elif m.shape[-2] == N and m.shape[-1] == N:
            m = m.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)

    else:
        return None

    return m.to(device=device, dtype=dtype)


def broadcast_cos_sin(
    cos: torch.Tensor, sin: torch.Tensor, q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast cos/sin embeddings to match query tensor rank."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    past_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding (RoPE) with independent Q and K slicing and past_len support."""
    if cos is None or sin is None:
        return q, k

    cos, sin = broadcast_cos_sin(cos, sin, q)

    L = q.shape[-2]
    S = cos.shape[-2]
    if past_len > 0 and S >= past_len + L:
        cos_q = cos[..., past_len : past_len + L, :]
        sin_q = sin[..., past_len : past_len + L, :]
    elif S >= L:
        cos_q = cos[..., :L, :]
        sin_q = sin[..., :L, :]
    else:
        repeats = (L + S - 1) // S
        cos_q = cos.repeat(1, 1, repeats, 1)[..., :L, :]
        sin_q = sin.repeat(1, 1, repeats, 1)[..., :L, :]

    N_k = k.shape[-2]
    if S >= N_k:
        cos_k = cos[..., :N_k, :]
        sin_k = sin[..., :N_k, :]
    else:
        repeats = (N_k + S - 1) // S
        cos_k = cos.repeat(1, 1, repeats, 1)[..., :N_k, :]
        sin_k = sin.repeat(1, 1, repeats, 1)[..., :N_k, :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_e = (q_rot * cos_q.to(q_rot.dtype)) + (
            _rotate_half(q_rot) * sin_q.to(q_rot.dtype)
        )
        k_rot_e = (k_rot * cos_k.to(k_rot.dtype)) + (
            _rotate_half(k_rot) * sin_k.to(k_rot.dtype)
        )
        q = torch.cat([q_rot_e, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_e
        k = torch.cat([k_rot_e, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_e
    else:
        cos_q_cut = cos_q[..., :head_dim].to(q.dtype)
        sin_q_cut = sin_q[..., :head_dim].to(q.dtype)
        q = (q * cos_q_cut) + (_rotate_half(q) * sin_q_cut)

        cos_k_cut = cos_k[..., :head_dim].to(k.dtype)
        sin_k_cut = sin_k[..., :head_dim].to(k.dtype)
        k = (k * cos_k_cut) + (_rotate_half(k) * sin_k_cut)

    return q, k


from neuralops.core.peft.lora import LoRALinear


__all__ = [
    "inject_attention",
    "available_attention_types",
    "compute_reconstruction_error",
    "_slice_fused_qkv_weights",
    "slice_fused_qkv",
    "_copy_projection_weights",
    "split_and_set_qkv_weights",
    "repeat_kv",
    "softpick",
    "softcap",
    "enable_deterministic_attention",
    "is_deterministic_attention",
    "safe_softmax",
    "safe_log",
    "apply_rope_after_up_projection",
    "_find_submodule_by_regex",
    "_svd_factorize_weights",
    "PROJ_REGEX_PATTERNS",
    "FUSED_QKV_REGEX",
    "safe_mask_value",
    "normalize_attention_mask",
    "broadcast_cos_sin",
    "apply_rotary_pos_emb",
    "LoRALinear",
]
