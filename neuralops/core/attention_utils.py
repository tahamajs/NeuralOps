"""
Unified Attention Injection & Weight Adaptation Engine for Foundation Models.

Public API entry points:
  - inject_attention
  - available_attention_types
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

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
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import apply_rope_after_up_projection

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
]
