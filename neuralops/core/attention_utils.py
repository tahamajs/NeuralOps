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
    "apply_rope_after_up_projection",
    "_find_submodule_by_regex",
    "_svd_factorize_weights",
    "PROJ_REGEX_PATTERNS",
    "FUSED_QKV_REGEX",
]
