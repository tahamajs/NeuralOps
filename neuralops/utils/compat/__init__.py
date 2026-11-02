#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Utils Compat Subpackage.
"""

from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn.functional as F

# Backward compatibility bridge for PyTorch < 2.0 (e.g. PyTorch 1.12 with CUDA 11.3)
if not hasattr(F, "scaled_dot_product_attention"):
    def _scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
        scores = query @ key.transpose(-2, -1) * scale_factor
        if is_causal:
            L, S = query.size(-2), key.size(-2)
            causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(0)
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            probs = torch.dropout(probs, dropout_p, train=False)
        return probs @ value

    F.scaled_dot_product_attention = _scaled_dot_product_attention

if not hasattr(torch, "compile"):
    def _compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
        if model is None:
            return lambda fn: fn
        return model

    torch.compile = _compile

from .aliases import install_projection_aliases
from .architectures import (
    ALLOWED_FFN_PATTERNS,
    SENSITIVE_PATTERNS,
    FamilyModuleWrapper,
    causal_mask_from_attention_mask,
    convert_mask,
    fp32_linalg,
    get_dynamic_compression_rank,
    init_output_scale,
    is_decoder_family,
    is_decoder_projection_name,
    maybe_apply_rotary,
    module_family_support,
    repeat_kv,
    should_compress_linear,
    should_compress_qwen3_5,
    wrap_for_family,
)
from .param_recovery import (
    PROTECTED_KEYWORDS,
    SVD_FACTOR_KEYWORDS,
    _is_protected_param,
    get_recovery_params,
)

__all__ = [
    "ALLOWED_FFN_PATTERNS",
    "FamilyModuleWrapper",
    "PROTECTED_KEYWORDS",
    "SENSITIVE_PATTERNS",
    "SVD_FACTOR_KEYWORDS",
    "_is_protected_param",
    "causal_mask_from_attention_mask",
    "convert_mask",
    "fp32_linalg",
    "get_dynamic_compression_rank",
    "get_recovery_params",
    "init_output_scale",
    "install_projection_aliases",
    "is_decoder_family",
    "is_decoder_projection_name",
    "maybe_apply_rotary",
    "module_family_support",
    "repeat_kv",
    "should_compress_linear",
    "should_compress_qwen3_5",
    "wrap_for_family",
]

from neuralops.utils.compat import aliases
from neuralops.utils.compat import architectures
from neuralops.utils.compat import param_recovery

