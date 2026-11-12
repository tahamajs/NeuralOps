#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Graph Surgery (Legacy Shim).

Deprecated: All core surgery routines have moved to `neuralops.models.surgery`.
This module is retained for backward compatibility.
"""

from __future__ import annotations

import warnings
from typing import Any

warnings.warn(
    "neuralops.core._attention_surgery is deprecated and will be removed in a future release. "
    "Please use neuralops.models.surgery instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.models.surgery import (
    FUSED_QKV_REGEX,
    PROJ_REGEX_PATTERNS,
    SubspaceAnchorWrapper,
    SurgeryTransactionError,
    _copy_projection_weights,
    _find_submodule_by_regex,
    _slice_fused_qkv_weights,
    _svd_factorize_weights,
    adapt_decoder_attention_forward,
    apply_backbone_surgery_hooks,
    calibrate_attention_subspace,
    calibrate_surgically_replaced_model,
    compute_attention_weights,
    compute_reconstruction_error,
    heal_replaced_model,
    replace_attention,
    replace_attention_in_model,
    slice_fused_qkv,
    split_and_set_qkv_weights,
    transplant_model_attention,
)


def __getattr__(name: str) -> Any:
    import neuralops.models.surgery as _surgery

    if hasattr(_surgery, name):
        return getattr(_surgery, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "replace_attention",
    "replace_attention_in_model",
    "transplant_model_attention",
    "SurgeryTransactionError",
    "compute_attention_weights",
    "PROJ_REGEX_PATTERNS",
    "FUSED_QKV_REGEX",
    "_find_submodule_by_regex",
    "compute_reconstruction_error",
    "_slice_fused_qkv_weights",
    "slice_fused_qkv",
    "_copy_projection_weights",
    "split_and_set_qkv_weights",
    "_svd_factorize_weights",
    "adapt_decoder_attention_forward",
    "apply_backbone_surgery_hooks",
    "calibrate_surgically_replaced_model",
    "calibrate_attention_subspace",
    "heal_replaced_model",
    "SubspaceAnchorWrapper",
]
