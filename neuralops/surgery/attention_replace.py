#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Surgery and Layer Replacement Infrastructure (Legacy Shim).

This module provides backwards compatibility for legacy attention replacement call sites.
New implementations should directly utilize neuralops.surgery.director.UniversalSurgeryDirector.
"""

from __future__ import annotations

import logging
from typing import Any
import torch.nn as nn

from neuralops.core._attention_surgery import (
    FUSED_QKV_REGEX,
    PROJ_REGEX_PATTERNS,
    _copy_projection_weights,
    _find_submodule_by_regex,
    _heal_replaced_model_impl,
    _slice_fused_qkv_weights,
    _svd_factorize_weights,
    adapt_decoder_attention_forward,
    apply_backbone_surgery_hooks,
    calibrate_attention_subspace,
    calibrate_surgically_replaced_model,
    compute_reconstruction_error,
    heal_replaced_model,
    slice_fused_qkv,
    split_and_set_qkv_weights,
)
from neuralops.core.replace_attention import replace_attention_in_model
from neuralops.surgery.director import UniversalSurgeryDirector
from neuralops.surgery.layer_replacement import replace_attention_recursive

logger = logging.getLogger("neuralops.surgery.attention_replace")


from neuralops.api import replace_attention


__all__ = [
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
    "_heal_replaced_model_impl",
    "replace_attention_in_model",
    "replace_attention_recursive",
    "replace_attention",
    "UniversalSurgeryDirector",
]
