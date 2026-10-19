# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Graph Surgery

AST and module-tree manipulation routines performing in-place surgery on neural network layers.
"""

from __future__ import annotations

from neuralops.core._attention_surgery import (
    PROJ_REGEX_PATTERNS,
    FUSED_QKV_REGEX,
    compute_reconstruction_error,
    slice_fused_qkv,
    split_and_set_qkv_weights,
    adapt_decoder_attention_forward,
    apply_backbone_surgery_hooks,
    calibrate_surgically_replaced_model,
    calibrate_attention_subspace,
    heal_replaced_model,
)

__all__ = [
    "PROJ_REGEX_PATTERNS",
    "FUSED_QKV_REGEX",
    "compute_reconstruction_error",
    "slice_fused_qkv",
    "split_and_set_qkv_weights",
    "adapt_decoder_attention_forward",
    "apply_backbone_surgery_hooks",
    "calibrate_surgically_replaced_model",
    "calibrate_attention_subspace",
    "heal_replaced_model",
]
