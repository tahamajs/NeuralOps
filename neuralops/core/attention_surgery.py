"""
Attention Graph Surgery

AST and module-tree manipulation routines performing in-place surgery on neural network layers.

Author: Taha Majlesi
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
