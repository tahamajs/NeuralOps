#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Neural Architecture Surgery Facade & Attention Transplantation Pipeline.

Exposes end-to-end model surgery, in-place attention transplantation, and
subspace healing primitives across all 14 attention families and 8 foundation
model topologies.
"""

from __future__ import annotations

from typing import Any, List, Optional
import torch.nn as nn

from neuralops.api import (
    replace_attention,
    transplant_attention,
)
from neuralops.surgery.scanner import (
    AttentionTargetSpec,
    UniversalArchitectureScanner,
)
from neuralops.surgery.endoskeletal_shell import (
    UniversalAttentionBridge,
    transfer_pretrained_qkv_weights,
)
from neuralops.surgery.shadow_twin import (
    check_surgery_divergence,
    collect_surgery_distillation_loss,
    finalize_surgery,
    set_surgery_alpha,
    set_surgery_stage,
)
from neuralops.surgery.universal_calibrator import (
    UniversalActivationCalibrator,
    compress_model_universal,
    transplant_and_compress,
)
from neuralops.surgery.healing import (
    calibrate_attention_subspace,
    heal_replaced_model,
)

__all__ = [
    "replace_attention",
    "transplant_attention",
    "AttentionTargetSpec",
    "UniversalArchitectureScanner",
    "UniversalAttentionBridge",
    "transfer_pretrained_qkv_weights",
    "check_surgery_divergence",
    "collect_surgery_distillation_loss",
    "finalize_surgery",
    "set_surgery_alpha",
    "set_surgery_stage",
    "UniversalActivationCalibrator",
    "compress_model_universal",
    "transplant_and_compress",
    "calibrate_attention_subspace",
    "heal_replaced_model",
]
