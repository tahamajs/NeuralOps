#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Multi-Head Latent Attention (MLA) layer facade."""

from neuralops.layers.modern.mla_attention import MLAAttention
from neuralops.surgery.mla_surgery import (
    DeepSeekMLASurgeryBridge,
    DualStreamLatentState,
    apply_rotary_pos_emb,
)

__all__ = [
    "MLAAttention",
    "DeepSeekMLASurgeryBridge",
    "DualStreamLatentState",
    "apply_rotary_pos_emb",
]
