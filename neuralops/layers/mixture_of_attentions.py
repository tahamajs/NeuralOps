#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compatibility alias for Mixture of Attentions (MoA)."""

from neuralops.layers.moa import (
    AttentionEntropyProfile,
    MixtureOfAttentions,
    MoARouter,
    compute_attention_entropy,
    profile_layer_sensitivity,
)

__all__ = [
    "MixtureOfAttentions",
    "MoARouter",
    "compute_attention_entropy",
    "profile_layer_sensitivity",
    "AttentionEntropyProfile",
]
