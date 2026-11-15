#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Samba hybrid SSM-attention architecture layer facade."""

from __future__ import annotations

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.ssm.samba_m3 import SAMBAM3Block, SpectralMamba

SambaBlock = SAMBAM3Block
ATTENTION_REGISTRY.register("samba", is_alias=True)(SambaBlock)

__all__ = ["SAMBAM3Block", "SambaBlock", "SpectralMamba"]
