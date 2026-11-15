#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Mixture of Experts (MoE) attention family facade."""

from __future__ import annotations

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.sparse.bam_umoe import BAMUMoEBlock
from neuralops.layers.sparse.expert_choice_moe import ExpertChoiceMoE, ExpertChoiceMoEAttention
from neuralops.layers.sparse.moe_deltakv_paged import MoEDeltaKVPagedAttention
from neuralops.layers.sparse.reap_moe import REAPMoE, REAPMoEAttention

MoEAttention = REAPMoEAttention
ATTENTION_REGISTRY.register("moe_attention", is_alias=True)(MoEAttention)

__all__ = [
    "REAPMoEAttention",
    "REAPMoE",
    "ExpertChoiceMoEAttention",
    "ExpertChoiceMoE",
    "BAMUMoEBlock",
    "MoEDeltaKVPagedAttention",
    "MoEAttention",
]
