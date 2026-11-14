#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Trainer Distillation Subpackage.
"""

from __future__ import annotations

from neuralops.trainer.distillation.multi_level_kd import (
    AttentionDistillationLoss,
    AttentionStabilityReg,
    BlockwiseOperatorCloningLoss,
    CosineAlignmentLoss,
    GramMatrixLoss,
    HedgehogFeatureMapKDLoss,
    HiddenMSELoss,
    LinearCKALoss,
    LogitDistillationLoss,
    LogitKLLoss,
    MiniLMValueRelationKDLoss,
    MultiLevelKDTrainer,
    SmoothMaxOverLayersCKASteeringLoss,
    SoftKLDivergenceLoss,
    SpikyFeatureMap,
    StreamingCKALoss,
    ValueRelationLoss,
    distill,
)

__all__ = [
    "AttentionDistillationLoss",
    "AttentionStabilityReg",
    "BlockwiseOperatorCloningLoss",
    "CosineAlignmentLoss",
    "GramMatrixLoss",
    "HedgehogFeatureMapKDLoss",
    "HiddenMSELoss",
    "LinearCKALoss",
    "LogitDistillationLoss",
    "LogitKLLoss",
    "MiniLMValueRelationKDLoss",
    "MultiLevelKDTrainer",
    "SmoothMaxOverLayersCKASteeringLoss",
    "SoftKLDivergenceLoss",
    "SpikyFeatureMap",
    "StreamingCKALoss",
    "ValueRelationLoss",
    "distill",
]
