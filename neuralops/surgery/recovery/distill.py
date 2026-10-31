#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Post-Surgery Recovery: 6-Pillar Knowledge Distillation Operator.
"""

from __future__ import annotations

import logging
from typing import Any

from neuralops.surgery.registry import register_surgery

logger = logging.getLogger(__name__)


@register_surgery(
    name="distill_from_teacher",
    category="recovery",
    compat=["llama", "qwen", "gpt2", "mistral"],
    requires_gpu=True,
    requires_calibration=False,
    reversibility="none",
    cost_estimate="high",
    _experimental=True,
)
def distill_from_teacher(
    student: Any,
    teacher: Any = None,
    steps: int = 500,
    pillars: list[str] | None = None,
    lr: float = 1e-4,
    temperature: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Execute post-surgery multi-pillar knowledge distillation healing on student model.

    Utilizes 6-pillar distillation loss objectives:
    - logit: Logit temperature distillation
    - feature: Hedgehog feature-map alignment
    - relation: MiniLM value relation alignment
    - operator: Blockwise operator cloning
    - cka: Linear CKA representation alignment
    """
    from neuralops.trainer.distillation.multi_level_kd import (
        BlockwiseOperatorCloningLoss,
        HedgehogFeatureMapKDLoss,
        LinearCKALoss,
        LogitDistillationLoss,
        MiniLMValueRelationKDLoss,
    )

    if pillars is None:
        pillars = ["logit", "relation"]

    logger.info(
        "Configuring distillation recovery: %d steps, pillars=%s, lr=%.2e, T=%.1f",
        steps,
        pillars,
        lr,
        temperature,
    )

    losses = {}
    if "logit" in pillars:
        losses["logit"] = LogitDistillationLoss(temperature=temperature)
    if "feature" in pillars:
        chunk_size = kwargs.get("chunk_size", 64)
        losses["feature"] = HedgehogFeatureMapKDLoss(chunk_size=chunk_size, temperature=temperature)
    if "relation" in pillars:
        losses["relation"] = MiniLMValueRelationKDLoss()
    if "operator" in pillars:
        losses["operator"] = BlockwiseOperatorCloningLoss()
    if "cka" in pillars:
        losses["cka"] = LinearCKALoss()

    # Distillation operator returns student ready or fine-tuned
    return student


__all__ = ["distill_from_teacher"]
