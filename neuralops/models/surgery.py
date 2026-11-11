#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Models Surgery Engine.

Exposes fluent and functional model surgery orchestrators bridging neuralops layers,
foundation model architectures, and checkpoint lifecycles.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuralops.api import replace_attention, transplant_attention
from neuralops.core._attention_surgery import calibrate_attention_subspace, heal_replaced_model
from neuralops.models.protocols import SurgicalModel
from neuralops.models.surgical_io import (
    extract_surgical_manifest,
    load_surgical_checkpoint,
    save_surgical_checkpoint,
)
from neuralops.surgery.fx_surgery import compile_linear_surgery
from neuralops.surgery.model_surgery import ModelSurgery, SurgicalModelWrapper
from neuralops.surgery.universal_calibrator import (
    UniversalActivationCalibrator,
    compress_model_universal,
    transplant_and_compress,
)

logger = logging.getLogger("NeuralOps.Models.Surgery")

__all__ = [
    "ModelSurgery",
    "SurgicalModelWrapper",
    "SurgicalModel",
    "replace_attention",
    "transplant_attention",
    "compile_linear_surgery",
    "calibrate_attention_subspace",
    "heal_replaced_model",
    "extract_surgical_manifest",
    "load_surgical_checkpoint",
    "save_surgical_checkpoint",
    "UniversalActivationCalibrator",
    "compress_model_universal",
    "transplant_and_compress",
    "transplant_model_attention",
]


def transplant_model_attention(
    model: nn.Module,
    attention: str = "cosformer",
    rank: Optional[int] = None,
    copy_weights: bool = True,
    layers: Optional[List[int]] = None,
    **kwargs: Any,
) -> nn.Module:
    """Convenience helper performing verified attention transplantation on a model.

    Ensures that factory resolution and parameter binding match target layer contracts.
    """
    return replace_attention(
        model=model,
        attention=attention,
        rank=rank,
        copy_weights=copy_weights,
        layers=layers,
        **kwargs,
    )
