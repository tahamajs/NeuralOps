# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Shadow twin surgery lifecycle: collect_surgery_distillation_loss, set_surgery_alpha, set_surgery_stage, finalize_surgery.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def collect_surgery_distillation_loss(model: nn.Module) -> torch.Tensor:
    """Collect shadow twin distillation loss across all transplanted attention layers."""
    from neuralops.surgery.shadow_twin import collect_surgery_distillation_loss as _collect

    return _collect(model)


def set_surgery_alpha(model: nn.Module, alpha: float) -> None:
    """Adjust distillation blending weight alpha in [0, 1] across all transplanted layers."""
    from neuralops.surgery.shadow_twin import set_surgery_alpha as _set_alpha

    _set_alpha(model, alpha)


def set_surgery_stage(model: nn.Module, stage: int) -> None:
    """Set multi-stage recovery training curriculum stage (1: Shadow, 2: Joint, 3: Finalized)."""
    from neuralops.surgery.shadow_twin import set_surgery_stage as _set_stage

    _set_stage(model, stage)


def finalize_surgery(model: nn.Module) -> nn.Module:
    """Sever teacher branches and permanently finalize student attention transplantation."""
    from neuralops.surgery.shadow_twin import finalize_surgery as _finalize

    return _finalize(model)
