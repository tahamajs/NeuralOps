#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Shadow-Twin Distillation Engine

Performs in-place shadow-twin distillation, homotopy scheduling, and surgery finalization.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.surgery.endoskeletal_shell import UniversalAttentionBridge

logger = logging.getLogger("NeuralOps.Surgery.ShadowTwin")


def compute_operator_cloning_loss(
    teacher_out: torch.Tensor,
    student_out: torch.Tensor,
) -> torch.Tensor:
    """Computes MSE + (1 - CosineSimilarity) output alignment loss."""
    mse_loss = F.mse_loss(student_out, teacher_out)
    cos_sim = F.cosine_similarity(student_out, teacher_out, dim=-1).mean()
    return mse_loss + 0.1 * (1.0 - cos_sim)


def compute_linear_cka_loss(
    teacher_act: torch.Tensor,
    student_act: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes Linear Centered Kernel Alignment (CKA) difference loss: 1.0 - CKA."""
    X = teacher_act.reshape(-1, teacher_act.shape[-1]).float()
    Y = student_act.reshape(-1, student_act.shape[-1]).float()

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    dot_xy = torch.norm(torch.matmul(X.t(), Y), p="fro") ** 2
    dot_xx = torch.norm(torch.matmul(X.t(), X), p="fro") ** 2
    dot_yy = torch.norm(torch.matmul(Y.t(), Y), p="fro") ** 2

    denom = torch.sqrt(dot_xx * dot_yy) + eps
    cka = dot_xy / denom
    return 1.0 - cka.clamp(min=0.0, max=1.0)


def enable_shadow_twin(
    bridge_or_model: nn.Module,
    stage: str = "stage2_block_cloning",
) -> None:
    """Enables shadow twin teacher branch inside UniversalAttentionBridge or all bridges in a model."""
    target_bridges = []
    if isinstance(bridge_or_model, UniversalAttentionBridge):
        target_bridges.append(bridge_or_model)
    else:
        for m in bridge_or_model.modules():
            if isinstance(m, UniversalAttentionBridge):
                target_bridges.append(m)

    for bridge in target_bridges:
        if bridge.teacher_branch is None:
            bridge.teacher_branch = bridge.orig_module
            bridge.teacher_branch.eval()
            for p in bridge.teacher_branch.parameters():
                p.requires_grad = False

        bridge.stage = stage
        bridge.alpha = 0.0
        patch_bridge_for_shadow_twin(bridge)


def set_surgery_alpha(model: nn.Module, alpha: float) -> None:
    """Sets homotopy blending factor alpha in [0.0, 1.0] across all bridges in model."""
    alpha = max(0.0, min(1.0, float(alpha)))
    for m in model.modules():
        if isinstance(m, UniversalAttentionBridge):
            m.alpha = alpha


def set_surgery_stage(model: nn.Module, stage: Union[str, int]) -> None:
    """
    Sets surgery training stage across all bridges:
      - 1 or 'stage1_orientation': freeze host, train linear feature maps / qkv projection.
      - 2 or 'stage2_block_cloning': local teacher-student block cloning with alpha=0.0.
      - 3 or 'stage3_end_to_end': full student path with alpha=1.0 or annealed.
    """
    if stage == 1:
        stage = "stage1_orientation"
    elif stage == 2:
        stage = "stage2_block_cloning"
    elif stage == 3:
        stage = "stage3_end_to_end"

    for m in model.modules():
        if isinstance(m, UniversalAttentionBridge):
            m.stage = stage
            if stage == "stage1_orientation":
                m.alpha = 0.0
            elif stage == "stage2_block_cloning":
                m.alpha = 0.0
            elif stage == "stage3_end_to_end":
                m.alpha = 1.0


def collect_surgery_distillation_loss(model: nn.Module) -> torch.Tensor:
    """Sums and averages distillation losses across all UniversalAttentionBridge modules."""
    losses = []
    device = None
    for m in model.modules():
        if isinstance(m, UniversalAttentionBridge) and hasattr(m, "_last_twin_loss") and m._last_twin_loss is not None:
            losses.append(m._last_twin_loss)
            if device is None:
                device = m._last_twin_loss.device

    if len(losses) == 0:
        param = next(model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(losses).mean()


def finalize_surgery(model: nn.Module) -> nn.Module:
    """
    Purges teacher branches and temporary distillation hooks, sets alpha=1.0,
    and runs garbage collection to yield a clean standalone student model.
    """
    count = 0
    # The teacher branch is registered as a child module. Iterate over a stable
    # snapshot so removing those children cannot invalidate Module.modules().
    for m in list(model.modules()):
        if isinstance(m, UniversalAttentionBridge):
            if m.teacher_branch is not None:
                del m.teacher_branch
                m.teacher_branch = None
            m.alpha = 1.0
            if hasattr(m, "_last_twin_loss"):
                m._last_twin_loss = None
            if getattr(m, "_is_shadow_twin_patched", False):
                m.forward = getattr(m, "_orig_unpatched_forward", m.forward)
                m._is_shadow_twin_patched = False
            count += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Finalized surgery across {count} UniversalAttentionBridge modules.")
    return model


def patch_bridge_for_shadow_twin(bridge: UniversalAttentionBridge) -> None:
    """Patches bridge forward to support in-place twin distillation and homotopy blending."""
    if getattr(bridge, "_is_shadow_twin_patched", False):
        return

    orig_forward = bridge.forward

    def shadow_twin_forward(*args: Any, **kwargs: Any) -> Any:
        if bridge.teacher_branch is None or getattr(bridge, "stage", None) == "inference":
            return orig_forward(*args, **kwargs)

        with torch.no_grad():
            teacher_ret = bridge.teacher_branch(*args, **kwargs)
            teacher_core = teacher_ret[0] if isinstance(teacher_ret, tuple) else teacher_ret

        student_ret = orig_forward(*args, **kwargs)
        student_core = student_ret[0] if isinstance(student_ret, tuple) else student_ret

        stage = getattr(bridge, "stage", "stage2_block_cloning")
        if stage in ("stage1_orientation", "stage2_block_cloning") and torch.is_grad_enabled():
            loss_clone = compute_operator_cloning_loss(teacher_core, student_core)
            loss_cka = compute_linear_cka_loss(teacher_core, student_core)
            bridge._last_twin_loss = loss_clone + 0.05 * loss_cka
        else:
            # Do not expose a training loss from an earlier forward during
            # evaluation, inference, or the end-to-end student stage.
            bridge._last_twin_loss = None

        alpha = getattr(bridge, "alpha", 1.0)
        if alpha < 1.0:
            blended_core = (1.0 - alpha) * teacher_core.detach() + alpha * student_core
            if isinstance(student_ret, tuple):
                return (blended_core, *student_ret[1:])
            return blended_core

        return student_ret

    bridge.forward = shadow_twin_forward
    bridge._orig_unpatched_forward = orig_forward
    bridge._is_shadow_twin_patched = True


__all__ = [
    "compute_operator_cloning_loss",
    "compute_linear_cka_loss",
    "enable_shadow_twin",
    "set_surgery_alpha",
    "set_surgery_stage",
    "collect_surgery_distillation_loss",
    "finalize_surgery",
    "patch_bridge_for_shadow_twin",
]
