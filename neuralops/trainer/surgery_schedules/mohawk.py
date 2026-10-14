"""
MOHAWK 3-Stage Surgery Protocol & Attention Homotopy Schedulers.
Stage 1: Matrix Orientation (Kernels & Gates only)
Stage 2: Parallel Block Hidden-State Alignment
Stage 3: End-to-End Weight Transfer & Knowledge Distillation
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer.distillation.multi_level_kd import (
    BlockwiseOperatorCloningLoss,
    HedgehogFeatureMapKDLoss,
    LinearCKALoss,
    LogitDistillationLoss,
    MiniLMValueRelationKDLoss,
)


class AttentionHomotopyScheduler:
    """Schedules homotopy parameter alpha: 0.0 (teacher) -> 1.0 (student)."""

    def __init__(self, total_steps: int, schedule: str = "cosine") -> None:
        self.total_steps = max(1, total_steps)
        self.schedule = schedule.lower()

    def get_alpha(self, step: int) -> float:
        progress = min(1.0, max(0.0, step / self.total_steps))
        if self.schedule == "linear":
            return progress
        elif self.schedule == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.schedule == "sigmoid":
            # Smooth S-curve
            x = (progress - 0.5) * 10.0
            return 1.0 / (1.0 + math.exp(-x))
        return progress


class AlphaBlendingAttentionWrapper(nn.Module):
    """Blends Teacher and Student attention outputs via alpha: (1-a)*Y_t + a*Y_s."""

    def __init__(self, teacher_attn: nn.Module, student_attn: nn.Module) -> None:
        super().__init__()
        self.teacher_attn = teacher_attn
        self.student_attn = student_attn
        self.alpha = 0.0

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(min(1.0, max(0.0, alpha)))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.alpha <= 0.0:
            return self.teacher_attn(*args, **kwargs)
        elif self.alpha >= 1.0:
            return self.student_attn(*args, **kwargs)

        out_t = self.teacher_attn(*args, **kwargs)
        out_s = self.student_attn(*args, **kwargs)

        val_t = out_t[0] if isinstance(out_t, tuple) else out_t
        val_s = out_s[0] if isinstance(out_s, tuple) else out_s

        blended = (1.0 - self.alpha) * val_t + self.alpha * val_s
        if isinstance(out_s, tuple):
            return (blended,) + out_s[1:]
        return blended


class MOHAWKTrainer:
    """Automated 3-Stage MOHAWK Subquadratic Conversion Curriculum."""

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        self.student = student_model
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.student.to(self.device)
        self.teacher.to(self.device)

        self.logit_loss_fn = LogitDistillationLoss(mode="decoupled_kd", temperature=2.0)
        self.block_loss_fn = BlockwiseOperatorCloningLoss()
        self.cka_loss_fn = LinearCKALoss()

    def step_stage1_matrix_orientation(self, batch: Any, optimizer: torch.optim.Optimizer) -> float:
        """Stage 1: Orient internal kernel projection and state frequencies."""
        optimizer.zero_grad()
        # Train student with frozen teacher supervision
        x = batch["input_ids"] if isinstance(batch, dict) else batch
        with torch.no_grad():
            t_out = self.teacher(x)
            t_logits = t_out.logits if hasattr(t_out, "logits") else t_out[0]

        s_out = self.student(x)
        s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]

        loss = self.logit_loss_fn(s_logits, t_logits)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def step_stage2_block_cloning(
        self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        """Stage 2: Parallel block cloning minimizing MSE + Cosine + CKA."""
        optimizer.zero_grad()
        loss = self.block_loss_fn(student_hidden, teacher_hidden) + 0.1 * self.cka_loss_fn(student_hidden, teacher_hidden)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def step_stage3_end_to_end(
        self, batch: Any, optimizer: torch.optim.Optimizer, alpha: float = 1.0
    ) -> float:
        """Stage 3: End-to-end full KD fine-tuning."""
        optimizer.zero_grad()
        x = batch["input_ids"] if isinstance(batch, dict) else batch
        labels = batch.get("labels", None) if isinstance(batch, dict) else None

        with torch.no_grad():
            t_out = self.teacher(x)
            t_logits = t_out.logits if hasattr(t_out, "logits") else t_out[0]

        s_out = self.student(x)
        s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]

        loss = self.logit_loss_fn(s_logits, t_logits, labels=labels)
        loss.backward()
        optimizer.step()
        return float(loss.item())


__all__ = [
    "AttentionHomotopyScheduler",
    "AlphaBlendingAttentionWrapper",
    "MOHAWKTrainer",
]
