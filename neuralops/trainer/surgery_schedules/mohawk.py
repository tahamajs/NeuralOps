"""
Mohawk Surgery Schedule

Curriculum distillation schedule for progressive attention layer replacement and parameter grafting.

Author: Taha Majlesi
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

    def __init__(
        self,
        total_steps: int,
        schedule: str = "cosine",
        warmup_steps: int = 0,
        schedule_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.total_steps = max(1, total_steps)
        if schedule_type is not None:
            schedule = schedule_type
        self.schedule = schedule.lower()
        self.warmup_steps = max(0, warmup_steps)

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


class HNCContinuationTrainer:
    """
    Hessian Null Space Continuation (HNC) Multi-Stage Distillation Trainer.
    Combines:
      - Closed-form Kronecker/Pearlmutter null-space projection for SVD layer alignment
      - SmoothMax over layerwise CKA steering
      - Logit distillation (DKD/forward KL)
      - Predictor-corrector continuation across training iterations
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 2.0,
        cka_weight: float = 0.5,
        hnc_eta: float = 0.01,
        hnc_mu: float = 1e-4,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.device = device
        self.student.to(self.device)
        self.teacher.to(self.device)

        from neuralops.trainer.distillation.multi_level_kd import (
            LogitDistillationLoss,
            SmoothMaxOverLayersCKASteeringLoss,
        )

        self.logit_loss_fn = LogitDistillationLoss(mode="decoupled_kd", temperature=temperature)
        self.cka_steering_fn = SmoothMaxOverLayersCKASteeringLoss(temperature=0.1)
        self.cka_weight = cka_weight
        self.hnc_eta = hnc_eta
        self.hnc_mu = hnc_mu

    def step_nullspace_spectral_continuation(
        self,
        probe_batch: Any,
        target_rank: int | None = None,
        num_steps: int = 5,
    ) -> dict[str, Any]:
        """Align all SVD linear attention layers along Hessian null space."""
        x = probe_batch["input_ids"] if isinstance(probe_batch, dict) else probe_batch
        x = x.to(self.device)
        results = {}
        for name, module in self.student.named_modules():
            if hasattr(module, "hnc_spectral_compress_and_align"):
                res = module.hnc_spectral_compress_and_align(probe_inputs=x, num_steps=num_steps, target_rank=target_rank)
                results[name] = res
        return results

    def step_distillation(
        self,
        batch: Any,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Execute one distillation step with SmoothMax CKA steering."""
        self.student.train()
        self.teacher.eval()
        optimizer.zero_grad()

        x = batch["input_ids"] if isinstance(batch, dict) else batch
        x = x.to(self.device)
        labels = batch.get("labels", None) if isinstance(batch, dict) else None
        if labels is not None:
            labels = labels.to(self.device)

        with torch.no_grad():
            t_out = self.teacher(x, output_hidden_states=True) if hasattr(self.teacher, "config") else self.teacher(x)
            t_logits = t_out.logits if hasattr(t_out, "logits") else t_out[0]
            t_hidden = getattr(t_out, "hidden_states", None)

        s_out = self.student(x, output_hidden_states=True) if hasattr(self.student, "config") else self.student(x)
        s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]
        s_hidden = getattr(s_out, "hidden_states", None)

        kd_loss = self.logit_loss_fn(s_logits, t_logits, labels=labels)

        total_loss = kd_loss
        cka_loss_val = 0.0
        if s_hidden is not None and t_hidden is not None:
            cka_loss = self.cka_steering_fn(s_hidden, t_hidden)
            cka_loss_val = float(cka_loss.item())
            total_loss = total_loss + self.cka_weight * cka_loss

        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "kd_loss": float(kd_loss.item()),
            "cka_loss": cka_loss_val,
        }


__all__ = [
    "AttentionHomotopyScheduler",
    "AlphaBlendingAttentionWrapper",
    "MOHAWKTrainer",
    "HNCContinuationTrainer",
]

