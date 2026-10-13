"""
Layer-wise Distillation Loss for Attention Mechanism Transfer.

Computes auxiliary loss between teacher attention outputs and student attention outputs
to guide representation alignment during knowledge distillation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LayerAttentionDistillLoss(nn.Module):
    """
    Computes Mean Squared Error (MSE) and optional cosine similarity loss
    between intermediate teacher attention outputs and student attention outputs.
    """

    def __init__(
        self,
        weight: float = 1.0,
        loss_type: str = "mse",
        normalize: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.loss_type = loss_type.lower()
        self.normalize = normalize

    def forward(
        self,
        student_attn_outputs: List[torch.Tensor] | Tuple[torch.Tensor, ...],
        teacher_attn_outputs: List[torch.Tensor] | Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Args:
            student_attn_outputs: List of [B, N, D] tensors from student attention layers.
            teacher_attn_outputs: List of [B, N, D] tensors from teacher attention layers.

        Returns:
            Scalar loss tensor.
        """
        if not student_attn_outputs or not teacher_attn_outputs:
            return torch.tensor(0.0, requires_grad=True)

        total_loss = torch.tensor(0.0, device=student_attn_outputs[0].device)
        num_layers = min(len(student_attn_outputs), len(teacher_attn_outputs))

        for l in range(num_layers):
            s_out = student_attn_outputs[l]
            t_out = teacher_attn_outputs[l]

            if self.normalize:
                s_out = F.normalize(s_out, p=2, dim=-1)
                t_out = F.normalize(t_out, p=2, dim=-1)

            if self.loss_type == "mse":
                layer_loss = F.mse_loss(s_out, t_out)
            elif self.loss_type == "smooth_l1":
                layer_loss = F.smooth_l1_loss(s_out, t_out)
            elif self.loss_type == "cosine":
                layer_loss = 1.0 - F.cosine_similarity(s_out, t_out, dim=-1).mean()
            else:
                layer_loss = F.mse_loss(s_out, t_out)

            total_loss = total_loss + layer_loss

        return self.weight * (total_loss / max(num_layers, 1))


class LayerDistillScheduler:
    """
    Schedules the auxiliary loss weight over training.
    Stage 1 (0 - warmup_steps): weight = max_weight
    Stage 2 (warmup_steps - end): weight = max_weight * (1 - progress) + final_weight * progress
    """

    def __init__(
        self,
        warmup_steps: int = 2000,
        total_steps: int = 10000,
        max_weight: float = 0.5,
        final_weight: float = 0.0,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_weight = max_weight
        self.final_weight = final_weight

    def get_weight(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_weight
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return self.max_weight * (1.0 - progress) + self.final_weight * progress

