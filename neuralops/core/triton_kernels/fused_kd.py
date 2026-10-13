"""
Fused Knowledge Distillation (KL Divergence) Loss Kernel and Module.
Computes temperature-scaled KL divergence between student and teacher logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedKDLoss(nn.Module):
    """
    Fused Knowledge Distillation Loss between student and teacher logits.
    Loss = T^2 * KL(Softmax(logits_s / T) || Softmax(logits_t / T))
    """

    def __init__(self, temperature: float = 2.0, reduction: str = "batchmean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits / T, dim=-1)
        loss = F.kl_div(p_s, p_t, reduction=self.reduction) * (T * T)
        return loss
