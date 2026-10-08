"""Knowledge Distillation (KD) loss functions, training step, and cache utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger(__name__)


class KDLoss(nn.Module):
    """
    Combined Task and Distillation Loss for Model Compression Recovery.

    Computes:
        Loss = alpha * TaskLoss + (1 - alpha) * (T^2 * KL(student || teacher))
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute distillation loss with temperature scaling."""
        p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        p_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        distill_loss = self.kl_div(p_student, p_teacher) * (self.temperature**2)

        if labels is not None:
            if labels.dtype in (torch.long, torch.int64):
                task_loss = F.cross_entropy(student_logits, labels)
            else:
                task_loss = F.mse_loss(student_logits.squeeze(-1), labels)
            return self.alpha * task_loss + (1.0 - self.alpha) * distill_loss

        return distill_loss


def kd_train_step(
    student: nn.Module,
    teacher: nn.Module,
    batch: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer,
    kd_loss_fn: KDLoss,
    device: str | torch.device = "cuda",
) -> float:
    """
    Execute a single gradient update step using knowledge distillation from teacher to student.

    Args:
        student: Compressed student model undergoing fine-tuning.
        teacher: Full-precision frozen teacher model.
        batch: Input tensors (input_ids, attention_mask, labels, etc.).
        optimizer: Student parameter optimizer.
        kd_loss_fn: KDLoss instance.
        device: Active device.

    Returns:
        Scalar training loss value.
    """
    student.train()
    teacher.eval()

    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device) if len(batch) > 2 else None

    with torch.no_grad():
        teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
        teacher_logits = teacher_outputs.logits

    student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
    student_logits = student_outputs.logits

    loss = kd_loss_fn(student_logits, teacher_logits, labels=labels)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss.item())


def build_teacher_cache(
    teacher: nn.Module,
    dataloader: Any,
    cache_path: Path | str,
    device: str | torch.device = "cuda",
) -> Path:
    """Cache teacher output activations and logits to disk to accelerate distillation."""
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    teacher.eval()

    cached_logits = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            out = teacher(input_ids=input_ids, attention_mask=attention_mask)
            cached_logits.append(out.logits.cpu())

    torch.save(cached_logits, p)
    LOG.info("Saved teacher activation cache to %s (%d batches).", p, len(cached_logits))
    return p


__all__ = [
    "KDLoss",
    "kd_train_step",
    "build_teacher_cache",
]
