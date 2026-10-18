"""
Knowledge Distillation (KD) loss functions, training step, and cache utilities

Implements kd functionality for NeuralOps.

Author: Taha Majlesi
"""

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

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        use_fused_linear: bool = False,
        chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.use_fused_linear = use_fused_linear
        self.chunk_size = chunk_size
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        student_hidden: torch.Tensor | None = None,
        student_weight: torch.Tensor | None = None,
        teacher_hidden: torch.Tensor | None = None,
        teacher_weight: torch.Tensor | None = None,
        student_bias: torch.Tensor | None = None,
        teacher_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute distillation loss with optional fused linear kernel execution."""
        from neuralops.core.triton_kernels.fused_ce import fused_linear_cross_entropy
        from neuralops.core.triton_kernels.fused_kd import fused_linear_kl_divergence

        # If hidden states and head weights are provided, execute fused linear KD & CE directly
        # preventing multi-gigabyte logits materialization in HBM!
        if (
            self.use_fused_linear
            and student_hidden is not None
            and student_weight is not None
            and teacher_hidden is not None
            and teacher_weight is not None
        ):
            distill_loss = fused_linear_kl_divergence(
                student_hidden=student_hidden,
                student_weight=student_weight,
                teacher_hidden=teacher_hidden,
                teacher_weight=teacher_weight,
                student_bias=student_bias,
                teacher_bias=teacher_bias,
                temperature=self.temperature,
                reduction="batchmean",
                chunk_size=self.chunk_size,
            )
            if labels is not None:
                task_loss = fused_linear_cross_entropy(
                    hidden_states=student_hidden,
                    weight=student_weight,
                    bias=student_bias,
                    targets=labels,
                    chunk_size=self.chunk_size,
                )
                return self.alpha * task_loss + (1.0 - self.alpha) * distill_loss
            return distill_loss

        # Standard logits fallback path
        if student_logits is None or teacher_logits is None:
            raise ValueError(
                "Either (student_logits, teacher_logits) or (student_hidden, student_weight, teacher_hidden, teacher_weight) must be provided."
            )

        # Numerical stabilization for SSM/Recurrent activations in FP16/AMP
        student_logits_stable = torch.nan_to_num(student_logits.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        teacher_logits_stable = torch.nan_to_num(teacher_logits.float(), nan=0.0, posinf=50.0, neginf=-50.0)

        p_student = F.log_softmax(student_logits_stable / self.temperature, dim=-1)
        p_teacher = F.softmax(teacher_logits_stable / self.temperature, dim=-1)

        distill_loss = self.kl_div(p_student, p_teacher) * (self.temperature**2)

        if labels is not None:
            if labels.dtype in (torch.long, torch.int64):
                task_loss = F.cross_entropy(student_logits_stable, labels)
            else:
                task_loss = F.mse_loss(student_logits_stable.squeeze(-1), labels.float())
            total_loss = self.alpha * task_loss + (1.0 - self.alpha) * distill_loss
            return total_loss

        return distill_loss


def kd_train_step(
    student: nn.Module,
    teacher: nn.Module | None,
    batch: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer,
    kd_loss_fn: KDLoss,
    device: str | torch.device = "cuda",
    mixed_precision: str = "no",
    scaler: Any = None,
    teacher_logits: torch.Tensor | None = None,
) -> float:
    """
    Execute a single gradient update step using knowledge distillation from teacher to student.

    Args:
        student: Compressed student model undergoing fine-tuning.
        teacher: Full-precision frozen teacher model (can be None if teacher_logits is provided).
        batch: Input tensors (input_ids, attention_mask, labels, etc.).
        optimizer: Student parameter optimizer.
        kd_loss_fn: KDLoss instance.
        device: Active device.
        mixed_precision: "no", "fp16", or "bf16".
        scaler: GradScaler instance for fp16.
        teacher_logits: Optional pre-cached teacher logits tensor to skip teacher inference.

    Returns:
        Scalar training loss value.
    """
    student.train()
    if teacher is not None:
        teacher.eval()

    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device) if len(batch) > 2 else None

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    mp_type = (mixed_precision or "no").lower()
    amp_enabled = mp_type in ("fp16", "bf16") and use_cuda
    amp_dtype = torch.bfloat16 if mp_type == "bf16" else torch.float16

    optimizer.zero_grad()
    def _call_model(mod: nn.Module) -> Any:
        try:
            if kd_loss_fn.use_fused_linear:
                return mod(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            return mod(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            return mod(input_ids=input_ids, attention_mask=attention_mask)

    if amp_enabled:
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            if teacher_logits is not None:
                t_logits = teacher_logits.to(device)
                teacher_outputs = None
            else:
                with torch.no_grad():
                    teacher_outputs = _call_model(teacher)
                t_logits = teacher_outputs.logits
            student_outputs = _call_model(student)
            
            # Check if models support extracting final hidden states and lm_head weight for fused loss
            s_head = getattr(student, "lm_head", None)
            t_head = getattr(teacher, "lm_head", None)
            if (
                kd_loss_fn.use_fused_linear
                and s_head is not None
                and t_head is not None
                and hasattr(student_outputs, "hidden_states")
                and hasattr(teacher_outputs, "hidden_states")
            ):
                s_h = student_outputs.hidden_states[-1]
                t_h = teacher_outputs.hidden_states[-1]
                loss = kd_loss_fn(
                    labels=labels,
                    student_hidden=s_h,
                    student_weight=s_head.weight,
                    teacher_hidden=t_h,
                    teacher_weight=t_head.weight,
                    student_bias=getattr(s_head, "bias", None),
                    teacher_bias=getattr(t_head, "bias", None),
                )
            else:
                loss = kd_loss_fn(
                    student_logits=student_outputs.logits,
                    teacher_logits=t_logits,
                    labels=labels,
                )
    else:
        if teacher_logits is not None:
            t_logits = teacher_logits.to(device)
            teacher_outputs = None
        else:
            with torch.no_grad():
                teacher_outputs = _call_model(teacher)
            t_logits = teacher_outputs.logits
        student_outputs = _call_model(student)
        s_head = getattr(student, "lm_head", None)
        t_head = getattr(teacher, "lm_head", None)
        if (
            kd_loss_fn.use_fused_linear
            and s_head is not None
            and t_head is not None
            and teacher_outputs is not None
            and hasattr(student_outputs, "hidden_states")
            and hasattr(teacher_outputs, "hidden_states")
        ):
            s_h = student_outputs.hidden_states[-1]
            t_h = teacher_outputs.hidden_states[-1]
            loss = kd_loss_fn(
                labels=labels,
                student_hidden=s_h,
                student_weight=s_head.weight,
                teacher_hidden=t_h,
                teacher_weight=t_head.weight,
                student_bias=getattr(s_head, "bias", None),
                teacher_bias=getattr(t_head, "bias", None),
            )
        else:
            loss = kd_loss_fn(
                student_logits=student_outputs.logits,
                teacher_logits=t_logits,
                labels=labels,
            )

    if amp_enabled and mp_type == "fp16" and scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
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
