"""
Fused Linear KL Divergence.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def fused_linear_kl_divergence(
    hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_weight: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 4096,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """
    KL(teacher || student) * T^2 without materializing student logits.
    """
    orig_shape = hidden.shape
    M = hidden.numel() // hidden.shape[-1]
    D = hidden.shape[-1]
    hidden_flat = hidden.view(M, D)
    teacher_flat = teacher_logits.view(M, -1)
    V = student_weight.shape[0]

    assert teacher_flat.shape[1] == V
    assert student_weight.shape[1] == D

    p_teacher = F.softmax(teacher_flat / temperature, dim=-1)
    log_p_teacher = torch.log(p_teacher + 1e-10)

    h_f = hidden_flat.float()
    w_f = student_weight.float()

    max_logit = torch.full((M,), -float("inf"), device=hidden.device, dtype=torch.float32)
    sum_exp = torch.zeros(M, device=hidden.device, dtype=torch.float32)

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        logits_chunk = h_f @ w_f[v_start:v_end].T
        chunk_max = logits_chunk.max(dim=-1).values
        new_max = torch.maximum(max_logit, chunk_max)
        sum_exp = sum_exp * (max_logit - new_max).exp()
        sum_exp = sum_exp + (logits_chunk - new_max.unsqueeze(-1)).exp().sum(dim=-1)
        max_logit = new_max

    log_z = max_logit + sum_exp.log()

    kl = torch.zeros(M, device=hidden.device, dtype=torch.float32)
    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        logits_chunk = h_f @ w_f[v_start:v_end].T
        kl = kl + (p_teacher[:, v_start:v_end] *
                   (log_p_teacher[:, v_start:v_end] - logits_chunk)).sum(dim=-1)

    kl = kl + log_z
    kl = kl * (temperature ** 2)

    if reduction == "batchmean" or reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "none":
        return kl.view(*orig_shape[:-1])
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
