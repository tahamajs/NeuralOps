"""
Fused Linear Cross Entropy.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Literal


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    chunk_size: int = 4096,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Compute cross entropy loss without materializing full logits tensor.
    """
    orig_shape = hidden.shape
    M = hidden.numel() // hidden.shape[-1]
    D = hidden.shape[-1]
    hidden_flat = hidden.view(M, D)
    labels_flat = labels.view(-1)
    V = weight.shape[0]

    assert weight.shape[1] == D, f"weight D mismatch: {weight.shape[1]} vs {D}"

    valid = labels_flat != ignore_index
    n_valid = int(valid.sum().item())

    target_logits = torch.zeros(M, device=hidden.device, dtype=torch.float32)
    if n_valid > 0:
        valid_labels = labels_flat[valid]
        target_weight = weight[valid_labels]
        target_logits[valid] = (hidden_flat[valid].float() * target_weight.float()).sum(dim=-1)

    max_logit = torch.full((M,), -float("inf"), device=hidden.device, dtype=torch.float32)
    sum_exp = torch.zeros(M, device=hidden.device, dtype=torch.float32)

    h_f = hidden_flat.float()
    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        w_chunk = weight[v_start:v_end].float()
        logits_chunk = h_f @ w_chunk.T

        chunk_max = logits_chunk.max(dim=-1).values
        new_max = torch.maximum(max_logit, chunk_max)

        sum_exp = sum_exp * (max_logit - new_max).exp()
        sum_exp = sum_exp + (logits_chunk - new_max.unsqueeze(-1)).exp().sum(dim=-1)
        max_logit = new_max

    log_z = max_logit + sum_exp.log()

    if label_smoothing > 0.0:
        ce = log_z - target_logits
        uniform_ce = log_z - (h_f @ weight.float().mean(dim=0)).sum(dim=-1)
        loss = (1 - label_smoothing) * ce + label_smoothing * uniform_ce
    else:
        loss = log_z - target_logits

    if reduction == "mean":
        return loss[valid].mean() if n_valid > 0 else loss.mean()
    elif reduction == "sum":
        return loss[valid].sum()
    elif reduction == "none":
        return loss.view(*orig_shape[:-1])
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
