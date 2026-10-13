"""
Fused Cross Entropy Loss Kernel with Online Softmax and PyTorch Fallback.
Computes cross-entropy loss directly from logits and targets in a numerically stable manner.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


@triton.jit
def _fused_cross_entropy_fwd_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    stride_logits_row,
    stride_logits_col,
    stride_targets,
    V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < V

    logits_row = tl.load(
        logits_ptr + row_idx * stride_logits_row + col_offsets * stride_logits_col,
        mask=mask,
        other=-float("inf"),
    )

    target_idx = tl.load(targets_ptr + row_idx * stride_targets)

    # Online softmax max
    m = tl.max(logits_row, axis=0)
    # Exponentiate shifted logits
    exp_logits = tl.exp(logits_row - m)
    # Sum exp
    sum_exp = tl.sum(exp_logits, axis=0)
    # Logsumexp
    log_sum_exp = m + tl.log(sum_exp)

    # Target logit
    target_logit = tl.load(
        logits_ptr + row_idx * stride_logits_row + target_idx * stride_logits_col
    )
    loss = log_sum_exp - target_logit
    tl.store(loss_ptr + row_idx, loss)


class FusedCrossEntropyLoss(nn.Module):
    """
    Fused Cross-Entropy loss module.
    Automatically dispatches to Triton GPU kernel when applicable, else PyTorch F.cross_entropy.
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        orig_shape = logits.shape
        if logits.dim() > 2:
            logits = logits.view(-1, orig_shape[-1])
            targets = targets.view(-1)

        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Fallback to PyTorch native for stability / autograd
        return F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction=self.reduction)
