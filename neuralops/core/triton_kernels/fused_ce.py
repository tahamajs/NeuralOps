"""
Fused Cross Entropy Loss Kernel with Online Softmax and Full Autograd Backward.
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


def _jit_guard(fn):
    if triton is not None and hasattr(triton, "jit"):
        return triton.jit(fn)
    return fn


@_jit_guard
def _fused_cross_entropy_fwd_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    stride_logits_row,
    stride_logits_col,
    stride_targets,
    V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
):
    row_idx = tl.program_id(0)
    target_idx = tl.load(targets_ptr + row_idx * stride_targets)

    if target_idx == IGNORE_INDEX:
        tl.store(loss_ptr + row_idx, 0.0)
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < V

    logits_row = tl.load(
        logits_ptr + row_idx * stride_logits_row + col_offsets * stride_logits_col,
        mask=mask,
        other=-float("inf"),
    )

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


@_jit_guard
def _fused_cross_entropy_bwd_kernel(
    logits_ptr,
    targets_ptr,
    grad_loss_ptr,
    grad_logits_ptr,
    stride_logits_row,
    stride_logits_col,
    stride_targets,
    stride_gl_row,
    stride_gl_col,
    V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
):
    row_idx = tl.program_id(0)
    target_idx = tl.load(targets_ptr + row_idx * stride_targets)
    g_loss = tl.load(grad_loss_ptr + row_idx)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < V

    if target_idx == IGNORE_INDEX:
        tl.store(
            grad_logits_ptr + row_idx * stride_gl_row + col_offsets * stride_gl_col,
            0.0,
            mask=mask,
        )
        return

    logits_row = tl.load(
        logits_ptr + row_idx * stride_logits_row + col_offsets * stride_logits_col,
        mask=mask,
        other=-float("inf"),
    )

    m = tl.max(logits_row, axis=0)
    exp_logits = tl.exp(logits_row - m)
    sum_exp = tl.sum(exp_logits, axis=0)

    # Softmax probabilities
    probs = exp_logits / sum_exp

    # Gradient: (probs - indicator) * grad_loss
    is_target = col_offsets == target_idx
    target_val = tl.where(is_target, 1.0, 0.0)
    d_logits = (probs - target_val) * g_loss

    tl.store(
        grad_logits_ptr + row_idx * stride_gl_row + col_offsets * stride_gl_col,
        d_logits,
        mask=mask,
    )


class _FusedCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int, reduction: str):
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.orig_shape = logits.shape

        logits_flat = logits.reshape(-1, logits.shape[-1]).contiguous()
        targets_flat = targets.reshape(-1).contiguous()
        N_tokens, V = logits_flat.shape

        if TRITON_AVAILABLE and logits.is_cuda and V <= 4096:
            loss_per_token = torch.empty((N_tokens,), device=logits.device, dtype=torch.float32)
            BLOCK_SIZE = min(4096, triton.next_power_of_2(V))
            _fused_cross_entropy_fwd_kernel[(N_tokens,)](
                logits_flat,
                targets_flat,
                loss_per_token,
                logits_flat.stride(0),
                logits_flat.stride(1),
                targets_flat.stride(0),
                V=V,
                BLOCK_SIZE=BLOCK_SIZE,
                IGNORE_INDEX=ignore_index,
            )
        else:
            loss_per_token = F.cross_entropy(
                logits_flat, targets_flat, ignore_index=ignore_index, reduction="none"
            )

        valid_mask = targets_flat != ignore_index
        total_valid = valid_mask.sum().clamp(min=1).item()

        ctx.save_for_backward(logits_flat, targets_flat)
        ctx.total_valid = total_valid

        if reduction == "mean":
            return loss_per_token.sum() / total_valid
        elif reduction == "sum":
            return loss_per_token.sum()
        else:
            return loss_per_token.view(ctx.orig_shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits_flat, targets_flat = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        reduction = ctx.reduction
        total_valid = ctx.total_valid
        N_tokens, V = logits_flat.shape

        if reduction == "mean":
            scale = grad_output / total_valid
            grad_per_token = scale.repeat(N_tokens)
        elif reduction == "sum":
            grad_per_token = grad_output.repeat(N_tokens)
        else:
            grad_per_token = grad_output.reshape(-1)

        grad_logits = torch.empty_like(logits_flat)
        if TRITON_AVAILABLE and logits_flat.is_cuda and V <= 4096:
            BLOCK_SIZE = min(4096, triton.next_power_of_2(V))
            _fused_cross_entropy_bwd_kernel[(N_tokens,)](
                logits_flat,
                targets_flat,
                grad_per_token.contiguous(),
                grad_logits,
                logits_flat.stride(0),
                logits_flat.stride(1),
                targets_flat.stride(0),
                grad_logits.stride(0),
                grad_logits.stride(1),
                V=V,
                BLOCK_SIZE=BLOCK_SIZE,
                IGNORE_INDEX=ignore_index,
            )
        else:
            probs = F.softmax(logits_flat, dim=-1)
            one_hot = torch.zeros_like(probs).scatter_(-1, targets_flat.clamp(min=0).unsqueeze(-1), 1.0)
            valid = (targets_flat != ignore_index).unsqueeze(-1)
            grad_logits = torch.where(valid, (probs - one_hot) * grad_per_token.unsqueeze(-1), torch.zeros_like(probs))

        return grad_logits.view(ctx.orig_shape), None, None, None


class FusedCrossEntropyLoss(nn.Module):
    """
    Fused Cross-Entropy loss module.
    Automatically dispatches to Triton GPU kernel with full autograd backward when applicable.
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _FusedCrossEntropyFunction.apply(logits, targets, self.ignore_index, self.reduction)


def fused_cross_entropy_pytorch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pure PyTorch fallback reference for cross-entropy."""
    orig_shape = logits.shape
    if logits.dim() > 2:
        logits = logits.reshape(-1, orig_shape[-1])
        targets = targets.reshape(-1)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)


def fused_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Fused cross-entropy loss with automatic Triton GPU / PyTorch fallback and autograd backward."""
    fn = FusedCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    return fn(logits, targets)


__all__ = [
    "FusedCrossEntropyLoss",
    "fused_cross_entropy",
    "fused_cross_entropy_pytorch",
]
