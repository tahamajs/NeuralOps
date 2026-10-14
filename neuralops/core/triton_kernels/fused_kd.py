"""
Fused Knowledge Distillation (KL Divergence) Loss Kernel and Module.
Computes temperature-scaled KL divergence between student and teacher logits with full Autograd backward support.
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
def _fused_kd_fwd_kernel(
    student_ptr,
    teacher_ptr,
    loss_ptr,
    stride_s_row,
    stride_s_col,
    stride_t_row,
    stride_t_col,
    stride_loss,
    V: tl.constexpr,
    INV_T: tl.constexpr,      # 1.0 / Temperature
    T2: tl.constexpr,         # Temperature^2
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused KL Divergence loss per token row:
      Loss_row = T^2 * sum_v [ p_t(v) * (log p_t(v) - log p_s(v)) ]
    where:
      p_s = softmax(logits_s * INV_T)
      p_t = softmax(logits_t * INV_T)
    """
    row_idx = tl.program_id(0)

    # Pass 1: Find row-max for student and teacher
    max_s = -float("inf")
    max_t = -float("inf")

    for col_start in range(0, V, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < V

        s_val = tl.load(
            student_ptr + row_idx * stride_s_row + col_offsets * stride_s_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T
        t_val = tl.load(
            teacher_ptr + row_idx * stride_t_row + col_offsets * stride_t_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T

        max_s = tl.maximum(max_s, tl.max(s_val, axis=0))
        max_t = tl.maximum(max_t, tl.max(t_val, axis=0))

    # Pass 2: Accumulate sum(exp(x - max))
    sum_exp_s = 0.0
    sum_exp_t = 0.0

    for col_start in range(0, V, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < V

        s_val = tl.load(
            student_ptr + row_idx * stride_s_row + col_offsets * stride_s_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T
        t_val = tl.load(
            teacher_ptr + row_idx * stride_t_row + col_offsets * stride_t_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T

        sum_exp_s += tl.sum(tl.where(mask, tl.exp(s_val - max_s), 0.0), axis=0)
        sum_exp_t += tl.sum(tl.where(mask, tl.exp(t_val - max_t), 0.0), axis=0)

    lse_s = max_s + tl.log(sum_exp_s)
    lse_t = max_t + tl.log(sum_exp_t)

    # Pass 3: Accumulate KL divergence
    row_loss = 0.0
    for col_start in range(0, V, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < V

        s_val = tl.load(
            student_ptr + row_idx * stride_s_row + col_offsets * stride_s_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T
        t_val = tl.load(
            teacher_ptr + row_idx * stride_t_row + col_offsets * stride_t_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T

        log_p_s = s_val - lse_s
        log_p_t = t_val - lse_t
        p_t = tl.exp(log_p_t)

        kl_elem = p_t * (log_p_t - log_p_s)
        row_loss += tl.sum(tl.where(mask, kl_elem, 0.0), axis=0)

    tl.store(loss_ptr + row_idx * stride_loss, row_loss * T2)


@_jit_guard
def _fused_kd_bwd_kernel(
    student_ptr,
    teacher_ptr,
    grad_loss_ptr,
    grad_s_ptr,
    stride_s_row,
    stride_s_col,
    stride_t_row,
    stride_t_col,
    stride_gs_row,
    stride_gs_col,
    V: tl.constexpr,
    INV_T: tl.constexpr,      # 1.0 / Temperature
    T_VAL: tl.constexpr,      # Temperature
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused KL Divergence backward pass w.r.t. student_logits:
      d(Loss)/d(logits_s) = T * (p_s - p_t) * grad_loss
    """
    row_idx = tl.program_id(0)
    g_loss = tl.load(grad_loss_ptr + row_idx)

    # Pass 1: Row max
    max_s = -float("inf")
    max_t = -float("inf")
    for col_start in range(0, V, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < V
        s_val = tl.load(
            student_ptr + row_idx * stride_s_row + col_offsets * stride_s_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T
        t_val = tl.load(
            teacher_ptr + row_idx * stride_t_row + col_offsets * stride_t_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T
        max_s = tl.maximum(max_s, tl.max(s_val, axis=0))
        max_t = tl.maximum(max_t, tl.max(t_val, axis=0))

    # Pass 2: Sum exp
    sum_exp_s = 0.0
    sum_exp_t = 0.0
    for col_start in range(0, V, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < V
        s_val = tl.load(
            student_ptr + row_idx * stride_s_row + col_offsets * stride_s_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T
        t_val = tl.load(
            teacher_ptr + row_idx * stride_t_row + col_offsets * stride_t_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T
        sum_exp_s += tl.sum(tl.where(mask, tl.exp(s_val - max_s), 0.0), axis=0)
        sum_exp_t += tl.sum(tl.where(mask, tl.exp(t_val - max_t), 0.0), axis=0)

    lse_s = max_s + tl.log(sum_exp_s)
    lse_t = max_t + tl.log(sum_exp_t)

    # Pass 3: Gradients
    scale = g_loss * T_VAL
    for col_start in range(0, V, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < V
        s_val = tl.load(
            student_ptr + row_idx * stride_s_row + col_offsets * stride_s_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T
        t_val = tl.load(
            teacher_ptr + row_idx * stride_t_row + col_offsets * stride_t_col,
            mask=mask,
            other=-float("inf"),
        ) * INV_T

        p_s = tl.exp(s_val - lse_s)
        p_t = tl.exp(t_val - lse_t)
        grad_s = (p_s - p_t) * scale
        tl.store(
            grad_s_ptr + row_idx * stride_gs_row + col_offsets * stride_gs_col,
            grad_s,
            mask=mask,
        )


class _FusedKDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float, reduction: str):
        ctx.temperature = float(temperature)
        ctx.reduction = reduction
        ctx.orig_shape = student_logits.shape

        s_flat = student_logits.reshape(-1, student_logits.shape[-1]).contiguous()
        t_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1]).contiguous()
        N_tokens, V = s_flat.shape

        if TRITON_AVAILABLE and student_logits.is_cuda:
            loss_per_token = torch.empty((N_tokens,), device=s_flat.device, dtype=torch.float32)
            BLOCK_SIZE = min(2048, triton.next_power_of_2(min(V, 2048)))
            _fused_kd_fwd_kernel[(N_tokens,)](
                s_flat,
                t_flat,
                loss_per_token,
                s_flat.stride(0),
                s_flat.stride(1),
                t_flat.stride(0),
                t_flat.stride(1),
                loss_per_token.stride(0),
                V=V,
                INV_T=float(1.0 / temperature),
                T2=float(temperature * temperature),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4 if BLOCK_SIZE <= 1024 else 8,
                num_stages=3,
            )
        else:
            p_s = F.log_softmax(s_flat / temperature, dim=-1)
            p_t = F.softmax(t_flat / temperature, dim=-1)
            loss_per_token = (p_t * (p_t.log() - p_s)).sum(dim=-1) * (temperature * temperature)

        ctx.save_for_backward(s_flat, t_flat)

        if reduction == "batchmean":
            batch_size = ctx.orig_shape[0] if ctx.orig_shape[0] > 0 else 1
            return loss_per_token.sum() / batch_size
        elif reduction == "mean":
            return loss_per_token.sum() / (N_tokens * V)
        elif reduction == "sum":
            return loss_per_token.sum()
        else:
            return loss_per_token.view(ctx.orig_shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        s_flat, t_flat = ctx.saved_tensors
        temperature = ctx.temperature
        reduction = ctx.reduction
        N_tokens, V = s_flat.shape

        if reduction == "batchmean":
            batch_size = ctx.orig_shape[0] if ctx.orig_shape[0] > 0 else 1
            scale = grad_output / batch_size
            grad_per_token = scale.repeat(N_tokens)
        elif reduction == "mean":
            scale = grad_output / (N_tokens * V)
            grad_per_token = scale.repeat(N_tokens)
        elif reduction == "sum":
            grad_per_token = grad_output.repeat(N_tokens)
        else:
            grad_per_token = grad_output.reshape(-1)

        grad_s = torch.empty_like(s_flat)
        if TRITON_AVAILABLE and s_flat.is_cuda:
            BLOCK_SIZE = min(2048, triton.next_power_of_2(min(V, 2048)))
            _fused_kd_bwd_kernel[(N_tokens,)](
                s_flat,
                t_flat,
                grad_per_token.contiguous(),
                grad_s,
                s_flat.stride(0),
                s_flat.stride(1),
                t_flat.stride(0),
                t_flat.stride(1),
                grad_s.stride(0),
                grad_s.stride(1),
                V=V,
                INV_T=float(1.0 / temperature),
                T_VAL=float(temperature),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4 if BLOCK_SIZE <= 1024 else 8,
                num_stages=3,
            )
        else:
            p_s = F.softmax(s_flat / temperature, dim=-1)
            p_t = F.softmax(t_flat / temperature, dim=-1)
            grad_s = (p_s - p_t) * (grad_per_token.unsqueeze(-1) * temperature)

        return grad_s.view(ctx.orig_shape), None, None, None


class FusedKDLoss(nn.Module):
    """
    Fused Knowledge Distillation Loss between student and teacher logits.
    Loss = T^2 * KL(Softmax(logits_s / T) || Softmax(logits_t / T))
    Fully supports autograd backward passes on GPU and CPU.
    """

    def __init__(self, temperature: float = 2.0, reduction: str = "batchmean"):
        super().__init__()
        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        return _FusedKDFunction.apply(student_logits, teacher_logits, self.temperature, self.reduction)


def fused_kd_loss_pytorch(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Pure PyTorch fallback reference for KD loss."""
    orig_shape = student_logits.shape
    if student_logits.dim() > 2:
        s_flat = student_logits.reshape(-1, orig_shape[-1])
        t_flat = teacher_logits.reshape(-1, orig_shape[-1])
    else:
        s_flat = student_logits
        t_flat = teacher_logits
    p_s = F.log_softmax(s_flat / temperature, dim=-1)
    p_t = F.softmax(t_flat / temperature, dim=-1)
    return F.kl_div(p_s, p_t, reduction=reduction) * (temperature * temperature)


def fused_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Fused KD loss with PyTorch/Triton dispatch and full autograd backward."""
    fn = FusedKDLoss(temperature=temperature, reduction=reduction)
    return fn(student_logits, teacher_logits)


__all__ = [
    "FusedKDLoss",
    "fused_kd_loss",
    "fused_kd_loss_pytorch",
]
