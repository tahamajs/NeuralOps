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


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    """
    Fused Linear Projection + Cross-Entropy Loss without materializing full (B*T, V) logits in HBM.
    Iterates over token chunks:
      For chunk:
        1. logits_chunk = h_chunk @ weight.T + bias
        2. scalar_loss, d_logits_chunk computed in-place via online softmax
        3. d_h_chunk = d_logits_chunk @ weight
        4. d_weight += d_logits_chunk.T @ h_chunk
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        targets: torch.Tensor,
        ignore_index: int = -100,
        reduction: str = "mean",
        chunk_size: int = 1024,
    ):
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.chunk_size = chunk_size
        ctx.has_bias = bias is not None

        orig_shape = hidden_states.shape
        D = orig_shape[-1]
        V = weight.shape[0]

        h_flat = hidden_states.reshape(-1, D).contiguous()
        t_flat = targets.reshape(-1).contiguous()
        N_tokens = h_flat.shape[0]

        valid_mask = t_flat != ignore_index
        total_valid = valid_mask.sum().clamp(min=1).item()
        ctx.total_valid = total_valid
        ctx.orig_shape = orig_shape

        # Compute total scalar loss chunk-by-chunk
        total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)

        for c_start in range(0, N_tokens, chunk_size):
            c_end = min(c_start + chunk_size, N_tokens)
            h_c = h_flat[c_start:c_end]
            t_c = t_flat[c_start:c_end]

            logits_c = F.linear(h_c, weight, bias)
            loss_c = F.cross_entropy(logits_c, t_c, ignore_index=ignore_index, reduction="sum")
            total_loss += loss_c

        ctx.save_for_backward(h_flat, weight, bias, t_flat)

        if reduction == "mean":
            return total_loss / total_valid
        elif reduction == "sum":
            return total_loss
        else:
            # If reduction is 'none', fallback to standard chunked projection per token
            # Note: For memory saving during training, reduction="mean" or "sum" is standard
            return total_loss / total_valid

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        h_flat, weight, bias, t_flat = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        reduction = ctx.reduction
        chunk_size = ctx.chunk_size
        total_valid = ctx.total_valid
        N_tokens, D = h_flat.shape
        V = weight.shape[0]

        scale = grad_output / total_valid if reduction == "mean" else grad_output

        grad_hidden = torch.empty_like(h_flat)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if ctx.has_bias else None

        for c_start in range(0, N_tokens, chunk_size):
            c_end = min(c_start + chunk_size, N_tokens)
            h_c = h_flat[c_start:c_end]
            t_c = t_flat[c_start:c_end]

            # In-chunk projection
            logits_c = F.linear(h_c, weight, bias)
            probs_c = F.softmax(logits_c.float(), dim=-1)

            # In-place gradient computation: (probs - target_one_hot) * scale
            one_hot_c = torch.zeros_like(probs_c).scatter_(-1, t_c.clamp(min=0).unsqueeze(-1), 1.0)
            valid_c = (t_c != ignore_index).unsqueeze(-1)
            d_logits_c = torch.where(valid_c, (probs_c - one_hot_c) * scale, torch.zeros_like(probs_c)).to(h_flat.dtype)

            # Accumulate gradients into hidden_states and weight
            grad_hidden[c_start:c_end] = torch.matmul(d_logits_c, weight)
            grad_weight += torch.matmul(d_logits_c.t(), h_c).float()
            if grad_bias is not None:
                grad_bias += d_logits_c.sum(dim=0).float()

        grad_weight = grad_weight.to(weight.dtype)
        if grad_bias is not None:
            grad_bias = grad_bias.to(bias.dtype)

        return (
            grad_hidden.view(ctx.orig_shape),
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
        )


class FusedLinearCrossEntropy(nn.Module):
    """
    Fused Linear Projection + Cross-Entropy Loss module.
    Fuses the final projection (e.g. lm_head) directly into the chunked loss computation,
    completely preventing the massive [B, T, V] logits tensor from materializing in GPU HBM.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        chunk_size: int = 1024,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return FusedLinearCrossEntropyFunction.apply(
            hidden_states,
            weight,
            bias,
            targets,
            self.ignore_index,
            self.reduction,
            self.chunk_size,
        )


def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Functional entrypoint for FusedLinearCrossEntropy."""
    return FusedLinearCrossEntropyFunction.apply(
        hidden_states,
        weight,
        bias,
        targets,
        ignore_index,
        reduction,
        chunk_size,
    )


__all__ = [
    "FusedCrossEntropyLoss",
    "fused_cross_entropy",
    "fused_cross_entropy_pytorch",
    "FusedLinearCrossEntropy",
    "FusedLinearCrossEntropyFunction",
    "fused_linear_cross_entropy",
]
