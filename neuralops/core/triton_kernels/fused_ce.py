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

    def _check_triton_available():
        if not torch.cuda.is_available():
            return False
        try:
            from neuralops.core.triton_kernels.hardware_profile import get_device_hardware_profile

            return get_device_hardware_profile(0).supports_triton_jit
        except Exception:
            return torch.cuda.get_device_capability()[0] >= 7

    TRITON_AVAILABLE = _check_triton_available()
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
    Follows the Liger-Kernel / FLA in-forward analytical gradient accumulation pattern:
      During forward (under no_grad for chunk logits):
        1. Pre-allocate grad_hidden and grad_weight
        2. In each token chunk:
           a. logits_c = h_c @ weight.T + bias
           b. Accumulate scalar cross-entropy loss
           c. Compute analytical softmax gradient in-place: d_logits_c = (probs - target_one_hot) / total_valid
           d. grad_hidden[start:end] = d_logits_c @ weight
           e. grad_weight.addmm_(d_logits_c.T, h_c)
           f. del logits_c, d_logits_c
        3. Save ONLY grad_hidden, grad_weight, grad_bias in ctx
      During backward:
        Simply return grad_hidden * grad_output, grad_weight * grad_output in O(1) time!
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
        chunk_size: int = 2048,
    ):
        ctx.reduction = reduction
        ctx.has_bias = bias is not None

        orig_shape = hidden_states.shape
        D = orig_shape[-1]
        V = weight.shape[0]

        h_flat = hidden_states.reshape(-1, D).contiguous()
        t_flat = targets.reshape(-1).contiguous()
        N_tokens = h_flat.shape[0]

        valid_mask = t_flat != ignore_index
        total_valid = valid_mask.sum().clamp(min=1).item()
        ctx.orig_shape = orig_shape

        scale_norm = 1.0 / total_valid if reduction == "mean" else 1.0

        compute_grads = (
            hidden_states.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
        )

        grad_hidden = torch.empty_like(h_flat) if compute_grads and hidden_states.requires_grad else None
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if compute_grads and weight.requires_grad else None
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if compute_grads and ctx.has_bias and bias.requires_grad else None

        total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        chunk_size = max(1, min(chunk_size, N_tokens))

        with torch.no_grad():
            for c_start in range(0, N_tokens, chunk_size):
                c_end = min(c_start + chunk_size, N_tokens)
                h_c = h_flat[c_start:c_end]
                t_c = t_flat[c_start:c_end]

                # 1. Chunked linear projection
                logits_c = F.linear(h_c, weight, bias)

                # 2. Scalar loss accumulation
                loss_c = F.cross_entropy(logits_c, t_c, ignore_index=ignore_index, reduction="sum")
                total_loss += loss_c

                # 3. Analytical in-place gradient accumulation
                if compute_grads:
                    probs_c = F.softmax(logits_c.float(), dim=-1)
                    one_hot_c = torch.zeros_like(probs_c).scatter_(-1, t_c.clamp(min=0).unsqueeze(-1), 1.0)
                    valid_c = (t_c != ignore_index).unsqueeze(-1)
                    d_logits_c = torch.where(
                        valid_c,
                        (probs_c - one_hot_c) * scale_norm,
                        torch.zeros_like(probs_c),
                    ).to(h_flat.dtype)

                    if grad_hidden is not None:
                        grad_hidden[c_start:c_end] = torch.matmul(d_logits_c, weight)
                    if grad_weight is not None:
                        grad_weight.addmm_(d_logits_c.t().float(), h_c.float())
                    if grad_bias is not None:
                        grad_bias += d_logits_c.sum(dim=0).float()

                    del logits_c, probs_c, one_hot_c, d_logits_c

        if grad_weight is not None:
            grad_weight = grad_weight.to(weight.dtype)
        if grad_bias is not None:
            grad_bias = grad_bias.to(bias.dtype)

        ctx.save_for_backward(grad_hidden, grad_weight, grad_bias)

        if reduction == "mean":
            out_loss = total_loss / total_valid
        elif reduction == "sum":
            out_loss = total_loss
        else:
            out_loss = total_loss / total_valid

        return out_loss.to(hidden_states.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_hidden, grad_weight, grad_bias = ctx.saved_tensors
        g_val = grad_output.to(torch.float32)

        out_grad_h = (grad_hidden.view(ctx.orig_shape) * g_val).to(grad_hidden.dtype) if grad_hidden is not None else None
        out_grad_w = (grad_weight * g_val).to(grad_weight.dtype) if grad_weight is not None else None
        out_grad_b = (grad_bias * g_val).to(grad_bias.dtype) if grad_bias is not None else None

        return (
            out_grad_h,
            out_grad_w,
            out_grad_b,
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
        if isinstance(targets, torch.Tensor) and targets.dtype in (torch.float16, torch.float32, torch.bfloat16) and bias is not None and bias.dtype in (torch.int32, torch.int64):
            # Arguments passed as (hidden_states, weight, bias, targets)
            actual_bias = targets
            actual_targets = bias
        else:
            actual_bias = bias
            actual_targets = targets

        return FusedLinearCrossEntropyFunction.apply(
            hidden_states,
            weight,
            actual_bias,
            actual_targets,
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
