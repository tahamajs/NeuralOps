"""
Fused Cross-Entropy Loss Interface, Autograd Functions, and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "fused_ce_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fused_ce.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fused_ce",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fused_ce_fwd", "fused_ce_bwd"],
        )
    return _MODULE


def fused_cross_entropy_pytorch_ref(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.fused_ce import fused_cross_entropy
        return fused_cross_entropy
    except ImportError:
        return None


class _FusedCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
        reduction: str,
    ) -> torch.Tensor:
        orig_shape = logits.shape
        logits_flat = logits.reshape(-1, logits.shape[-1]).contiguous()
        targets_flat = targets.reshape(-1).contiguous()
        N_tokens = logits_flat.shape[0]

        mod = _get_module()
        loss_per_token = mod.fused_ce_fwd(logits_flat, targets_flat, ignore_index)

        valid_mask = targets_flat != ignore_index
        total_valid = valid_mask.sum().clamp(min=1)

        ctx.save_for_backward(logits_flat, targets_flat, total_valid)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.orig_shape = orig_shape

        if reduction == "mean":
            return loss_per_token.sum() / total_valid
        elif reduction == "sum":
            return loss_per_token.sum()
        else:
            return loss_per_token.view(orig_shape[:-1])

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        logits_flat, targets_flat, total_valid = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        reduction = ctx.reduction

        if reduction == "mean":
            grad_loss = (grad_output / total_valid).to(torch.float32)
        elif reduction == "sum":
            grad_loss = grad_output.to(torch.float32)
        else:
            grad_loss = grad_output.reshape(-1).to(torch.float32).contiguous()

        mod = _get_module()
        grad_logits = mod.fused_ce_bwd(logits_flat, targets_flat, grad_loss, ignore_index)
        return grad_logits.view(ctx.orig_shape), None, None, None


def fused_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """3-Level Dispatched Fused Cross-Entropy Loss."""
    def cuda_fn(logits, targets, ignore_index=-100, reduction="mean"):
        if (not logits.requires_grad) or (not torch.is_grad_enabled()):
            orig_shape = logits.shape
            logits_flat = logits.reshape(-1, logits.shape[-1]).contiguous()
            targets_flat = targets.reshape(-1).contiguous()
            mod = _get_module()
            loss = mod.fused_ce_fwd(logits_flat, targets_flat, ignore_index)
            if reduction == "mean":
                if ignore_index < 0:
                    return loss.mean()
                valid = (targets_flat != ignore_index).sum().clamp(min=1)
                return loss.sum() / valid
            elif reduction == "sum":
                return loss.sum()
            else:
                return loss.view(orig_shape[:-1])
        return _FusedCrossEntropyFunction.apply(logits, targets, ignore_index, reduction)

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fused_cross_entropy_pytorch_ref,
        logits,
        targets,
        ignore_index=ignore_index,
        reduction=reduction,
        kernel_name="fused_cross_entropy",
    )
