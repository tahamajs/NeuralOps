"""
Fused Knowledge Distillation Loss Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "kd_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fused_kd.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fused_kd",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fused_kd_loss_fwd"],
        )
    return _MODULE


def fused_kd_loss_pytorch_ref(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Pure PyTorch reference implementation using F.kl_div."""
    import torch.nn.functional as F
    orig_shape = student_logits.shape
    s_flat = student_logits.reshape(-1, orig_shape[-1])
    t_flat = teacher_logits.reshape(-1, orig_shape[-1])
    p_s = F.log_softmax(s_flat / temperature, dim=-1)
    p_t = F.softmax(t_flat / temperature, dim=-1)

    if reduction == "batchmean":
        batch_size = orig_shape[0] if orig_shape[0] > 0 else 1
        return F.kl_div(p_s, p_t, reduction="sum") * (temperature * temperature) / batch_size
    elif reduction == "none":
        loss = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1) * (temperature * temperature)
        return loss.view(*orig_shape[:-1])
    elif reduction == "sum":
        return F.kl_div(p_s, p_t, reduction="sum") * (temperature * temperature)
    else:
        # "mean"
        return F.kl_div(p_s, p_t, reduction="mean") * (temperature * temperature)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.fused_kd import fused_kd_loss
        return fused_kd_loss
    except ImportError:
        return None


def fused_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """3-Level Dispatched Fused KD Loss."""
    def cuda_fn(s_logits, t_logits, temperature=2.0, reduction="batchmean", **kwargs):
        mod = _get_module()
        return mod.fused_kd_loss_fwd(
            s_logits.contiguous(),
            t_logits.contiguous(),
            float(temperature),
            str(reduction),
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fused_kd_loss_pytorch_ref,
        student_logits,
        teacher_logits,
        temperature=temperature,
        reduction=reduction,
        kernel_name="fused_kd_loss",
    )
