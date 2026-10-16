"""
Flash GQA Python Interface, Autograd Functions, and 3-Level Fallback (Module 4).
Supports arbitrary N_q <= N_kv, FP16/BF16/FP32, and exact analytical backward pass.
"""

from __future__ import annotations

import math
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
        with open(curr_dir / "flash_gqa_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "flash_gqa.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_flash_gqa",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["flash_gqa_fwd"],
        )
    return _MODULE


def flash_gqa_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Pure PyTorch GQA reference supporting N_Q != N_KV and FP32 numerical stability."""
    B, H, N_Q, D = q.shape
    H_KV = k.shape[1]
    N_KV = k.shape[2]
    group_size = H // H_KV

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    if group_size > 1:
        k_exp = k.repeat_interleave(group_size, dim=1)
        v_exp = v.repeat_interleave(group_size, dim=1)
    else:
        k_exp, v_exp = k, v

    scores = torch.matmul(q.float(), k_exp.float().transpose(-1, -2)) * scale
    if is_causal:
        q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
        k_idx = torch.arange(N_KV, device=q.device).unsqueeze(0)
        causal_mask = k_idx <= (q_idx + (N_KV - N_Q))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, v_exp.float()).to(q.dtype)
    return out


def _flash_gqa_triton_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor | None:
    # Triton online-softmax kernel requires power-of-2 head dimensions (32, 64, 128)
    # and equal sequence lengths
    if q.shape[-1] not in (32, 64, 128) or q.shape[2] != k.shape[2]:
        return None

    try:
        from neuralops.core.triton_kernels.gqa import fused_weighted_gqa_online_softmax_triton
        return fused_weighted_gqa_online_softmax_triton(
            q=q,
            k=k,
            v=v,
            head_weights=None,
            is_causal=is_causal,
            scale=scale,
        )
    except Exception:
        return None


def _flash_gqa_forward_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    mod = _get_module()
    cuda_fn = mod.flash_gqa_fwd if mod is not None else None

    return dispatch(
        cuda_fn,
        _flash_gqa_triton_fallback,
        flash_gqa_pytorch_ref,
        q,
        k,
        v,
        is_causal,
        scale,
        kernel_name="flash_gqa",
    )


class _FlashGQAAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal: bool, scale: float):
        out = _flash_gqa_forward_dispatch(q, k, v, is_causal, scale)
        ctx.save_for_backward(q, k, v, out)
        ctx.is_causal = is_causal
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, out = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale

        B, H_Q, N_Q, D = q.shape
        H_KV = k.shape[1]
        N_KV = k.shape[2]
        group_size = H_Q // H_KV

        k_exp = k.repeat_interleave(group_size, dim=1) if group_size > 1 else k
        v_exp = v.repeat_interleave(group_size, dim=1) if group_size > 1 else v

        # Recompute attention probabilities
        scores = torch.matmul(q.float(), k_exp.float().transpose(-1, -2)) * scale
        if is_causal:
            q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
            k_idx = torch.arange(N_KV, device=q.device).unsqueeze(0)
            causal_mask = k_idx <= (q_idx + (N_KV - N_Q))
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        p = F.softmax(scores, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)

        grad_out_f = grad_output.float()
        grad_v_exp = torch.matmul(p.transpose(-1, -2), grad_out_f)

        grad_p = torch.matmul(grad_out_f, v_exp.float().transpose(-1, -2))
        sum_p_grad_p = (p * grad_p).sum(dim=-1, keepdim=True)
        grad_scores = p * (grad_p - sum_p_grad_p) * scale

        grad_q = torch.matmul(grad_scores, k_exp.float()).to(q.dtype)
        grad_k_exp = torch.matmul(grad_scores.transpose(-1, -2), q.float())

        if group_size > 1:
            grad_k = grad_k_exp.view(B, H_KV, group_size, N_KV, D).sum(dim=2).to(k.dtype)
            grad_v = grad_v_exp.view(B, H_KV, group_size, N_KV, D).sum(dim=2).to(v.dtype)
        else:
            grad_k = grad_k_exp.to(k.dtype)
            grad_v = grad_v_exp.to(v.dtype)

        return grad_q, grad_k, grad_v, None, None


def flash_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Flash Grouped Query Attention with autograd backward and 3-level fallback."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    if any(t.requires_grad for t in (q, k, v) if isinstance(t, torch.Tensor)):
        return _FlashGQAAutograd.apply(q, k, v, is_causal, scale)

    return _flash_gqa_forward_dispatch(q, k, v, is_causal, scale)
