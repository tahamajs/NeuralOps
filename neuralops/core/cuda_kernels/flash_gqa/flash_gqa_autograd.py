"""
Flash GQA Python Interface with 3-Level Fallback (Module 4).
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
    B, H, N, D = q.shape
    H_KV = k.shape[1]
    group_size = H // H_KV

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def flash_gqa(
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

    triton_fn = None
    try:
        from neuralops.core.triton_kernels.gqa import fused_weighted_gqa_online_softmax_triton
        # Map args if compatible
    except Exception:
        pass

    return dispatch(
        cuda_fn,
        None,
        flash_gqa_pytorch_ref,
        q,
        k,
        v,
        is_causal,
        scale,
        kernel_name="flash_gqa",
    )
