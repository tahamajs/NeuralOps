#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gated Linear Attention Scan Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

logger = logging.getLogger(__name__)

_MODULE = None
_MODULE_ERROR: Exception | None = None


def _get_module() -> Any:
    """Lazily compile and load native CUDA GLA scan module."""
    global _MODULE, _MODULE_ERROR
    if _MODULE_ERROR is not None:
        return None
    if _MODULE is None:
        try:
            curr_dir = Path(__file__).parent
            with open(curr_dir / "gla_binding.cpp", "r", encoding="utf-8") as f:
                cpp_src = f.read()
            with open(curr_dir / "gla_scan.cu", "r", encoding="utf-8") as f:
                cuda_src = f.read()
            _MODULE = load_cuda_module(
                name="cuda_gla_scan",
                cpp_sources=cpp_src,
                cuda_sources=cuda_src,
                functions=["gla_scan_fwd"],
            )
        except Exception as exc:
            _MODULE_ERROR = exc
            logger.warning("GLA CUDA extension unavailable; using lower-level fallback: %s", exc)
            return None
    return _MODULE


def gla_scan_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of Gated Linear Attention (GLA) scan.

    Args:
        q: Query tensor of shape (B, H, L, D).
        k: Key tensor of shape (B, H, L, D).
        v: Value tensor of shape (B, H, L, D).
        alpha: Optional decay / gating factor of shape (B, H, L). Defaults to None (all 1.0).

    Returns:
        Output tensor of shape (B, H, L, D) matching q.dtype.
    """
    B, H, L, D = q.shape
    if L == 0:
        return torch.empty((B, H, 0, D), device=q.device, dtype=q.dtype)

    if alpha is not None and alpha.dim() == 4 and alpha.shape[-1] == 1:
        alpha = alpha.squeeze(-1)

    scale = 1.0 / math.sqrt(D)
    S = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        q_t = q[:, :, t, :].float()
        k_t = k[:, :, t, :].float()
        v_t = v[:, :, t, :].float()
        a_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1).float() if alpha is not None else 1.0
        kv = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
        S = a_t * S + kv
        y_t = torch.matmul(q_t.unsqueeze(-2), S).squeeze(-2) * scale
        ys.append(y_t)
    return torch.stack(ys, dim=2).to(q.dtype)


def _get_triton_impl() -> Any:
    """Return Triton GLA scan implementation if available."""
    try:
        from neuralops.ops.triton.gla_scan import gla_scan_triton

        return gla_scan_triton
    except (ImportError, AttributeError):
        return None


def _gla_scan_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched GLA Scan across CUDA C++, Triton, and PyTorch fallback."""

    def cuda_fn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        mod = _get_module()
        if mod is None:
            raise RuntimeError(f"GLA CUDA module failed to load: {_MODULE_ERROR}")
        return mod.gla_scan_fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            alpha.contiguous() if alpha is not None else None,
        )

    triton_impl = _get_triton_impl()

    def triton_fn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        if triton_impl is None:
            raise RuntimeError("GLA Triton implementation is unavailable")
        return triton_impl(q, k, v, lambdas=alpha)

    cuda_target = cuda_fn if q.shape[-1] <= 128 else None

    return dispatch(
        cuda_target,
        triton_fn,
        gla_scan_pytorch_ref,
        q,
        k,
        v,
        alpha=alpha,
        kernel_name="gla_scan",
    )


class _Gla_scanFunction(torch.autograd.Function):
    """Autograd function for Gated Linear Attention scan."""

    @staticmethod
    def forward(
        ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q_det = q.detach()
        k_det = k.detach()
        v_det = v.detach()
        alpha_det = alpha.detach() if alpha is not None else None
        out = _gla_scan_dispatch(q_det, k_det, v_det, alpha_det)

        ctx.save_for_backward(*[t for t in (q, k, v, alpha) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (q, k, v, alpha)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (q, k, v, alpha)]
        return out

    @staticmethod
    def backward(ctx: Any, *douts: Any) -> tuple[Optional[torch.Tensor], ...]:
        saved = list(ctx.saved_tensors)
        inputs = []
        for is_t, nt in zip(ctx.tensors_mask, ctx.non_tensors):
            if is_t:
                t = saved.pop(0)
                inputs.append(t.detach().requires_grad_(t.requires_grad))
            else:
                inputs.append(nt)

        diff_inputs = [inp for inp in inputs if isinstance(inp, torch.Tensor) and inp.requires_grad]
        if not diff_inputs:
            return tuple(None for _ in inputs)

        with torch.enable_grad():
            out_p = gla_scan_pytorch_ref(*inputs)

        dout_val = douts[0] if len(douts) == 1 else douts
        grads = torch.autograd.grad(out_p, diff_inputs, dout_val, allow_unused=True)
        grad_map = dict(zip(diff_inputs, grads))
        return tuple(grad_map.get(inp, None) if isinstance(inp, torch.Tensor) else None for inp in inputs)


GlaScanFunction = _Gla_scanFunction


def gla_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Gated Linear Attention (GLA) recurrent scan.

    Dispatches across CUDA C++ hardware kernel, Triton, or pure PyTorch fallback.

    Args:
        q: Query tensor of shape (B, H, L, D).
        k: Key tensor of shape (B, H, L, D).
        v: Value tensor of shape (B, H, L, D).
        alpha: Optional decay / gating factor of shape (B, H, L). Defaults to None (all 1.0).

    Returns:
        Output tensor of shape (B, H, L, D) matching q.dtype.
    """
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (q, k, v, alpha)):
        return _Gla_scanFunction.apply(q, k, v, alpha)
    return _gla_scan_dispatch(q, k, v, alpha)
