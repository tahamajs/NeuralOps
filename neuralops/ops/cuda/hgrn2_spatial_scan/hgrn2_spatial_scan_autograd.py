#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HGRN2 Spatial Scan Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

logger = logging.getLogger(__name__)

_MODULE: Any = None
_MODULE_ERROR: Exception | None = None


def _get_module() -> Any:
    global _MODULE, _MODULE_ERROR
    if _MODULE_ERROR is not None:
        return None
    if _MODULE is None:
        try:
            curr_dir = Path(__file__).parent
            with open(curr_dir / "hgrn2_spatial_scan_binding.cpp", "r", encoding="utf-8") as f:
                cpp_src = f.read()
            with open(curr_dir / "hgrn2_spatial_scan.cu", "r", encoding="utf-8") as f:
                cuda_src = f.read()
            _MODULE = load_cuda_module(
                name="cuda_hgrn2_spatial_scan",
                cpp_sources=cpp_src,
                cuda_sources=cuda_src,
                functions=["hgrn2_spatial_scan_fwd"],
            )
        except Exception as exc:
            _MODULE_ERROR = exc
            logger.warning("Failed to load CUDA module 'cuda_hgrn2_spatial_scan': %s", exc)
            return None
    return _MODULE


def hgrn2_spatial_scan_pytorch_ref(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Pure PyTorch reference implementation of HGRN2 spatial scan recurrence.

    Computes the recurrent scan equation:
        s_t = sigmoid(g_t) * s_{t-1} + (1 - sigmoid(g_t)) * x_t
    optionally followed by output scaling with query projection q.

    Args:
        *args: Either 2 positional arguments (x, gate) or 4 positional arguments (q, k, v, gate).
        **kwargs: Optional keyword arguments for inputs (x, gate/g, q, k, v).

    Returns:
        torch.Tensor: Output tensor with the same shape and dtype as input x.

    Raises:
        ValueError: If input argument count is not 2 or 4.
    """
    if len(args) == 4:
        q, k, v, g = args
        x = k * v
        gate = g
        q_mul = q
    elif len(args) == 2:
        x, gate = args
        q_mul = None
    elif "x" in kwargs and "gate" in kwargs:
        x = kwargs["x"]
        gate = kwargs["gate"]
        q_mul = kwargs.get("q_mul", None)
    elif "x" in kwargs and "g" in kwargs:
        x = kwargs["x"]
        gate = kwargs["g"]
        q_mul = kwargs.get("q_mul", None)
    elif all(k in kwargs for k in ("q", "k", "v", "g")):
        q_mul = kwargs["q"]
        x = kwargs["k"] * kwargs["v"]
        gate = kwargs["g"]
    else:
        raise ValueError(f"hgrn2_spatial_scan expected 2 or 4 arguments, got {len(args)}")

    if gate.dim() == 3:
        gate = gate.unsqueeze(-1).expand_as(x)
    elif gate.shape != x.shape:
        gate = gate.expand_as(x)

    B, H, N, D = x.shape
    out = torch.empty_like(x)
    sig_g = torch.sigmoid(gate)
    s = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)
    for t in range(N):
        gt = sig_g[:, :, t, :]
        xt = x[:, :, t, :]
        s = gt * s + (1.0 - gt) * xt
        out[:, :, t, :] = s
    if q_mul is not None:
        out = q_mul * out
    return out


def _hgrn2_spatial_scan_dispatch(*args: Any, **kwargs: Any) -> torch.Tensor:
    def cuda_fn(*call_args: Any, **call_kwargs: Any) -> torch.Tensor:
        mod = _get_module()
        if mod is None:
            raise RuntimeError("CUDA module not loaded for 'hgrn2_spatial_scan'")
        fn = mod.hgrn2_spatial_scan_fwd
        if len(call_args) == 4:
            q, k, v, g = call_args
            x = (k * v).contiguous()
            gate = g
            if gate.dim() == 3:
                gate = gate.unsqueeze(-1).expand_as(x).contiguous()
            elif gate.shape != x.shape:
                gate = gate.expand_as(x).contiguous()
            res = fn(x, gate.contiguous())
            return q * res
        elif len(call_args) == 2:
            x, gate = call_args
            if gate.dim() == 3:
                gate = gate.unsqueeze(-1).expand_as(x).contiguous()
            elif gate.shape != x.shape:
                gate = gate.expand_as(x).contiguous()
            return fn(x.contiguous(), gate.contiguous())
        else:
            raise ValueError(f"hgrn2_spatial_scan expected 2 or 4 arguments, got {len(call_args)}")

    return dispatch(
        cuda_fn,
        hgrn2_spatial_scan_pytorch_ref,
        hgrn2_spatial_scan_pytorch_ref,
        *args,
        kernel_name="hgrn2_spatial_scan",
        **kwargs,
    )


class _Hgrn2_spatial_scanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any) -> torch.Tensor:
        out = _hgrn2_spatial_scan_dispatch(*args)
        ctx.save_for_backward(*[t for t in args if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in args]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in args]
        return out

    @staticmethod
    def backward(ctx: Any, *douts: Any) -> tuple[torch.Tensor | None, ...]:
        saved = list(ctx.saved_tensors)
        inputs: list[Any] = []
        for is_t, nt in zip(ctx.tensors_mask, ctx.non_tensors):
            if is_t:
                t = saved.pop(0)
                inputs.append(t.detach().requires_grad_(t.requires_grad))
            else:
                inputs.append(nt)
        with torch.enable_grad():
            out_p = hgrn2_spatial_scan_pytorch_ref(*inputs)
        diff_inputs = [inp for inp, is_t in zip(inputs, ctx.tensors_mask) if (is_t and inp.requires_grad)]
        dout_val = douts[0] if len(douts) == 1 else douts
        if not diff_inputs:
            return tuple([None] * len(ctx.tensors_mask))
        grads = torch.autograd.grad(out_p, diff_inputs, dout_val, allow_unused=True)
        grad_map = dict(zip(diff_inputs, grads))
        return tuple(
            grad_map.get(inp, None) if (is_t and inp.requires_grad) else None
            for inp, is_t in zip(inputs, ctx.tensors_mask)
        )


def hgrn2_spatial_scan(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Dispatches HGRN2 spatial scan with analytical/autograd gradient support.

    Computes hierarchical gated linear recurrence over sequence or spatial dimension.
    Supports either (x, gate) or (q, k, v, gate) input patterns.

    Args:
        *args: Either 2 positional tensors (x, gate) or 4 positional tensors (q, k, v, gate).
        **kwargs: Optional keyword arguments for backend override or input tensors.

    Returns:
        torch.Tensor: Recurrently scanned output tensor.
    """
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in args) or any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in kwargs.values()
    ):
        return _Hgrn2_spatial_scanFunction.apply(*args)
    return _hgrn2_spatial_scan_dispatch(*args, **kwargs)
