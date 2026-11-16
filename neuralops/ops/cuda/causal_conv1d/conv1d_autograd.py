#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Causal 1D Convolution Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients
and three-level fallback dispatch (CUDA C++ -> Triton JIT -> Vectorized PyTorch reference).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE: Any = None


def _get_module() -> Any:
    """Lazily load and compile native CUDA module for causal 1D convolution."""
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "conv1d_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "causal_conv1d.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_causal_conv1d",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["causal_conv1d_fwd"],
        )
    return _MODULE


def causal_conv1d_pytorch_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    silu_activation: bool = False,
) -> torch.Tensor:
    """Pure PyTorch reference implementation for causal 1D depthwise convolution.

    Args:
        x: Input tensor of shape (B, D, L).
        weight: Convolution kernel weight of shape (D, Kw) or (D, 1, Kw).
        bias: Optional bias tensor of shape (D,).
        silu_activation: Whether to apply SiLU (swish) activation after conv.

    Returns:
        Output tensor of shape (B, D, L).
    """
    B, D, L = x.shape
    if weight.dim() == 2:
        Kw = weight.shape[1]
        w = weight.unsqueeze(1)
    elif weight.dim() == 3:
        Kw = weight.shape[-1]
        w = weight
    else:
        raise ValueError(f"weight must be 2D or 3D, got shape {weight.shape}")
    x_pad = F.pad(x, (Kw - 1, 0))
    out: torch.Tensor = F.conv1d(x_pad, w, bias=bias, groups=D)
    if silu_activation:
        out = F.silu(out)
    return out


def _get_triton_impl() -> Optional[Callable[..., torch.Tensor]]:
    """Retrieve Triton implementation or sequence-level adapter if available."""
    try:
        from neuralops.ops.triton.state_update import causal_conv1d_step

        def causal_conv1d_triton(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            silu_activation: bool = False,
            **kwargs: Any,
        ) -> torch.Tensor:
            B, D, L = x.shape
            w = weight.squeeze(1) if weight.dim() == 3 else weight
            Kw = w.shape[-1]
            conv_state = torch.zeros(B, D, Kw - 1, dtype=x.dtype, device=x.device)
            outs = []
            for t in range(L):
                out_t, conv_state = causal_conv1d_step(x[:, :, t], conv_state, w, bias)
                if silu_activation:
                    out_t = F.silu(out_t)
                outs.append(out_t)
            return torch.stack(outs, dim=-1)

        return causal_conv1d_triton
    except (ImportError, AttributeError):
        return None


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    silu_activation: bool = False,
) -> torch.Tensor:
    """3-Level Dispatched Causal 1D Convolution.

    Routes execution through Level 1 (CUDA C++), Level 2 (Triton sequence step),
    or Level 3 (Vectorized PyTorch reference) depending on hardware availability
    and system fallback policies.

    Args:
        x: Input tensor of shape (B, D, L).
        weight: Convolution kernel weight of shape (D, Kw) or (D, 1, Kw).
        bias: Optional bias tensor of shape (D,).
        silu_activation: Whether to apply SiLU (swish) activation after conv.

    Returns:
        Output tensor of shape (B, D, L).
    """

    def cuda_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        silu_activation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        mod = _get_module()
        res: torch.Tensor = mod.causal_conv1d_fwd(
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            silu_activation,
        )
        return res

    triton_fn = _get_triton_impl()

    result: torch.Tensor = dispatch(
        cuda_fn,
        triton_fn,
        causal_conv1d_pytorch_ref,
        x,
        weight,
        bias=bias,
        silu_activation=silu_activation,
        kernel_name="causal_conv1d",
    )
    return result
