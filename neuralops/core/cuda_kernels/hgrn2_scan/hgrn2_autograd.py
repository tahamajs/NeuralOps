"""
HGRN2 Selective Scan Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations
from typing import Any
"""
HGRN2 Bidirectional Scan Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""


from pathlib import Path

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "hgrn2_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "hgrn2_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_hgrn2_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["hgrn2_bidirectional_scan_fwd"],
        )
    return _MODULE


def hgrn2_fused_bwd_scan_pytorch_ref(
    x: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.hgrn2_scan import triton_hgrn2_fused_bwd_scan_pytorch_ref
    return triton_hgrn2_fused_bwd_scan_pytorch_ref(x, g)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.hgrn2_scan import triton_hgrn2_fused_bwd_scan
        return triton_hgrn2_fused_bwd_scan
    except ImportError:
        return None


def _hgrn2_fused_bwd_scan_dispatch(
    x: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched Bidirectional HGRN2 Scan."""
    def cuda_fn(x, g):
        mod = _get_module()
        return mod.hgrn2_bidirectional_scan_fwd(x.contiguous(), g.contiguous())

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        hgrn2_fused_bwd_scan_pytorch_ref,
        x,
        g,
        kernel_name="hgrn2_fused_bwd_scan",
    )


class _Hgrn2_fused_bwd_scanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, g):
        out = _hgrn2_fused_bwd_scan_dispatch(x, g)
        ctx.save_for_backward(*[t for t in (x, g) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (x, g)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (x, g)]
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        saved = list(ctx.saved_tensors)
        inputs = []
        for is_t, nt in zip(ctx.tensors_mask, ctx.non_tensors):
            if is_t:
                t = saved.pop(0)
                inputs.append(t.detach().requires_grad_(t.requires_grad))
            else:
                inputs.append(nt)
        with torch.enable_grad():
            out_p = hgrn2_fused_bwd_scan_pytorch_ref(*inputs)
        tensor_inputs = [inp for inp, is_t in zip(inputs, ctx.tensors_mask) if is_t]
        dout_val = douts[0] if len(douts) == 1 else douts
        grads = torch.autograd.grad(out_p, tensor_inputs, dout_val, allow_unused=True)
        grad_list: list[Any] = list(grads)
        res = []
        for is_t in ctx.tensors_mask:
            if is_t:
                res.append(grad_list.pop(0))
            else:
                res.append(None)
        return tuple(res)

def hgrn2_fused_bwd_scan(x, g):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (x, g)):
        return _Hgrn2_fused_bwd_scanFunction.apply(x, g)
    return _hgrn2_fused_bwd_scan_dispatch(x, g)
