"""
FP4 (E2M1) Quantized GEMM Interface, Analytical Autograd STE Backward, and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "fp4_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fp4_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fp4_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fp4_gemm_fwd", "fp4_gemm_bwd"],
        )
    return _MODULE


def fp4_gemm_pytorch_ref(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N, K_packed = w_packed.shape
    K = K_packed * 2

    lut = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], device=x.device, dtype=torch.float32)

    w_unpacked = torch.empty((N, K), dtype=torch.float32, device=x.device)
    w_unpacked[:, 0::2] = lut[(w_packed & 0x0F).long()]
    w_unpacked[:, 1::2] = lut[((w_packed >> 4) & 0x0F).long()]

    w_scaled = w_unpacked * scales.unsqueeze(-1)
    y = torch.matmul(x_2d, w_scaled.t().to(x.dtype))
    if bias is not None:
        y = y + bias
    return y.reshape(*orig_shape[:-1], N)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.fused_norm_linear import fp4_gemm_triton
        return fp4_gemm_triton
    except (ImportError, AttributeError):
        return None


class _FP4GEMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, w_packed, scales, bias=None):
        mod = _get_module()
        out = mod.fp4_gemm_fwd(
            x.contiguous(),
            w_packed.contiguous(),
            scales.contiguous(),
            bias.contiguous() if bias is not None else None,
        )
        ctx.save_for_backward(x, w_packed, scales, bias if bias is not None else torch.Tensor())
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx: Any, grad_out):
        x, w_packed, scales, bias = ctx.saved_tensors
        mod = _get_module()
        grads = mod.fp4_gemm_bwd(
            grad_out.contiguous(),
            x.contiguous(),
            w_packed.contiguous(),
            scales.contiguous(),
            bias.contiguous() if ctx.has_bias else None,
        )
        gx, gscales, gbias = grads[0], grads[1], (grads[2] if ctx.has_bias else None)
        return gx, None, gscales, gbias


def fp4_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched FP4 GEMM with analytical autograd STE backward."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, scales, bias) if isinstance(t, torch.Tensor)
    )
    if needs_grad and x.is_cuda and x.dim() == 2:
        return _FP4GEMMFunction.apply(x, w_packed, scales, bias)

    def cuda_fn(x, w_packed, scales, bias=None, **kwargs):
        mod = _get_module()
        return mod.fp4_gemm_fwd(
            x.contiguous(),
            w_packed.contiguous(),
            scales.contiguous(),
            bias.contiguous() if bias is not None else None,
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fp4_gemm_pytorch_ref,
        x,
        w_packed,
        scales,
        bias=bias,
        kernel_name="fp4_gemm",
    )
