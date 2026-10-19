# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Differential Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

logger = logging.getLogger(__name__)

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "differential_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "differential_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_differential_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["diff_attn_fwd", "diff_attn_bwd"]
        )
    return _MODULE

def differential_attention_ref(q1: torch.Tensor, k1: torch.Tensor, q2: torch.Tensor, k2: torch.Tensor, v: torch.Tensor, lambda_val: float = 0.8) -> torch.Tensor:
    out1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True)
    out2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True)
    return (out1.float() - float(lambda_val) * out2.float()).to(q1.dtype)

class DifferentialAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q1, k1, q2, k2, v, lambda_val: float):
        ctx.save_for_backward(q1, k1, q2, k2, v)
        ctx.lambda_val = float(lambda_val)
        mod = _get_module()
        return mod.diff_attn_fwd(
            q1.contiguous(),
            k1.contiguous(),
            q2.contiguous(),
            k2.contiguous(),
            v.contiguous(),
            float(lambda_val)
        )

    @staticmethod
    def backward(ctx, grad_out):
        q1, k1, q2, k2, v = ctx.saved_tensors
        lambda_val = ctx.lambda_val
        mod = _get_module()
        grad_q1, grad_k1, grad_q2, grad_k2, grad_v = mod.diff_attn_bwd(
            grad_out.contiguous(),
            q1.contiguous(),
            k1.contiguous(),
            q2.contiguous(),
            k2.contiguous(),
            v.contiguous(),
            float(lambda_val)
        )
        return grad_q1, grad_k1, grad_q2, grad_k2, grad_v, None

def differential_attention(q1: torch.Tensor, k1: torch.Tensor, q2: torch.Tensor, k2: torch.Tensor, v: torch.Tensor, lambda_val: float = 0.8) -> torch.Tensor:
    def cuda_fn(q1, k1, q2, k2, v, lambda_val=0.8, **kwargs):
        return DifferentialAttentionFunction.apply(q1, k1, q2, k2, v, float(lambda_val))
    if q1.shape[-1] > 128:
        cuda_fn = None
    try:
        from neuralops.core.triton_kernels.differential_attention import triton_differential_attention
        triton_fn = triton_differential_attention
    except (RuntimeError, MemoryError, AssertionError, ValueError, ImportError, AttributeError) as exc:
        logger.warning("Failed to initialize Triton differential_attention, falling back", exc_info=True)
        triton_fn = differential_attention_ref
    return dispatch(cuda_fn, triton_fn, differential_attention_ref, q1, k1=k1, q2=q2, k2=k2, v=v, lambda_val=lambda_val, kernel_name="differential_attention")
