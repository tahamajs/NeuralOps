from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "split_head_norm_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "split_head_norm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_split_head_norm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["split_head_norm_fwd"],
        )
    return _MODULE

def split_head_norm_pytorch_ref(x, eps=1e-5):
    gf = x.float()
    rms = torch.rsqrt(gf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (gf * rms).to(x.dtype)
def split_head_norm_triton_ref(x, eps=1e-5):
    from neuralops.core.triton_kernels.elementwise import fused_split_head_hybrid_concat_norm_triton
    return fused_split_head_hybrid_concat_norm_triton([x], eps=eps)

def split_head_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "split_head_norm_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        split_head_norm_triton_ref,
        split_head_norm_pytorch_ref,
        x, eps=eps,
        kernel_name="split_head_norm",
    )
