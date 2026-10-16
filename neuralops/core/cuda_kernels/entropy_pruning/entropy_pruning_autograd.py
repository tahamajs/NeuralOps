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
        with open(curr_dir / "entropy_pruning_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "entropy_pruning.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_entropy_pruning",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["entropy_row_reduction_fwd"],
        )
    return _MODULE

def entropy_pruning_pytorch_ref(scores):
    p = torch.softmax(scores.float(), dim=-1)
    return (-p * torch.log(p.clamp(min=1e-8))).sum(dim=-1).to(scores.dtype)
def entropy_pruning_triton_ref(scores):
    from neuralops.core.triton_kernels.elementwise import fused_entropy_token_pruning_mask_triton
    _, h = fused_entropy_token_pruning_mask_triton(scores)
    return h

def entropy_pruning(scores: torch.Tensor) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "entropy_row_reduction_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        entropy_pruning_triton_ref,
        entropy_pruning_pytorch_ref,
        scores,
        kernel_name="entropy_pruning",
    )
