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
        with open(curr_dir / "entmax_bisect_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "entmax_bisect.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_entmax_bisect",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["entmax15_fwd"],
        )
    return _MODULE

from neuralops.core.triton_kernels.elementwise import fused_entmax_bisect_pytorch, fused_entmax_bisect_triton

def entmax_bisect(scores: torch.Tensor, n_iter: int = 15) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "entmax15_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        fused_entmax_bisect_triton,
        fused_entmax_bisect_pytorch,
        scores, n_iter=n_iter,
        kernel_name="entmax_bisect",
    )
