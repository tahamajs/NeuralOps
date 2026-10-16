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
        with open(curr_dir / "logcumsumexp_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "logcumsumexp.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_logcumsumexp",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["logcumsumexp_fwd"],
        )
    return _MODULE

def logcumsumexp_pytorch_ref(x):
    return torch.logcumsumexp(x.float(), dim=-1).to(x.dtype)
def logcumsumexp_triton_ref(x):
    return logcumsumexp_pytorch_ref(x)

def logcumsumexp(x: torch.Tensor) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "logcumsumexp_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        logcumsumexp_triton_ref,
        logcumsumexp_pytorch_ref,
        x,
        kernel_name="logcumsumexp",
    )
