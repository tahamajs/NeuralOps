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
        with open(curr_dir / "paged_kv_evict_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "paged_kv_evict.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_paged_kv_evict",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["paged_kv_compact_fwd"],
        )
    return _MODULE

def paged_kv_compact_pytorch_ref(k_in, v_in, keep_indices):
    return [k_in[keep_indices], v_in[keep_indices]]
def paged_kv_compact_triton_ref(k_in, v_in, keep_indices):
    return paged_kv_compact_pytorch_ref(k_in, v_in, keep_indices)

paged_kv_evict_pytorch_ref = paged_kv_compact_pytorch_ref
paged_kv_evict_ref = paged_kv_compact_pytorch_ref

def paged_kv_evict(k_in: torch.Tensor, v_in: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "paged_kv_compact_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        paged_kv_compact_triton_ref,
        paged_kv_compact_pytorch_ref,
        k_in, v_in, keep_indices,
        kernel_name="paged_kv_evict",
    )
