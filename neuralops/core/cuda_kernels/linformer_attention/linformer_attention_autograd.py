from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "linformer_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "linformer_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_linformer_attention", cpp_src, cuda_src, ["linformer_attn_fwd"])
    return _MODULE

def linformer_attention_ref(q: torch.Tensor, k_proj: torch.Tensor, v_proj: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k_proj, v_proj)

def linformer_attention(q: torch.Tensor, k_proj: torch.Tensor, v_proj: torch.Tensor) -> torch.Tensor:
    def cuda_fn(q, k_proj, v_proj):
        mod = _get_module()
        return mod.linformer_attn_fwd(q.contiguous(), k_proj.contiguous(), v_proj.contiguous())
    return dispatch(cuda_fn, linformer_attention_ref, linformer_attention_ref, q, k_proj=k_proj, v_proj=v_proj, kernel_name="linformer_attention")
