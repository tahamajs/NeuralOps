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
        with open(curr_dir / "differential_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "differential_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_differential_attention", cpp_src, cuda_src, ["diff_attn_fwd"])
    return _MODULE

def differential_attention_ref(q1: torch.Tensor, k1: torch.Tensor, q2: torch.Tensor, k2: torch.Tensor, v: torch.Tensor, lambda_val: float = 0.8) -> torch.Tensor:
    out1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True)
    out2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True)
    return (out1.float() - float(lambda_val) * out2.float()).to(q1.dtype)

def differential_attention(q1: torch.Tensor, k1: torch.Tensor, q2: torch.Tensor, k2: torch.Tensor, v: torch.Tensor, lambda_val: float = 0.8) -> torch.Tensor:
    def cuda_fn(q1, k1, q2, k2, v, lambda_val=0.8):
        mod = _get_module()
        return mod.diff_attn_fwd(q1.contiguous(), k1.contiguous(), q2.contiguous(), k2.contiguous(), v.contiguous(), float(lambda_val))
    if q1.shape[-1] > 128:
        cuda_fn = None
    return dispatch(cuda_fn, differential_attention_ref, differential_attention_ref, q1, k1=k1, q2=q2, k2=k2, v=v, lambda_val=lambda_val, kernel_name="differential_attention")
