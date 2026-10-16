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
    B, H, N, D = q1.shape
    scale = 1.0 / (D ** 0.5)
    s1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
    s2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
    causal = torch.triu(torch.full((N, N), float('-inf'), device=q1.device), diagonal=1)
    a1 = F.softmax(s1 + causal, dim=-1)
    a2 = F.softmax(s2 + causal, dim=-1)
    diff = F.relu(a1 - lambda_val * a2)
    norm = diff.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.matmul(diff / norm, v)

def differential_attention(q1: torch.Tensor, k1: torch.Tensor, q2: torch.Tensor, k2: torch.Tensor, v: torch.Tensor, lambda_val: float = 0.8) -> torch.Tensor:
    def cuda_fn(q1, k1, q2, k2, v, lambda_val=0.8):
        mod = _get_module()
        return mod.diff_attn_fwd(q1.contiguous(), k1.contiguous(), q2.contiguous(), k2.contiguous(), v.contiguous(), float(lambda_val))
    return dispatch(cuda_fn, differential_attention_ref, differential_attention_ref, q1, k1=k1, q2=q2, k2=k2, v=v, lambda_val=lambda_val, kernel_name="differential_attention")
