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
        with open(curr_dir / "fast_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "fast_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_fast_attn", cpp_src, cuda_src, ["fast_attn_fwd"])
    return _MODULE

def fast_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    phi_q = F.relu(q) + 1.0
    phi_k = F.relu(k) + 1.0
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    s = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
    z = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
    for t in range(N):
        s = s + phi_k[:, :, t, :] * v[:, :, t, :]
        z = z + phi_k[:, :, t, :]
        out[:, :, t, :] = (phi_q[:, :, t, :] * s) / z.clamp_min(eps)
    return out

def fast_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    def cuda_fn(q, k, v, eps=1e-6):
        mod = _get_module()
        return mod.fast_attn_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(eps))
    return dispatch(cuda_fn, fast_attention_ref, fast_attention_ref, q, k, v, eps=eps, kernel_name="fast_attention")
