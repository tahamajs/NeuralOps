from __future__ import annotations
from pathlib import Path
import torch
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "rwkv6_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "rwkv6.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_rwkv6", cpp_src, cuda_src, ["rwkv6_time_mix_fwd"])
    return _MODULE

def rwkv6_ref(r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    B, H, N, D = r.shape
    out = torch.empty_like(r)
    s = torch.zeros(B, H, D, device=r.device, dtype=r.dtype)
    r_gate = torch.sigmoid(r)
    k_exp = torch.exp(k)
    decay = torch.exp(-torch.exp(w))
    u_exp = u.view(1, H, 1, D)
    for t in range(N):
        kv = k_exp[:, :, t, :] * v[:, :, t, :]
        y = r_gate[:, :, t, :] * (u_exp.squeeze(2) * kv + s)
        s = decay[:, :, t, :] * s + kv
        out[:, :, t, :] = y
    return out

def rwkv6(r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    def cuda_fn(r, k, v, w, u):
        mod = _get_module()
        return mod.rwkv6_time_mix_fwd(r.contiguous(), k.contiguous(), v.contiguous(), w.contiguous(), u.contiguous())
    return dispatch(cuda_fn, rwkv6_ref, rwkv6_ref, r, k=k, v=v, w=w, u=u, kernel_name="rwkv6")
