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
        with open(curr_dir / "rfa_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "rfa_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_rfa", cpp_src, cuda_src, ["rfa_fwd"])
    return _MODULE

def rfa_attention_ref(phi_q: torch.Tensor, phi_k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, H, N, D = phi_q.shape
    out = torch.empty_like(phi_q)
    s = torch.zeros(B, H, D, device=phi_q.device, dtype=torch.float32)
    denom = torch.zeros(B, H, D, device=phi_q.device, dtype=torch.float32)
    phi_q_f = phi_q.float()
    phi_k_f = phi_k.float()
    v_f = v.float()
    for t in range(N):
        s = s + phi_k_f[:, :, t, :] * v_f[:, :, t, :]
        denom = denom + phi_k_f[:, :, t, :]
        norm = torch.clamp(torch.abs(denom), min=1e-5)
        out[:, :, t, :] = ((phi_q_f[:, :, t, :] * s) / norm).to(phi_q.dtype)
    return out

def rfa_attention(phi_q: torch.Tensor, phi_k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    def cuda_fn(phi_q, phi_k, v):
        mod = _get_module()
        return mod.rfa_fwd(phi_q.contiguous(), phi_k.contiguous(), v.contiguous())
    return dispatch(cuda_fn, rfa_attention_ref, rfa_attention_ref, phi_q, phi_k, v, kernel_name="rfa_attention")
