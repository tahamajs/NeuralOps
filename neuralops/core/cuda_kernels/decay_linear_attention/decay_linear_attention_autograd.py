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
        with open(curr_dir / "decay_linear_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "decay_linear_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_decay_linear_attention", cpp_src, cuda_src, ["decay_linear_fwd"])
    return _MODULE

def decay_linear_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    s = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
    g = gamma.view(1, H, 1, D)
    for t in range(N):
        s = g.squeeze(2) * s + k[:, :, t, :] * v[:, :, t, :]
        out[:, :, t, :] = q[:, :, t, :] * s
    return out

def decay_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    def cuda_fn(q, k, v, gamma):
        mod = _get_module()
        return mod.decay_linear_fwd(q.contiguous(), k.contiguous(), v.contiguous(), gamma.contiguous())
    return dispatch(cuda_fn, decay_linear_attention_ref, decay_linear_attention_ref, q, k=k, v=v, gamma=gamma, kernel_name="decay_linear_attention")
