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
        with open(curr_dir / "gla_unnormalized_scan_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "gla_unnormalized_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_gla_unnormalized_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["gla_unnormalized_scan_fwd"],
        )
    return _MODULE

def gla_unnormalized_scan_pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    B, H, N, D = q.shape
    g = decay.unsqueeze(-1).expand(B, H, N, D) if decay.dim() == 3 else decay
    s = torch.zeros(B, H, D, device=q.device, dtype=torch.float32)
    ys = []
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    for t in range(N):
        gate = torch.exp(g_f[:, :, t, :])
        k_val = k_f[:, :, t, :]
        v_val = v_f[:, :, t, :]
        q_val = q_f[:, :, t, :]
        s = gate * s + k_val * v_val
        ys.append((q_val * s).to(q.dtype))
    return torch.stack(ys, dim=2)

def gla_unnormalized_scan_triton_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    return gla_unnormalized_scan_pytorch_ref(q, k, v, decay)

def gla_unnormalized_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "gla_unnormalized_scan_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        gla_unnormalized_scan_triton_ref,
        gla_unnormalized_scan_pytorch_ref,
        q, k, v, decay,
        kernel_name="gla_unnormalized_scan",
    )
