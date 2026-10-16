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
        with open(curr_dir / "goldfinch_ahn_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "goldfinch_ahn.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_goldfinch_ahn", cpp_src, cuda_src, ["goldfinch_ahn_fwd"])
    return _MODULE

def goldfinch_ahn_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    s = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
    for t in range(N):
        pred = k[:, :, t, :] * s
        err = v[:, :, t, :] - pred
        s = s + beta[:, :, t, :] * k[:, :, t, :] * err
        out[:, :, t, :] = q[:, :, t, :] * s
    return out

def goldfinch_ahn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    def cuda_fn(q, k, v, beta):
        mod = _get_module()
        return mod.goldfinch_ahn_fwd(q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous())
    return dispatch(cuda_fn, goldfinch_ahn_ref, goldfinch_ahn_ref, q, k=k, v=v, beta=beta, kernel_name="goldfinch_ahn")
