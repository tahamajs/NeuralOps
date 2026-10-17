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
        with open(curr_dir / "hgrn2_spatial_scan_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "hgrn2_spatial_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_hgrn2_spatial_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["hgrn2_spatial_scan_fwd"],
        )
    return _MODULE

def hgrn2_spatial_scan_pytorch_ref(*args, **kwargs) -> torch.Tensor:
    if len(args) == 4:
        q, k, v, g = args
        x = (k * v)
        gate = g
        q_mul = q
    elif len(args) == 2:
        x, gate = args
        q_mul = None
    else:
        raise ValueError(f"hgrn2_spatial_scan expected 2 or 4 arguments, got {len(args)}")

    if gate.dim() == 3:
        gate = gate.unsqueeze(-1).expand_as(x)
    
    B, H, N, D = x.shape
    out = torch.empty_like(x)
    sig_g = torch.sigmoid(gate)
    s = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)
    for t in range(N):
        gt = sig_g[:, :, t, :]
        xt = x[:, :, t, :]
        s = gt * s + (1.0 - gt) * xt
        out[:, :, t, :] = s
    if q_mul is not None:
        out = q_mul * out
    return out

def hgrn2_spatial_scan(*args, **kwargs) -> torch.Tensor:
    def cuda_fn(*call_args, **call_kwargs):
        mod = _get_module()
        fn = getattr(mod, "hgrn2_spatial_scan_fwd")
        if len(call_args) == 4:
            q, k, v, g = call_args
            x = (k * v).contiguous()
            gate = g
            if gate.dim() == 3:
                gate = gate.unsqueeze(-1).expand_as(x).contiguous()
            res = fn(x, gate.contiguous())
            return q * res
        elif len(call_args) == 2:
            x, gate = call_args
            if gate.dim() == 3:
                gate = gate.unsqueeze(-1).expand_as(x).contiguous()
            return fn(x.contiguous(), gate.contiguous())
        else:
            raise ValueError(f"hgrn2_spatial_scan expected 2 or 4 arguments, got {len(call_args)}")

    return dispatch(
        cuda_fn,
        hgrn2_spatial_scan_pytorch_ref,
        hgrn2_spatial_scan_pytorch_ref,
        *args,
        kernel_name="hgrn2_spatial_scan",
        **kwargs,
    )
