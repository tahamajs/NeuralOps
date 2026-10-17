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
        with open(curr_dir / "gdn_delta_rule_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "gdn_delta_rule.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_gdn_delta_rule",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["gdn_delta_rule_fwd"],
        )
    return _MODULE

def gdn_delta_rule_pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference implementation for GDN Delta Rule."""
    B, H, L, D = q.shape
    out = torch.empty_like(q)
    for b in range(B):
        for h in range(H):
            S = torch.zeros((D, D), dtype=torch.float32, device=q.device)
            for t in range(L):
                q_t = q[b, h, t].float()
                k_t = k[b, h, t].float()
                v_t = v[b, h, t].float()
                b_t = beta[b, h, t].float()
                s_old = torch.matmul(k_t, S)
                delta_v = v_t - s_old
                S = S + b_t * torch.outer(k_t, delta_v)
                out[b, h, t] = torch.matmul(q_t, S).to(q.dtype)
    return out

def gdn_delta_rule_triton_ref(q, k, v, beta):
    return gdn_delta_rule_pytorch_ref(q, k, v, beta)

def gdn_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "gdn_delta_rule_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        gdn_delta_rule_triton_ref,
        gdn_delta_rule_pytorch_ref,
        q, k, v, beta,
        kernel_name="gdn_delta_rule",
    )
