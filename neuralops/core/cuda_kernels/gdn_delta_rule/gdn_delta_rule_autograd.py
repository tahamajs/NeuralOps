from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
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

def gdn_delta_rule_triton_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return gdn_delta_rule_pytorch_ref(q, k, v, beta)

def _gdn_delta_rule_fwd_dispatch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    def cuda_fn(q, k, v, beta, **kwargs):
        mod = _get_module()
        return mod.gdn_delta_rule_fwd(q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous())
    return dispatch(
        cuda_fn,
        gdn_delta_rule_triton_ref,
        gdn_delta_rule_pytorch_ref,
        q, k, v, beta,
        kernel_name="gdn_delta_rule",
    )

class _GDNDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, beta):
        out = _gdn_delta_rule_fwd_dispatch(q, k, v, beta)
        ctx.save_for_backward(q, k, v, beta)
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        dout = douts[0] if len(douts) == 1 else douts
        q, k, v, beta = ctx.saved_tensors
        q_p = q.detach().requires_grad_(q.requires_grad)
        k_p = k.detach().requires_grad_(k.requires_grad)
        v_p = v.detach().requires_grad_(v.requires_grad)
        beta_p = beta.detach().requires_grad_(beta.requires_grad)

        with torch.enable_grad():
            out_p = gdn_delta_rule_pytorch_ref(q_p, k_p, v_p, beta_p)

        grads = torch.autograd.grad(out_p, [q_p, k_p, v_p, beta_p], dout, allow_unused=True)
        return grads[0], grads[1], grads[2], grads[3]

def gdn_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    if any(t.requires_grad for t in (q, k, v, beta) if isinstance(t, torch.Tensor)):
        return _GDNDeltaRuleFunction.apply(q, k, v, beta)
    return _gdn_delta_rule_fwd_dispatch(q, k, v, beta)
