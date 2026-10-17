"""
Gated DeltaNet Chunked Scan Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "gdn_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "chunked_delta_gate_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_gdn_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["chunked_delta_gate_scan_fwd"],
        )
    return _MODULE

def chunked_delta_gate_scan_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    from neuralops.core.triton_kernels.gdn_scan import triton_chunked_delta_gate_scan_pytorch_ref
    return triton_chunked_delta_gate_scan_pytorch_ref(q, k, v, alpha, beta)

def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.gdn_scan import triton_chunked_delta_gate_scan
        return triton_chunked_delta_gate_scan
    except ImportError:
        return None

def _chunked_delta_gate_scan_fwd_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    if q.shape[-1] > 64 or v.shape[-1] > 64:
        cuda_fn = None
    else:
        def cuda_fn(q, k, v, alpha, beta, **kwargs):
            mod = _get_module()
            return mod.chunked_delta_gate_scan_fwd(
                q.contiguous(), k.contiguous(), v.contiguous(), alpha.contiguous(), beta.contiguous()
            )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        chunked_delta_gate_scan_pytorch_ref,
        q,
        k,
        v,
        alpha,
        beta,
        kernel_name="chunked_delta_gate_scan",
    )

class _GDNScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, alpha, beta):
        out = _chunked_delta_gate_scan_fwd_dispatch(q, k, v, alpha, beta)
        ctx.save_for_backward(q, k, v, alpha, beta)
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        q, k, v, alpha, beta = ctx.saved_tensors
        q_p = q.detach().requires_grad_(q.requires_grad)
        k_p = k.detach().requires_grad_(k.requires_grad)
        v_p = v.detach().requires_grad_(v.requires_grad)
        alpha_p = alpha.detach().requires_grad_(alpha.requires_grad)
        beta_p = beta.detach().requires_grad_(beta.requires_grad)

        with torch.enable_grad():
            out_p = chunked_delta_gate_scan_pytorch_ref(q_p, k_p, v_p, alpha_p, beta_p)

        grads = torch.autograd.grad(out_p, [q_p, k_p, v_p, alpha_p, beta_p], dout, allow_unused=True)
        return grads[0], grads[1], grads[2], grads[3], grads[4]

def chunked_delta_gate_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    if any(t.requires_grad for t in (q, k, v, alpha, beta) if isinstance(t, torch.Tensor)):
        return _GDNScanFunction.apply(q, k, v, alpha, beta)
    return _chunked_delta_gate_scan_fwd_dispatch(q, k, v, alpha, beta)
