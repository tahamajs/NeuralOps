#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SVD Linear Decay Autograd Interface.

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients,
Triton JIT secondary tier, and vectorized PyTorch reference fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

logger = logging.getLogger(__name__)

_MODULE = None
_MODULE_ERROR: Exception | None = None


def _get_module():
    global _MODULE, _MODULE_ERROR
    if _MODULE_ERROR is not None:
        return None
    if _MODULE is None:
        try:
            curr_dir = Path(__file__).parent
            with open(curr_dir / "svd_linear_decay_binding.cpp", "r", encoding="utf-8") as f:
                cpp_src = f.read()
            with open(curr_dir / "svd_linear_decay.cu", "r", encoding="utf-8") as f:
                cuda_src = f.read()

            _MODULE = load_cuda_module(
                name="cuda_svd_linear_decay",
                cpp_sources=cpp_src,
                cuda_sources=cuda_src,
                functions=["svd_linear_decay_fwd"],
            )
        except Exception as exc:
            _MODULE_ERROR = exc
            logger.warning(
                "SVD Linear Decay CUDA extension unavailable; using lower-level fallback: %s",
                exc,
            )
            return None
    return _MODULE


def svd_linear_decay_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_decay: torch.Tensor,
    chunk_size: int = 64,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Vectorized PyTorch reference implementation (Level 3 fallback).

    Chunkwise scan parallelization with intra-chunk causal attention and boundary state.
    """
    B, H, N, D_k = q.shape
    D_v = v.shape[-1]
    compute_dtype = torch.float32

    phi_q = F.elu(q.to(compute_dtype)) + 1.0
    phi_k = F.elu(k.to(compute_dtype)) + 1.0
    v_f = v.to(compute_dtype)

    channel_wise = bool(lambda_decay.dim() == 4 and lambda_decay.shape[-1] == D_k)

    if not channel_wise:
        num_chunks = (N + chunk_size - 1) // chunk_size
        S_chunk = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
        Z_chunk = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)
        outputs = []

        lam_dec = lambda_decay.to(compute_dtype)
        if lam_dec.dim() == 4:
            lam_dec = lam_dec.squeeze(-1).squeeze(-1)
        elif lam_dec.dim() == 2:
            lam_dec = lam_dec.unsqueeze(-1).expand(B, H, N)

        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)
            qc = phi_q[:, :, start:end, :]
            kc = phi_k[:, :, start:end, :]
            vc = v_f[:, :, start:end, :]
            lam_c = lam_dec[:, :, start:end]
            Lc = end - start

            log_lam_c = torch.log(torch.clamp(lam_c, min=1e-6, max=1.0))
            log_gamma_c = torch.cumsum(log_lam_c, dim=-1)

            log_M_intra = log_gamma_c.unsqueeze(-1) - log_gamma_c.unsqueeze(-2)
            causal_bool = torch.tril(torch.ones(Lc, Lc, dtype=torch.bool, device=qc.device))
            M_intra = torch.exp(torch.clamp(log_M_intra, max=0.0)) * causal_bool.unsqueeze(
                0
            ).unsqueeze(0)

            aff_intra = torch.matmul(qc, kc.transpose(-1, -2).contiguous()) * M_intra
            num_intra = torch.matmul(aff_intra, vc)
            den_intra = aff_intra.sum(dim=-1, keepdim=True)

            decay_from_past = torch.exp(torch.clamp(log_gamma_c, max=0.0)).unsqueeze(-1)
            num_inter = torch.matmul(qc, S_chunk) * decay_from_past
            den_inter = torch.matmul(qc, Z_chunk) * decay_from_past

            out_c = (num_intra + num_inter) / (den_intra + den_inter + eps)
            outputs.append(out_c)

            total_chunk_decay = torch.exp(
                torch.clamp(log_gamma_c[:, :, -1:], max=0.0)
            ).unsqueeze(-1)
            log_w_to_end = log_gamma_c[:, :, -1:] - log_gamma_c
            w_to_end = torch.exp(torch.clamp(log_w_to_end, max=0.0)).unsqueeze(-1)
            kc_w = kc * w_to_end

            S_chunk = total_chunk_decay * S_chunk + torch.matmul(
                kc_w.transpose(-1, -2).contiguous(), vc
            )
            Z_chunk = (
                total_chunk_decay * Z_chunk
                + kc_w.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()
            )

        return torch.cat(outputs, dim=2).to(q.dtype)

    # Channel-wise sequential scan fallback
    S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
    Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)
    outputs = []
    lam_seq = lambda_decay.to(compute_dtype)
    if lam_seq.dim() == 3:
        lam_seq = lam_seq.unsqueeze(-1)

    for t in range(N):
        qt = phi_q[:, :, t : t + 1, :]
        kt = phi_k[:, :, t : t + 1, :]
        vt = v_f[:, :, t : t + 1, :]
        lam_t = lam_seq[:, :, t : t + 1, :].transpose(-1, -2).contiguous()

        kt_t = kt.transpose(-1, -2).contiguous()
        kv_t = torch.matmul(kt_t, vt)
        S = lam_t * S + kv_t
        Z = lam_t * Z + kt_t

        num_t = torch.matmul(qt, S)
        den_t = torch.matmul(qt, Z) + eps
        outputs.append(num_t / den_t)

    return torch.cat(outputs, dim=2).to(q.dtype)


def _svd_linear_decay_fwd_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_decay: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    mod = _get_module()

    def cuda_fn(q_in, k_in, v_in, lam_in):
        return mod.svd_linear_decay_fwd(
            q_in.contiguous(),
            k_in.contiguous(),
            v_in.contiguous(),
            lam_in.contiguous(),
            eps,
        )

    triton_fn = None
    try:
        from neuralops.ops.triton.svd_linear_decay_triton import svd_linear_decay_triton

        def _triton_wrapper(q_in, k_in, v_in, lam_in):
            return svd_linear_decay_triton(q_in, k_in, v_in, lam_in, eps=eps)

        triton_fn = _triton_wrapper
    except Exception as exc:
        logger.warning("Failed to initialize Triton implementation for svd_linear_decay: %s", exc)

    return dispatch(
        cuda_fn if (mod is not None and q.shape[-1] <= 64 and v.shape[-1] <= 64) else None,
        triton_fn,
        svd_linear_decay_pytorch_ref,
        q,
        k,
        v,
        lambda_decay,
        kernel_name="svd_linear_decay",
    )


class _SVDLinearDecayFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lambda_decay: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        out = _svd_linear_decay_fwd_dispatch(q, k, v, lambda_decay, eps=eps)
        ctx.save_for_backward(q, k, v, lambda_decay)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx: Any, *douts: Any) -> tuple[torch.Tensor | None, ...]:
        dout = douts[0] if len(douts) == 1 else douts
        q, k, v, lambda_decay = ctx.saved_tensors

        q_p = q.detach().requires_grad_(q.requires_grad)
        k_p = k.detach().requires_grad_(k.requires_grad)
        v_p = v.detach().requires_grad_(v.requires_grad)
        lam_p = lambda_decay.detach().requires_grad_(lambda_decay.requires_grad)

        diff_tensors = [t for t in (q_p, k_p, v_p, lam_p) if t.requires_grad]
        if not diff_tensors:
            return None, None, None, None, None

        with torch.enable_grad():
            out_p = svd_linear_decay_pytorch_ref(q_p, k_p, v_p, lam_p, eps=ctx.eps)

        grads = torch.autograd.grad(out_p, diff_tensors, dout, allow_unused=True)
        grad_map = dict(zip(diff_tensors, grads))

        dq = grad_map.get(q_p)
        dk = grad_map.get(k_p)
        dv = grad_map.get(v_p)
        dlam = grad_map.get(lam_p)
        return dq, dk, dv, dlam, None


def svd_linear_decay(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_decay: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """SVD Linear Decay Attention operator with 3-tier fallback hierarchy."""
    if any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in (q, k, v, lambda_decay)
    ):
        return _SVDLinearDecayFunction.apply(q, k, v, lambda_decay, eps)
    return _svd_linear_decay_fwd_dispatch(q, k, v, lambda_decay, eps=eps)


__all__ = [
    "svd_linear_decay",
    "svd_linear_decay_pytorch_ref",
]
