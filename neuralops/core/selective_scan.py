#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Selective Scan Kernel Interface

State space selective scan operator providing CUDA mamba_ssm bindings and numerically stable PyTorch fallbacks.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _mamba_scan_fn,
    )

    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False
    _mamba_scan_fn = None


def _auto_chunk_size(
    D_dim: int,
    N: int,
    B_size: int,
    max_bytes: int = 32 * 1024 * 1024,
    default: int = 16,
    min_chunk: int = 4,
) -> int:
    """
    Choose chunk_size so the (B, D, C, C, N) intermediate fits in max_bytes.

    Memory per chunk ≈ B_size * D_dim * C^2 * N * 4 bytes (fp32).
    Solve for C.
    """
    denom = B_size * D_dim * N * 4
    if denom <= 0:
        return default
    max_c2 = max_bytes // denom
    if max_c2 < 1:
        return min_chunk
    c = int(math.isqrt(max_c2))
    c = max(min_chunk, min(c, default))
    return c


def selective_scan_chunked(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    Selective scan via numerically stable chunked recurrence.

    For chunk [s, e) of length C:
      dA[t] = Δ[t] · A <= 0
      dA_cum[t] = cumsum(dA)
      decay[j, i] = exp(dA_cum[j] - dA_cum[i]) for j >= i, else 0
      h_intra[j] = sum_i decay[j, i] · dB_u[i]
      h_carry = exp(dA_cum) · h_prev
      h_chunk = h_intra + h_carry
      y_chunk = sum_N(h_chunk · C)
    """
    B_size, D_dim, L = u.shape
    if L == 0:
        y = torch.zeros_like(u)
        if z is not None:
            y = y * F.silu(z)
        return y

    N = A.shape[1]
    device = u.device
    dtype = u.dtype

    u_f = u.float() if dtype != torch.float32 else u
    delta_f = delta.float() if delta.dtype != torch.float32 else delta
    A_f = A.float() if A.dtype != torch.float32 else A
    B_f = B.float() if B.dtype != torch.float32 else B
    C_f = C.float() if C.dtype != torch.float32 else C

    if delta_bias is not None:
        delta_f = delta_f + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta_f = F.softplus(delta_f)


    dA = delta_f.unsqueeze(-1) * A_f.unsqueeze(0).unsqueeze(2)
    dA = dA.clamp(max=0.0)

    B_f_t = B_f.transpose(1, 2).contiguous()
    dB_u = delta_f.unsqueeze(-1) * u_f.unsqueeze(-1) * B_f_t.unsqueeze(1)

    if L <= 128:
        dA_exp = torch.exp(dA)
        h = torch.zeros(B_size, D_dim, N, device=device, dtype=torch.float32)
        use_cp = torch.is_grad_enabled()
        if use_cp:
            import torch.utils.checkpoint as cp

        def _step_subchunk_scan(h_in: torch.Tensor, s_idx: int, e_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            curr_h = h_in
            sub_ys = []
            for t in range(s_idx, e_idx):
                dA_t = dA_exp[:, :, t, :]
                dBu_t = dB_u[:, :, t, :]
                curr_h = dA_t * curr_h + dBu_t
                yt = torch.sum(curr_h * C_f[:, :, t].unsqueeze(1), dim=-1)
                sub_ys.append(yt)
            return torch.stack(sub_ys, dim=2), curr_h

        ys = []
        c_step = 16
        for s in range(0, L, c_step):
            e = min(s + c_step, L)
            if use_cp:
                chunk_y, h = cp.checkpoint(_step_subchunk_scan, h, s, e, use_reentrant=False)
            else:
                chunk_y, h = _step_subchunk_scan(h, s, e)
            ys.append(chunk_y)
        y = torch.cat(ys, dim=2)
    else:
        if chunk_size <= 0:
            chunk_size = _auto_chunk_size(D_dim, N, B_size)
        else:
            max_chunk = _auto_chunk_size(D_dim, N, B_size, max_bytes=64 * 1024 * 1024)
            chunk_size = min(chunk_size, max_chunk)
        chunk_size = max(1, chunk_size)

        num_chunks = (L + chunk_size - 1) // chunk_size
        h_prev = torch.zeros(B_size, D_dim, N, device=device, dtype=torch.float32)

        y_chunks = []
        for c in range(num_chunks):
            s = c * chunk_size
            e = min(L, s + chunk_size)
            C_len = e - s

            dA_c = dA[:, :, s:e, :]
            dBu_c = dB_u[:, :, s:e, :]
            C_c = C_f[:, :, s:e].transpose(1, 2)

            dA_cum = torch.cumsum(dA_c, dim=2)

            log_decay = dA_cum.unsqueeze(3) - dA_cum.unsqueeze(2)
            tril = torch.tril(torch.ones(C_len, C_len, dtype=torch.bool, device=device))
            log_decay = log_decay.masked_fill(~tril.view(1, 1, C_len, C_len, 1), float("-inf"))
            decay = torch.exp(log_decay)

            BD = B_size * D_dim
            decay_bd = decay.reshape(BD, C_len, C_len, N)
            dBu_bd = dBu_c.reshape(BD, C_len, N)
            h_intra_bd = torch.einsum("bjic,bic->bjc", decay_bd, dBu_bd)
            h_intra = h_intra_bd.reshape(B_size, D_dim, C_len, N)

            h_carry = torch.exp(dA_cum) * h_prev.unsqueeze(2)
            h_chunk = h_intra + h_carry

            y_c = torch.einsum("bdcn,bcn->bdc", h_chunk, C_c)

            y_chunks.append(y_c)
            h_prev = h_chunk[:, :, -1, :]

        y = torch.cat(y_chunks, dim=2)

    if D is not None:
        y = y + D.float().view(1, -1, 1) * u_f

    if z is not None:
        y = y * F.silu(z.float())

    return y.to(dtype)


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    Dispatch to CUDA kernel if available, else stable chunked fallback.
    """
    if HAS_MAMBA_CUDA and u.is_cuda:
        try:
            return _mamba_scan_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        except Exception as e:
            logger.warning(
                "mamba_ssm CUDA selective_scan failed (%s); "
                "trying Triton selective_scan_fn.",
                e,
            )

    if u.is_cuda and z is None and delta_bias is None:
        try:
            from neuralops.core.triton_kernels.selective_scan import selective_scan_fn as _triton_scan_fn
            return _triton_scan_fn(u, delta, A, B, C, D=D, delta_softplus=delta_softplus)
        except Exception as e:
            logger.debug("Triton selective_scan_fn failed, falling back to chunked PyTorch: %s", e)

    return selective_scan_chunked(
        u=u,
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        chunk_size=chunk_size,
    )


__all__ = [
    "selective_scan",
    "selective_scan_chunked",
    "HAS_MAMBA_CUDA",
]
