# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Scan routines and state reading for MambaBlock."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

try:
    from neuralops.core.selective_scan import HAS_MAMBA_CUDA, selective_scan
except Exception as e:
    logging.getLogger(__name__).warning("selective_scan import failed: %s", e)
    selective_scan = None
    HAS_MAMBA_CUDA = False

try:
    from neuralops.ops.cuda.selective_scan.selective_scan_autograd import (
        _get_module as _get_cuda_ss_module,
        selective_scan as cuda_selective_scan,
    )
except Exception as _e:
    logging.getLogger(__name__).warning("CUDA selective_scan unavailable: %s", _e)
    cuda_selective_scan = None
    _get_cuda_ss_module = None

_WARNED_CUDA_SS_UNAVAILABLE = False

try:
    from neuralops.ops.triton.selective_scan import selective_scan_fn as triton_selective_scan

    HAS_TRITON_MAMBA = True
except Exception as _e:
    logging.getLogger(__name__).warning("Triton selective_scan unavailable: %s", _e)
    triton_selective_scan = None
    HAS_TRITON_MAMBA = False

logger = logging.getLogger(__name__)

def _read_state(self, past_state: Any) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
    """Returns (h, conv_state, dt_state, past_len)."""
    if past_state is None:
        return None, None, None, 0
    if isinstance(past_state, (tuple, list)):
        if len(past_state) >= 3:
            h = past_state[0]
            conv_state = past_state[1]
            dt_state = past_state[2] if len(past_state) > 2 else None
            past_len = h.shape[-1] if h is not None else 0
            if conv_state is not None:
                past_len = conv_state.shape[-1] - (self.d_conv - 1)
                past_len = max(0, past_len)
            return h, conv_state, dt_state, past_len
        elif len(past_state) == 2:
            return past_state[0], past_state[1], None, 0
        elif len(past_state) == 1:
            return past_state[0], None, None, 0
    return None, None, None, 0

def _recurrent_scan(
    self,
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_ssm: torch.Tensor,
    C_ssm: torch.Tensor,
    D: torch.Tensor,
    state: torch.Tensor | None,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recurrent selective scan with chunked activation checkpointing and bmm.

    Computes discretized parameters inside each chunk so full-sequence
    (B, d_inner, L, d_state) tensors are never retained in VRAM.
    """
    B, d_inner, L = u.shape
    d_state = A.shape[1]

    target_dtype = u.dtype
    h = (
        state.to(device=u.device, dtype=torch.float32)
        if state is not None
        else torch.zeros(B, d_inner, d_state, device=u.device, dtype=torch.float32)
    )

    A_f = A.float().view(1, d_inner, 1, d_state)

    def _scan_chunk(
        h_init: torch.Tensor,
        u_c: torch.Tensor,
        delta_c: torch.Tensor,
        b_c: torch.Tensor,
        c_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt_c = F.softplus(delta_c.float())
        da_c = torch.exp((dt_c.unsqueeze(-1) * A_f).clamp(-20.0, 0.0))
        dbu_c = dt_c.unsqueeze(-1) * b_c.float().permute(0, 2, 1).unsqueeze(1) * u_c.float().unsqueeze(-1)
        c_f = c_c.float()
        L_c = u_c.shape[2]
        ys_c = []
        curr_h = h_init
        for i in range(L_c):
            curr_h = da_c[:, :, i, :] * curr_h + dbu_c[:, :, i, :]
            c_t = c_f[:, :, i].contiguous().unsqueeze(-1)
            y_t = torch.bmm(curr_h, c_t).squeeze(-1)
            ys_c.append(y_t)
        return torch.stack(ys_c, dim=-1), curr_h

    ys_all = []
    import torch.utils.checkpoint as cp

    use_cp = torch.is_grad_enabled()

    for s in range(0, L, chunk_size):
        e = min(s + chunk_size, L)
        u_slice = u[:, :, s:e]
        dt_slice = delta[:, :, s:e]
        b_slice = B_ssm[:, :, s:e]
        c_slice = C_ssm[:, :, s:e]

        if use_cp:
            y_chunk, h = cp.checkpoint(
                _scan_chunk,
                h,
                u_slice,
                dt_slice,
                b_slice,
                c_slice,
                use_reentrant=False,
            )
        else:
            y_chunk, h = _scan_chunk(
                h,
                u_slice,
                dt_slice,
                b_slice,
                c_slice,
            )
        ys_all.append(y_chunk)

    y = torch.cat(ys_all, dim=-1).to(target_dtype)
    y = y + u * D.view(1, d_inner, 1).to(target_dtype)
    return y, h.to(target_dtype)

def _parallel_scan(
    self,
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_ssm: torch.Tensor,
    C_ssm: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel scan dispatch with 3-tier fallback:
      Path 1: Official CUDA kernel via selective_scan (Linux only, when compiled)
      Path 2: Triton selective_scan (if mamba_ssm Triton ops available)
      Path 3: Fast chunked parallel scan (framework.core.selective_scan, O(1) ops per chunk)
      Path 4: Chunked recurrent scan with activation checkpointing (failsafe fallback)
    """
    global _WARNED_CUDA_SS_UNAVAILABLE
    B_sz, D_dim, L_sz = u.shape
    N_state = A.shape[1]
    is_pow2_n = N_state > 0 and (N_state & (N_state - 1)) == 0
    ws_bytes = B_sz * D_dim * L_sz * N_state * 4
    allow_full_ws = (not torch.is_grad_enabled()) or (ws_bytes <= 32 * 1024 * 1024)

    # Level 1 Native CUDA C++ (128-bit register cached)
    if cuda_selective_scan is not None and u.is_cuda and N_state <= 128 and allow_full_ws:
        cuda_mod = _get_cuda_ss_module() if _get_cuda_ss_module is not None else None
        if cuda_mod is not None:
            try:
                delta_sp = F.softplus(delta)
                res = cuda_selective_scan(u, delta_sp, A, B_ssm, C_ssm, D)
                return res[0] if isinstance(res, tuple) else res
            except Exception as e:
                logger.warning("Level 1 CUDA selective_scan failed (%s); trying Level 2 Triton.", e)
        elif not _WARNED_CUDA_SS_UNAVAILABLE:
            _WARNED_CUDA_SS_UNAVAILABLE = True
            logger.warning(
                "MambaBlock: CUDA selective_scan extension unavailable (compilation failed); "
                "falling back to Level 2 Triton / chunked PyTorch."
            )

    # Level 2 Triton JIT (In-SRAM zero-alloc)
    if HAS_TRITON_MAMBA and triton_selective_scan is not None and u.is_cuda and is_pow2_n and allow_full_ws:
        try:
            y = triton_selective_scan(
                u.transpose(1, 2).contiguous(),
                delta.transpose(1, 2).contiguous(),
                A.contiguous(),
                B_ssm.transpose(1, 2).contiguous(),
                C_ssm.transpose(1, 2).contiguous(),
                D.contiguous() if D is not None else None,
                delta_softplus=True,
            )
            return y.transpose(1, 2).contiguous()
        except Exception as e:
            logger.warning("Level 2 Triton selective_scan failed: %s; trying Level 3 fallback.", e)

    # Level 3 PyTorch Reference / Core Dispatcher Fallback (when workspace fits budget or non-CUDA)
    if selective_scan is not None and ((not u.is_cuda) or allow_full_ws):
        try:
            return selective_scan(
                u=u,
                delta=delta,
                A=A,
                B=B_ssm,
                C=C_ssm,
                D=D,
                delta_softplus=True,
            )
        except Exception as e:
            logger.warning("Core selective_scan dispatch failed (%s); using chunked recurrent scan.", e)

    y, _ = self._recurrent_scan(u, delta, A, B_ssm, C_ssm, D, None, chunk_size=32)
    return y

