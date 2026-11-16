# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""SSM scan routines for MambaSVDFlash."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from .mamba_svd_flash_state import SVDS6StateSpace

logger = logging.getLogger(__name__)

def _ssm_scan_from_module(
    self,
    ssm_mod: SVDS6StateSpace,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-bounded chunked SVD-S6 scan that projects U/V/Sigma/B/C per chunk
    inside activation checkpointing so (B, L, D, r) tensors are never
    retained across the full sequence or across layers.
    """
    B, L, D = x.shape
    r = ssm_mod.rank
    device = x.device
    z_all = ssm_mod.down(x.to(ssm_mod.down.weight.dtype))

    h = (
        torch.zeros(B, D, device=device, dtype=torch.float32)
        if h0 is None
        else h0.to(device=device, dtype=torch.float32)
    )

    use_cp = torch.is_grad_enabled()
    if use_cp:
        import torch.utils.checkpoint as cp

    def _chunk_fwd(
        h_in: torch.Tensor,
        z_c: torch.Tensor,
        x_c: torch.Tensor,
        m_c: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_len = z_c.shape[1]
        U_c = F.normalize(ssm_mod.up_u(z_c).view(B, c_len, D, r).float(), p=2, dim=-2, eps=1e-6)
        V_c = F.normalize(ssm_mod.up_v(z_c).view(B, c_len, r, D).float(), p=2, dim=-2, eps=1e-6)
        Sig_c = torch.sigmoid(ssm_mod.up_sigma(z_c).float())
        B_c = ssm_mod.up_b(z_c).float()
        C_c = ssm_mod.up_c(z_c).float()

        US_c = U_c * Sig_c.unsqueeze(-2)
        bx_c = B_c * x_c.float()

        curr_h = h_in
        curr_col = curr_h.unsqueeze(-1)
        sub_h = []
        for t in range(c_len):
            v_t = torch.bmm(V_c[:, t], curr_col)
            h_trans = torch.bmm(US_c[:, t], v_t).squeeze(-1)
            h_next = h_trans + bx_c[:, t]
            if m_c is not None:
                curr_h = torch.where(m_c[:, t : t + 1].bool(), h_next, curr_h)
            else:
                curr_h = h_next
            curr_h = torch.nan_to_num(curr_h, nan=0.0, posinf=1e4, neginf=-1e4)
            curr_col = curr_h.unsqueeze(-1)
            sub_h.append(curr_h)
        H_c = torch.stack(sub_h, dim=1)
        return C_c * H_c, curr_h

    out_chunks = []
    for s in range(0, L, chunk_size):
        e = min(s + chunk_size, L)
        z_slice = z_all[:, s:e]
        x_slice = x[:, s:e]
        m_slice = mask[:, s:e] if mask is not None else None
        if use_cp:
            out_c, h = cp.checkpoint(_chunk_fwd, h, z_slice, x_slice, m_slice, use_reentrant=False)
        else:
            out_c, h = _chunk_fwd(h, z_slice, x_slice, m_slice)
        out_chunks.append(out_c)

    out = torch.cat(out_chunks, dim=1)
    return out.to(x.dtype), h.to(x.dtype)

def _ssm_scan(
    self,
    U: torch.Tensor,
    Sigma: torch.Tensor,
    V: torch.Tensor,
    B_seq: torch.Tensor,
    C_seq: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sequential scan with a chunked loop.
    Returns (outputs, h_final).
    - For forward pass, h_final corresponds to step L-1.
    - For reverse pass, the caller flips before/after.
    """
    B, L, D = x.shape
    r = Sigma.shape[-1]
    device = x.device
    is_pow2_r = r > 0 and (r & (r - 1)) == 0
    ws_bytes = B * D * L * r * 4
    allow_triton = (not torch.is_grad_enabled()) or (ws_bytes <= 32 * 1024 * 1024)

    if x.is_cuda and mask is None and h0 is None and not reverse and r <= 64 and is_pow2_r and allow_triton:
        try:
            from neuralops.ops.triton._types import check_triton_suitability
            from neuralops.ops.triton.selective_scan import selective_scan_fn

            suitable, _ = check_triton_suitability(
                x, "selective_scan", head_dim=None, seq_len=L, supports_backward=True
            )
            if suitable:
                delta = Sigma.mean(dim=-1, keepdim=True).expand(-1, -1, D).contiguous()
                A = -torch.ones(D, r, device=device, dtype=torch.float32)
                B_mat = V.mean(dim=-1).transpose(1, 2).contiguous()
                C_mat = U.mean(dim=-2).transpose(1, 2).contiguous()
                y_triton = selective_scan_fn(x.contiguous(), delta, A, B_mat, C_mat, D=None, delta_softplus=False)
                return y_triton.to(x.dtype), y_triton[:, -1].to(x.dtype)
        except Exception as e:
            logger.warning("Triton selective scan failed in MambaSVDFlash (%s); falling back to chunked PyTorch.", e)

    h = torch.zeros(B, D, device=device, dtype=torch.float32) if h0 is None else h0.float()
    US = (U.float() * Sigma.float().unsqueeze(-2)).contiguous()
    V_f = V.float().contiguous()
    bx = (B_seq.float() * x.float()).contiguous()

    steps = list(range(L - 1, -1, -1) if reverse else range(L))
    chunk_size = 32
    use_cp = torch.is_grad_enabled()
    if use_cp:
        import torch.utils.checkpoint as cp

    def _step_subchunk(h_in: torch.Tensor, sub_steps: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        sub_h_list = []
        curr_h = h_in
        curr_col = curr_h.unsqueeze(-1)
        for t in sub_steps:
            v_t = torch.bmm(V_f[:, t], curr_col)
            h_trans = torch.bmm(US[:, t], v_t).squeeze(-1)
            h_next = h_trans + bx[:, t]
            if mask is not None:
                m_t = mask[:, t].unsqueeze(-1).to(h_next.dtype)
                curr_h = torch.where(m_t.bool(), h_next, curr_h)
            else:
                curr_h = h_next
            curr_h = torch.nan_to_num(curr_h, nan=0.0, posinf=1e4, neginf=-1e4)
            curr_col = curr_h.unsqueeze(-1)
            sub_h_list.append(curr_h)
        return torch.stack(sub_h_list, dim=1), curr_h

    h_chunks = []
    for s in range(0, L, chunk_size):
        sub_steps = steps[s : min(s + chunk_size, L)]
        if use_cp:
            chunk_H, h = cp.checkpoint(_step_subchunk, h, sub_steps, use_reentrant=False)
        else:
            chunk_H, h = _step_subchunk(h, sub_steps)
        h_chunks.append(chunk_H)

    H = torch.cat(h_chunks, dim=1)
    if reverse:
        H = torch.flip(H, dims=[1])

    out = C_seq.float() * H
    return out.to(x.dtype), h.to(x.dtype)

