from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class Mamba2(nn.Module):
    r"""Simplified Mamba-2 State Space Model with parallel scan.

    [FIX-1] Vectorized parallel scan (no Python for-loop over sequence length).
    [FIX-2] Chunked scan for long sequences (bounded memory).
    [FIX-2b] Supports step-by-step inference via `past_state` / `next_state`.

    Recurrence:
        h_t = dA_t * h_{t-1} + dBx_t
        y_t = sum(h_t * D) * silu(z_t)

    where:
        dA_t = exp(dt_t * A)          (element-wise, A < 0)
        dBx_t = dt_t * x_t            (broadcast over state dim)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lora_gate = None
        self.d_model: int = d_model
        self.d_state: int = d_state
        self.expand: int = expand
        self.chunk_size: int = chunk_size
        inner_dim: int = d_model * expand
        self.inner_dim: int = inner_dim

        self.in_proj: nn.Linear = nn.Linear(d_model, inner_dim * 2)
        self.conv1d: nn.Conv1d = nn.Conv1d(inner_dim, inner_dim, d_conv, groups=inner_dim, padding=d_conv - 1)
        self.dt_proj: nn.Linear = nn.Linear(inner_dim, inner_dim)
        self.A_log: nn.Parameter = nn.Parameter(torch.log(torch.rand(inner_dim, d_state) * 0.1 + 1e-3))
        self.D: nn.Parameter = nn.Parameter(torch.ones(inner_dim))
        self.out_proj: nn.Linear = nn.Linear(inner_dim, d_model)

    @staticmethod
    def _vectorized_scan_with_chunks(
        dA: torch.Tensor,
        dBx: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        chunk_size: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked parallel scan for the linear recurrence with initial state.

        Returns:
            h: (B, L, inner, state)
            final_state: (B, inner, state)
        """
        B, L, inner, state = dA.shape
        device, dtype = dA.device, dA.dtype

        h = torch.empty(B, L, inner, state, device=device, dtype=dtype)

        running_h = (
            initial_state if initial_state is not None else torch.zeros(B, inner, state, device=device, dtype=dtype)
        )

        if L == 1:
            running_h = dA[:, 0] * running_h + dBx[:, 0]
            h[:, 0] = running_h
            return h, running_h

        num_chunks = (L + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, L)

            dA_chunk = dA[:, start:end]
            dBx_chunk = dBx[:, start:end]

            log_dA = torch.log(dA_chunk.clamp(min=1e-20))
            cum_log_dA = torch.cumsum(log_dA, dim=1)

            dBx_scaled = dBx_chunk * torch.exp(-cum_log_dA)
            cum_dBx = torch.cumsum(dBx_scaled, dim=1)

            h_intra = torch.exp(cum_log_dA) * cum_dBx
            h_chunk = h_intra + torch.exp(cum_log_dA) * running_h.unsqueeze(1)

            h[:, start:end] = h_chunk
            running_h = h_chunk[:, -1]

        return h, running_h

    def forward(
        self,
        x: torch.Tensor,
        past_state: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, L, d_model) input
            past_state: (B, inner, state) optional recurrent state

        Returns:
            y: (B, L, d_model) output
            next_state: (B, inner, state) final state for cache
        """
        B, L, D = x.shape
        inner_dim = self.inner_dim

        x_proj = self.in_proj(x)
        x_ssm, z = x_proj.chunk(2, dim=-1)

        x_ssm = self.conv1d(x_ssm.transpose(1, 2).contiguous())[..., :L].transpose(1, 2).contiguous()

        dt = F.softplus(self.dt_proj(x_ssm))
        A_f = -torch.exp(torch.clamp(self.A_log.float(), min=-20.0, max=10.0))

        chunk_len = min(L, 32) if L > 0 else 1
        num_chunks = (L + chunk_len - 1) // chunk_len if L > 0 else 0
        running_h = (
            past_state.float()
            if past_state is not None
            else torch.zeros(B, inner_dim, self.d_state, device=x.device, dtype=torch.float32)
        )
        D_weight = self.D.float().view(1, 1, inner_dim, 1)

        def _scan_chunk(
            x_c: torch.Tensor,
            dt_c: torch.Tensor,
            h_prev: torch.Tensor,
            A_param: torch.Tensor,
            d_w: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x_cf = x_c.float()
            dt_cf = dt_c.float()
            dA_log_c = torch.einsum("bli,in->blin", dt_cf, A_param).clamp(min=-20.0, max=0.0)
            cum_log_dA = torch.cumsum(dA_log_c, dim=1)

            dBx_c = x_cf.unsqueeze(-1) * dt_cf.unsqueeze(-1)
            dBx_scaled = dBx_c * torch.exp((-cum_log_dA).clamp(max=40.0))
            cum_dBx = torch.cumsum(dBx_scaled, dim=1)

            h_c = torch.exp(cum_log_dA) * (cum_dBx + h_prev.unsqueeze(1))
            y_c = (h_c * d_w).sum(dim=-1).to(x_c.dtype)
            return y_c, h_c[:, -1]

        use_ckpt = (
            self.training
            and torch.is_grad_enabled()
            and L > chunk_len
            and (x_ssm.requires_grad or dt.requires_grad or running_h.requires_grad)
        )

        y_chunks = []
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_len
            end = min(start + chunk_len, L)
            x_c = x_ssm[:, start:end]
            dt_c = dt[:, start:end]

            if use_ckpt:
                from torch.utils.checkpoint import checkpoint

                y_c, next_h = checkpoint(_scan_chunk, x_c, dt_c, running_h, A_f, D_weight, use_reentrant=False)
            else:
                y_c, next_h = _scan_chunk(x_c, dt_c, running_h, A_f, D_weight)

            running_h = next_h.detach() if not self.training else next_h
            y_chunks.append(y_c)

        y = (
            torch.cat(y_chunks, dim=1)
            if len(y_chunks) > 1
            else (y_chunks[0] if y_chunks else x_ssm.new_zeros(B, 0, inner_dim))
        )

        y = y * F.silu(z)
        return self.out_proj(y), running_h.to(x.dtype)
