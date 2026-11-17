import torch
import torch.nn as nn
from typing import Any

class Mamba2(nn.Module):
    r"""Simplified Mamba-2 State Space Model with parallel scan."""

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
        """Chunked parallel scan for the linear recurrence with initial state."""
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

        for i in range(0, L, chunk_size):
            end = min(i + chunk_size, L)
            chunk_dA = dA[:, i:end]
            chunk_dBx = dBx[:, i:end]
            
            chunk_h = torch.empty(B, end - i, inner, state, device=device, dtype=dtype)
            curr = running_h
            for t in range(end - i):
                curr = chunk_dA[:, t] * curr + chunk_dBx[:, t]
                chunk_h[:, t] = curr
            h[:, i:end] = chunk_h
            running_h = curr
        
        return h, running_h

    def forward(
        self,
        u: torch.Tensor,
        past_state: torch.Tensor | None = None,
        return_next_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = u.shape
        z, x = self.in_proj(u).chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)
        
        dt = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dBx = dt.unsqueeze(-1) * x.unsqueeze(-1)
        
        h, next_state = self._vectorized_scan_with_chunks(dA, dBx, initial_state=past_state, chunk_size=self.chunk_size)
        
        y = (h * self.D.view(1, 1, -1, 1)).sum(dim=-1) * F.silu(z)
        out = self.out_proj(y)
        
        if return_next_state:
            return out, next_state
        return out
