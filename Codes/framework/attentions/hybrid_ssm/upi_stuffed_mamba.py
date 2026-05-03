import torch
import torch.nn as nn
import torch.nn.functional as F

class UPIMambaBlock(nn.Module):
    """
    UPI‑Stuffed Mamba: DPLR structure with UPI stabilization.
    Drop‑in replacement for attention layers.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        inner_dim = dim * expand
        self.in_proj = nn.Linear(dim, inner_dim * 2)
        self.conv1d = nn.Conv1d(inner_dim, inner_dim, d_conv, groups=inner_dim, padding=d_conv-1)
        # DPLR parameters
        self.A_log = nn.Parameter(torch.log(torch.rand(dim, d_state) * 0.1))
        self.D = nn.Parameter(torch.ones(dim))
        self.dt_proj = nn.Linear(inner_dim, dim)
        # UPI control
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_proj = nn.Linear(inner_dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        x_proj = self.in_proj(x)   # (B, L, 2*inner)
        x_ssm, z = x_proj.chunk(2, dim=-1)   # each (B, L, inner)
        # Causal conv
        x_ssm = x_ssm.transpose(1,2)         # (B, inner, L)
        x_ssm = self.conv1d(x_ssm)[..., :L]
        x_ssm = x_ssm.transpose(1,2)         # (B, L, inner)

        # Selective scan over inner dimension? Mamba scans over input channels.
        # We'll do a simplified scan using first dim dimensions for clarity.
        # Real Mamba uses all inner dimensions; we'll approximate by taking the first dim elements.
        x_ssm_trunc = x_ssm[:, :, :self.dim]  # (B, L, dim)
        dt = F.softplus(self.dt_proj(x_ssm_trunc))   # (B, L, dim)
        A = -torch.exp(self.A_log)                    # (dim, d_state)

        # Recurrent scan
        h = torch.zeros(B, self.dim, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            xt = x_ssm_trunc[:, t]                      # (B, dim)
            # State update: h = h * exp(dt * A) + xt * B  (B simplified as dt)
            h = h * torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0)) + \
                (xt.unsqueeze(-1) * dt[:, t].unsqueeze(-1))
            # UPI stabilization: pull toward zero
            h = h - self.gamma * h
            out_t = (h * self.D.unsqueeze(-1)).sum(dim=-1)   # (B, dim)
            outputs.append(out_t)
        out_ssm = torch.stack(outputs, dim=1)   # (B, L, dim)
        # Combine with gated branch (z)
        out_ssm = out_ssm * F.silu(z[:, :, :self.dim])
        # Project back
        return self.out_proj(torch.cat([out_ssm, x_ssm_trunc], dim=-1))   # fill inner dim later? Simplified to keep dim
        # Better: just use out_proj on the concatenation of ssm and residual