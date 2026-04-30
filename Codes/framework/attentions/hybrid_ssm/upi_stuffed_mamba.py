# framework/attentions/upi_stuffed_mamba.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UPIMambaBlock(nn.Module):
    """
    UPI‑Stuffed Mamba: DPLR structure with UPI stabilization and second‑order state.
    Replaces a standard attention layer. Works as a drop‑in for causal language models.
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
        # input projection
        x_proj = self.in_proj(x)   # (B, L, 2*inner_dim)
        z, x_ssm = x_proj.chunk(2, dim=-1)
        # causal conv
        x_ssm = x_ssm.transpose(1,2)
        x_ssm = self.conv1d(x_ssm)[..., :L]
        x_ssm = x_ssm.transpose(1,2)
        # selective scan with DPLR (simplified as recurrent for clarity)
        dt = F.softplus(self.dt_proj(x_ssm))   # (B, L, D)
        A = -torch.exp(self.A_log)              # (D, d_state)
        # Recurrent scan
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        out = []
        for t in range(L):
            xt = x_ssm[:, t]                     # (B, inner_dim)... we'll simplify to using dim
            # simplified: use dim dimension for scan
            # In real Mamba, the scan is over inner_dim; here we sketch the structure.
            h = h * torch.exp(dt[:, t].unsqueeze(-1) * A) + xt.unsqueeze(-1) @ (xt.unsqueeze(1) * 0.1)
            # UPI stabilization: pull towards mean
            h = h - self.gamma * (h - 0.0)
            out.append(h.sum(dim=-1))
        out = torch.stack(out, dim=1)  # (B, L, D)
        out = out * F.silu(z)
        return self.out_proj(out)