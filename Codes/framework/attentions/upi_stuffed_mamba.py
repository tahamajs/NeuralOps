import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


class UPIMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        inner_dim = dim * expand
        self.in_proj = nn.Linear(dim, inner_dim * 2)
        self.conv1d = nn.Conv1d(
            inner_dim, inner_dim, d_conv, groups=inner_dim, padding=d_conv - 1
        )
        self.A_log = nn.Parameter(torch.log(torch.rand(dim, d_state) * 0.1))
        self.D = nn.Parameter(torch.ones(dim))
        self.dt_proj = nn.Linear(inner_dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_proj = nn.Linear(inner_dim, dim)

    def forward(self, x, **kwargs):
        B, L, D = x.shape
        x_proj = self.in_proj(x)
        x_ssm, z = x_proj.chunk(2, dim=-1)
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[..., :L].transpose(1, 2)
        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log)
        x_ssm_trunc = x_ssm[:, :, : self.dim]
        h = torch.zeros(B, self.dim, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            xt = x_ssm_trunc[:, t]
            h = h * torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0)) + (
                xt.unsqueeze(-1) * dt[:, t].unsqueeze(-1)
            )
            h = h - self.gamma * h
            out_t = (h * self.D.unsqueeze(-1)).sum(dim=-1)
            outputs.append(out_t)
        out_ssm = torch.stack(outputs, dim=1)
        out_ssm_full = F.pad(out_ssm, (0, x_ssm.size(-1) - self.dim))
        out = out_ssm_full * F.silu(z)
        return self.out_proj(out), None


@ATTENTION_REGISTRY.register("upi_stuffed_mamba")
def inject_upi_mamba(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    dim = model_config.hidden_size
    new_block = UPIMambaBlock(dim=dim)
    layer.attention = new_block
    return new_block
