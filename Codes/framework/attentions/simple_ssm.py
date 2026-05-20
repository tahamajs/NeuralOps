import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("simple_ssm")
class SimpleSSM(nn.Module):
    """
    Simplified SSM block with correct gating and output projection.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        inner_dim = d_model * expand

        self.in_proj = nn.Linear(d_model, inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(inner_dim, inner_dim, kernel_size=d_conv,
                                groups=inner_dim, padding=d_conv - 1)
        self.dt_proj = nn.Linear(inner_dim, d_model, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.act = nn.SiLU()
        # Output projection from d_model to d_model (after gating)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        inner_dim = d_model * (self.in_proj.out_features // (2 * d_model))

        x_proj = self.in_proj(x)
        x_ssm, z = x_proj.chunk(2, dim=-1)

        # Mask padding
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = attention_mask.bool().to(x.device)
                valid_expanded = valid.unsqueeze(-1).to(x.dtype)
                x_ssm = x_ssm * valid_expanded
                z = z * valid_expanded

        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[..., :seq_len]
        x_ssm = x_ssm.transpose(1, 2)

        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log)

        h = torch.zeros(batch, d_model, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            xt = x_ssm[:, t, :]          # (B, inner)
            xt_slice = xt[:, :d_model]   # (B, d_model)
            h = h * torch.exp(dt[:, t].unsqueeze(-1) * A) + (xt_slice.unsqueeze(-1) * dt[:, t].unsqueeze(-1))
            out_t = (h * self.D.unsqueeze(-1)).sum(dim=-1)
            outputs.append(out_t)
        out_ssm = torch.stack(outputs, dim=1)   # (B, L, d_model)

        # Gate with first d_model channels of z
        gate = self.act(z[:, :, :d_model])
        out = out_ssm * gate
        out = self.out_proj(out)   # (B, L, d_model)
        return out