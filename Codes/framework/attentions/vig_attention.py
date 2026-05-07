import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("vig")
class ViGAttention(BaseAttention):
    def __init__(self, config, rank=None):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.gate_net = nn.Linear(self.dim, 2 * self.num_heads)

    def forward(self, x, **kwargs):
        B, N, D = x.shape
        # Find H and W such that H*W >= N
        H = int(N ** 0.5)
        W = (N + H - 1) // H
        total = H * W
        pad_len = total - N
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(B, pad_len, D, device=x.device)], dim=1)
            N_pad = total
        else:
            N_pad = N

        Hd = self.num_heads
        Dh = self.head_dim

        q = self.q_proj(x).view(B, N_pad, Hd, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, N_pad, Hd, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N_pad, Hd, Dh).transpose(1, 2)

        gates = torch.sigmoid(self.gate_net(x)).view(B, N_pad, 2, Hd, 1)

        # Reshape to grid
        q = q.reshape(B, Hd, H, W, Dh).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, Hd, H, W, Dh).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, Hd, H, W, Dh).permute(0, 1, 3, 2, 4)

        # Row attention
        q_row = q.reshape(B, Hd * W, H, Dh).transpose(2, 3)
        k_row = k.reshape(B, Hd * W, H, Dh).transpose(2, 3)
        v_row = v.reshape(B, Hd * W, H, Dh).transpose(2, 3)

        q_row_lin = F.elu(q_row) + 1.0
        k_row_lin = F.elu(k_row) + 1.0
        kv_row = torch.einsum("b m n d, b m n e -> b m d e", k_row_lin, v_row)
        z_row = k_row_lin.sum(dim=2, keepdim=True)
        denom_row = torch.einsum("b m n d, b m d -> b m n", q_row_lin, z_row.squeeze(2)).unsqueeze(-1)
        out_row = torch.einsum("b m n d, b m d e -> b m n e", q_row_lin, kv_row) / (denom_row + 1e-8)
        out_row = out_row.transpose(2, 3).reshape(B, Hd, W, H, Dh).permute(0, 1, 3, 2, 4)

        # Column attention
        q_col = q.permute(0, 1, 2, 4, 3)
        k_col = k.permute(0, 1, 2, 4, 3)
        v_col = v.permute(0, 1, 2, 4, 3)
        q_col_flat = q_col.reshape(B * Hd * H, W, Dh)
        k_col_flat = k_col.reshape(B * Hd * H, W, Dh)
        v_col_flat = v_col.reshape(B * Hd * H, W, Dh)
        q_col_lin = F.elu(q_col_flat) + 1.0
        k_col_lin = F.elu(k_col_flat) + 1.0
        kv_col = torch.einsum("b n d, b n e -> b d e", k_col_lin, v_col_flat)
        z_col = k_col_lin.sum(dim=1, keepdim=True)
        denom_col = torch.einsum("b n d, b d -> b n", q_col_lin, z_col.squeeze(1)).unsqueeze(-1)
        out_col_flat = torch.einsum("b n d, b d e -> b n e", q_col_lin, kv_col) / (denom_col + 1e-8)
        out_col = out_col_flat.view(B, Hd, H, W, Dh).permute(0, 1, 3, 2, 4)

        # Gates
        gate_row = gates[:, :, 0:1].transpose(1, 2).reshape(B, Hd, H, W, 1)
        gate_col = gates[:, :, 1:2].transpose(1, 2).reshape(B, Hd, H, W, 1)
        out = gate_row * out_row + gate_col * out_col
        out = out.permute(0, 2, 3, 1, 4).reshape(B, total, D)
        if pad_len > 0:
            out = out[:, :N, :]
        return self.out_proj(out), None