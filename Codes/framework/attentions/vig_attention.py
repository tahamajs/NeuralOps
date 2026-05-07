# framework/attentions/specialized/vig_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


@ATTENTION_REGISTRY.register("vig")
class ViGAttention(BaseAttention):
    """
    ViG: Linear‑complexity Visual Sequence Learning with Gated Linear Attention.
    Paper: https://arxiv.org/abs/2405.18482
    """

    def __init__(self, config, rank: int = None):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

        # Gate network: produces two scalars per head (row gate, col gate)
        self.gate_net = nn.Linear(self.dim, 2 * self.num_heads)

    def forward(self, x, H=None, W=None, **kwargs):
        B, N, D = x.shape
        # Infer H and W if not given (assume square)
        if H is None:
            H = int(N ** 0.5)
            W = H if H * H == N else N // H
        Hd = self.num_heads
        Dh = self.head_dim

        q = self.q_proj(x).view(B, N, Hd, Dh).transpose(1, 2)   # (B, Hd, N, Dh)
        k = self.k_proj(x).view(B, N, Hd, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, Hd, Dh).transpose(1, 2)

        gates = torch.sigmoid(self.gate_net(x)).view(B, N, 2, Hd, 1)

        # Reshape to spatial grid
        q = q.reshape(B, Hd, H, W, Dh).permute(0, 1, 3, 2, 4)   # rows: (B, Hd, W, H, Dh)
        k = k.reshape(B, Hd, H, W, Dh).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, Hd, H, W, Dh).permute(0, 1, 3, 2, 4)

        # Row‑wise linear attention (treat each row as sequence)
        q_row = q.contiguous().view(B, Hd * W, H, Dh).transpose(2, 3)   # (B, Hd*W, H, Dh) -> (B, Hd*W, Dh, H)
        k_row = k.contiguous().view(B, Hd * W, H, Dh).transpose(2, 3)
        v_row = v.contiguous().view(B, Hd * W, H, Dh).transpose(2, 3)

        q_row_lin = F.elu(q_row) + 1.0
        k_row_lin = F.elu(k_row) + 1.0
        kv_row = torch.einsum("b m n d, b m n e -> b m d e", k_row_lin, v_row)
        z_row = k_row_lin.sum(dim=2) + 1e-8
        denom_row = torch.einsum("b m n d, b m d -> b m n", q_row_lin, z_row).unsqueeze(-1)
        out_row = torch.einsum("b m n d, b m d e -> b m n e", q_row_lin, kv_row) / denom_row
        out_row = out_row.transpose(2, 3).view(B, Hd, W, H, Dh).permute(0, 1, 3, 2, 4)

        # Column‑wise linear attention
        q_col = q.permute(0, 1, 2, 4, 3)   # (B, Hd, H, W, Dh) -> (B, Hd, H, Dh, W)
        k_col = k.permute(0, 1, 2, 4, 3)
        v_col = v.permute(0, 1, 2, 4, 3)

        q_col_lin = F.elu(q_col) + 1.0
        k_col_lin = F.elu(k_col) + 1.0
        kv_col = torch.einsum("b h m n d, b h m n e -> b h m d e", k_col_lin, v_col)
        z_col = k_col_lin.sum(dim=3) + 1e-8
        denom_col = torch.einsum("b h m n d, b h m d -> b h m n", q_col_lin, z_col).unsqueeze(-1)
        out_col = torch.einsum("b h m n d, b h m d e -> b h m n e", q_col_lin, kv_col) / denom_col
        out_col = out_col.permute(0, 1, 2, 4, 3)

        out_col = out_col.reshape(B, Hd, H, W, Dh)

        # Combine using learned gates
        gate_row = gates[:, :, 0:1].transpose(1, 2).reshape(B, Hd, H, W, 1)
        gate_col = gates[:, :, 1:2].transpose(1, 2).reshape(B, Hd, H, W, 1)

        out = gate_row * out_row + gate_col * out_col
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, N, D)
        return self.out_proj(out), None