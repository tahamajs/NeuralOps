"""
ViG Attention: Vision GNN‑inspired Grid Linear Attention with Gated Fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("vig")
class ViGAttention(BaseAttention):
    """
    Grid‑based linear attention: splits sequence into H×W grid, applies
    kernelised linear attention along rows and columns, and fuses them with
    learned gates.
    """

    def __init__(self, config, rank=None, pretrained_attn=None):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.gate_net = nn.Linear(self.dim, 2 * self.num_heads)
        self.dropout = nn.Dropout(
            config.dropout
        )  # applied after softmax? Not directly used in lin attn,
        # but we keep for consistency.

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src):
        with torch.no_grad():
            self.q_proj.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.q_proj.bias.copy_(src.q_lin.bias)
            self.k_proj.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.k_proj.bias.copy_(src.k_lin.bias)
            self.v_proj.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.v_proj.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        B, N, D = hidden_states.shape

        # Build grid
        H = int(N**0.5)
        W = (N + H - 1) // H
        total = H * W
        pad_len = total - N

        if pad_len > 0:
            # Pad with zeros – these will not contribute because we will mask them later
            x = torch.cat(
                [
                    hidden_states,
                    torch.zeros(
                        B,
                        pad_len,
                        D,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ),
                ],
                dim=1,
            )
            N_pad = total
        else:
            x = hidden_states
            N_pad = N

        # Create a validity mask for the grid (True = valid token)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = attention_mask.bool().to(x.device)  # (B, N)
                if pad_len > 0:
                    valid = torch.cat(
                        [
                            valid,
                            torch.zeros(B, pad_len, dtype=torch.bool, device=x.device),
                        ],
                        dim=1,
                    )
            else:
                # Assume already broadcastable bool
                valid = attention_mask.bool()
        else:
            valid = torch.ones(B, N_pad, dtype=torch.bool, device=x.device)

        Hh = self.num_heads
        Dh = self.head_dim

        q = (
            self.q_proj(x).view(B, N_pad, Hh, Dh).permute(0, 2, 1, 3)
        )  # (B, Hh, N_pad, Dh)
        k = self.k_proj(x).view(B, N_pad, Hh, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, N_pad, Hh, Dh).permute(0, 2, 1, 3)

        gates = torch.sigmoid(self.gate_net(x)).view(
            B, N_pad, 2, Hh, 1
        )  # (B, N_pad, 2, Hh, 1)

        # Reshape to grid
        q = q.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)  # (B, Hh, W, H, Dh)
        k = k.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)

        # Row attention (along height)
        q_row = q.reshape(B, Hh * W, H, Dh).transpose(2, 3)  # (B, Hh*W, Dh, H)
        k_row = k.reshape(B, Hh * W, H, Dh).transpose(2, 3)
        v_row = v.reshape(B, Hh * W, H, Dh).transpose(2, 3)

        # Apply ELU+1 kernel
        q_row_lin = F.elu(q_row) + 1.0
        k_row_lin = F.elu(k_row) + 1.0

        # Mask invalid tokens (zero them out in k and v)
        # valid shape (B, N_pad) -> reshape to grid and select axis
        valid_grid = valid.view(B, H, W).permute(0, 2, 1)  # (B, W, H)
        valid_row = (
            valid_grid.reshape(B, Hh * W, H).unsqueeze(-2).to(k_row.dtype)
        )  # (B, Hh*W, 1, H)
        k_row_lin = k_row_lin * valid_row
        v_row = v_row * valid_row

        kv_row = torch.einsum("b m n d, b m n e -> b m d e", k_row_lin, v_row)
        z_row = k_row_lin.sum(dim=2, keepdim=True)
        denom_row = torch.einsum(
            "b m n d, b m d -> b m n", q_row_lin, z_row.squeeze(2)
        ).unsqueeze(-1)
        out_row = torch.einsum("b m n d, b m d e -> b m n e", q_row_lin, kv_row) / (
            denom_row + 1e-8
        )
        out_row = (
            out_row.transpose(2, 3).reshape(B, Hh, W, H, Dh).permute(0, 1, 3, 2, 4)
        )  # (B, Hh, H, W, Dh)

        # Column attention (along width)
        q_col = q.permute(0, 1, 2, 4, 3)  # (B, Hh, H, Dh, W)
        k_col = k.permute(0, 1, 2, 4, 3)
        v_col = v.permute(0, 1, 2, 4, 3)
        q_col_flat = q_col.reshape(B * Hh * H, W, Dh)
        k_col_flat = k_col.reshape(B * Hh * H, W, Dh)
        v_col_flat = v_col.reshape(B * Hh * H, W, Dh)

        q_col_lin = F.elu(q_col_flat) + 1.0
        k_col_lin = F.elu(k_col_flat) + 1.0

        valid_col = (
            valid.reshape(B, H, W)
            .permute(0, 2, 1)
            .reshape(B * Hh * H, W)
            .unsqueeze(-1)
            .to(k_col_lin.dtype)
        )
        k_col_lin = k_col_lin * valid_col
        v_col_flat = v_col_flat * valid_col

        kv_col = torch.einsum("b n d, b n e -> b d e", k_col_lin, v_col_flat)
        z_col = k_col_lin.sum(dim=1, keepdim=True)
        denom_col = torch.einsum(
            "b n d, b d -> b n", q_col_lin, z_col.squeeze(1)
        ).unsqueeze(-1)
        out_col_flat = torch.einsum("b n d, b d e -> b n e", q_col_lin, kv_col) / (
            denom_col + 1e-8
        )
        out_col = out_col_flat.view(B, Hh, H, W, Dh).permute(
            0, 1, 3, 2, 4
        )  # (B, Hh, W, H, Dh) -> later to (B, Hh, H, W, Dh)

        # Gates
        gate_row = (
            gates[:, :, 0:1].permute(0, 3, 1, 2).reshape(B, Hh, H, W, 1)
        )  # (B, Hh, H, W, 1)
        gate_col = gates[:, :, 1:2].permute(0, 3, 1, 2).reshape(B, Hh, H, W, 1)

        out = gate_row * out_row + gate_col * out_col  # (B, Hh, H, W, Dh)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, total, D)

        if pad_len > 0:
            out = out[:, :N, :]

        out = self.out_proj(out)

        if output_attentions:
            return out, None  # linear attention does not materialise weights
        return out, None


def inject_vig(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    new_attn = ViGAttention(config=model_config, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
