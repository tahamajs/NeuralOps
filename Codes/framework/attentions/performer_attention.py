# framework/attentions/specialized/performer_attention.py
"""
Performer: Fast Attention Via Positive Orthogonal Random features (FAVOR+)
Paper: https://arxiv.org/abs/2009.14794
"""

import math

import torch
import torch.nn as nn
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("performer")
class PerformerAttention(BaseAttention):
    """
    Performer attention with orthogonal random features and proper masking.
    Uses the exponential feature map: phi(x) = exp( (x·R - ||x||²/2) )/√r.
    """

    def __init__(
        self,
        config,
        rank: int = 256,
        redraw_projection: bool = False,
        pretrained_attn: nn.Module = None,
    ):
        super().__init__(config, rank)
        self.rank = rank
        self.redraw_projection = redraw_projection
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Random projection matrix buffer (one per head)
        self.register_buffer("projection", None)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src):
        with torch.no_grad():
            self.query.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.query.bias.copy_(src.q_lin.bias)
            self.key.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.key.bias.copy_(src.k_lin.bias)
            self.value.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.value.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def _reshape(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _get_projection_matrix(self, device):
        """Create per‑head orthogonal random features (FAVOR+)."""
        if self.projection is None or self.redraw_projection:
            proj = []
            for _ in range(self.num_heads):
                ortho = torch.randn(self.head_dim, self.rank, device=device)
                q, _ = torch.linalg.qr(ortho)
                proj.append(q)
            self.projection = torch.stack(proj, dim=0)  # (H, D, R)
        return self.projection

    def _feature_map(self, x, proj):
        """
        x: (B, H, N, D)
        proj: (H, D, R)
        Returns: (B, H, N, R)
        """
        # 1. Compute x·R
        x_proj = torch.einsum("bhnd,hdr->bhnr", x, proj)  # (B, H, N, R)
        # 2. Compute squared norm of x (L2 norm² / 2)
        x_norm = (x * x).sum(dim=-1, keepdim=True) / 2.0  # (B, H, N, 1)
        # 3. Exponential map
        phi = torch.exp(x_proj - x_norm) / math.sqrt(self.rank)
        return phi

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        proj = self._get_projection_matrix(Q.device)

        Q_feat = self._feature_map(Q, proj)  # (B, H, N, R)
        K_feat = self._feature_map(K, proj)  # (B, H, N, R)

        # Mask padding tokens in K and V (so they contribute nothing)
        if attention_mask is not None:
            # Convert to bool/float mask
            if attention_mask.dim() == 2:
                valid = attention_mask.bool().to(Q.device)  # (B, N)
            else:
                valid = attention_mask.squeeze(1).squeeze(1).bool()
            valid_expanded = valid.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            K_feat = K_feat * valid_expanded.to(K_feat.dtype)
            V = V * valid_expanded.to(V.dtype)

        # Linear attention
        KV = torch.einsum("bhnr,bhnd->bhrd", K_feat, V)  # (B, H, R, D)
        K_sum = K_feat.sum(dim=2, keepdim=True)  # (B, H, 1, R)
        num = torch.einsum("bhnr,bhrd->bhnd", Q_feat, KV)
        den = (
            torch.einsum("bhnr,bhr->bhn", Q_feat, K_sum.squeeze(2)).unsqueeze(-1) + 1e-8
        )
        context = num / den

        # Optional: zero out padding positions in output
        if attention_mask is not None:
            valid_out = valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)
            context = context * valid_out

        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        out = self.out_proj(context)

        if output_attentions:
            return out, None  # No materialised attention weights
        return out, None


def inject_performer(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    rank = getattr(comp_config, "rank", 256)
    redraw = getattr(comp_config, "redraw_projection", False)
    new_attn = PerformerAttention(
        config=model_config,
        rank=rank,
        redraw_projection=redraw,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
