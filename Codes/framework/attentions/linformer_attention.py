# framework/attentions/specialized/linformer_attention.py
"""
Linformer: Linear‑complexity self‑attention via sequence‑length projection.
Paper: https://arxiv.org/abs/2006.04768
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("linformer")
class LinformerAttention(BaseAttention):
    """
    Linformer Attention with learnable projection matrices E and F.
    Compresses the key and value length dimension to a fixed size `k_dim`.
    """

    def __init__(
        self,
        config,
        rank: int = None,
        k_dim: int = 128,
        pretrained_attn: nn.Module = None,
    ):
        super().__init__(config, rank)
        self.k_dim = k_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        # Linear projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Learnable projection matrices (k_dim, max_len)
        max_len = getattr(config, "max_position_embeddings", 512)
        self.E = nn.Parameter(torch.randn(k_dim, max_len) * 0.02)
        self.F = nn.Parameter(torch.randn(k_dim, max_len) * 0.02)

        # Zero‑shot weight transfer
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src):
        """Copy pretrained Q/K/V/O weights and biases."""
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
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = (
            self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        )  # (B, H, L, Dh)
        k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        # Truncate projection matrices to actual length
        E_eff = self.E[:, :L]  # (k_dim, L)
        F_eff = self.F[:, :L]  # (k_dim, L) – use separate F for values
        k_proj = torch.einsum("kl,bhld->bhkd", E_eff, k)  # (B, H, k_dim, Dh)
        v_proj = torch.einsum("kl,bhld->bhkd", F_eff, v)  # (B, H, k_dim, Dh)

        # Scaled dot‑product attention
        scale = Dh**-0.5
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) * scale  # (B, H, L, k_dim)

        # Mask handling: convert typical 2D binary mask to additive form
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, L) -> (B, 1, 1, L)
                mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
                scores = scores + (1.0 - mask) * -1e9
            else:
                # Already broadcastable – assume additive (0 = keep, -inf = mask)
                scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_proj)  # (B, H, L, Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if output_attentions:
            return out, attn
        return out, None


def inject_linformer(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    """Injection function for DistilBERT."""
    k_dim = getattr(comp_config, "k_dim", 128)
    new_attn = LinformerAttention(
        config=model_config, k_dim=k_dim, pretrained_attn=old_attn
    )
    layer.attention = new_attn
    return new_attn
