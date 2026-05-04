# framework/attentions/specialized/linformer_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("linformer")
class LinformerAttention(BaseAttention):
    """
    Linformer: Linear‑complexity self‑attention.
    Paper: https://arxiv.org/abs/2006.04768
    """

    def __init__(self, config, rank: int = None, k_dim: int = 128):
        super().__init__(config, rank)
        self.k_dim = k_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Learnable projection matrices E and F (k_dim × max_pos)
        # For simplicity, we use a large max_len; during forward we truncate to L.
        max_len = getattr(config, "max_position_embeddings", 512)
        self.E = nn.Parameter(torch.randn(k_dim, max_len) * 0.02)
        self.F = nn.Parameter(torch.randn(k_dim, max_len) * 0.02)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        # Apply projections to reduce sequence length
        E_eff = self.E[:, :L]   # (k_dim, L)
        k_proj = torch.einsum("kl,bhl d->bhk d", E_eff, k)   # (B, H, k_dim, Dh)
        v_proj = torch.einsum("kl,bhl d->bhk d", E_eff, v)

        scores = torch.matmul(q, k_proj.transpose(-2, -1)) / (Dh ** 0.5)
        if attention_mask is not None:
            # Mask is applied only on the query side (projected keys are already compressed)
            scores = scores + (attention_mask.float() - 1.0) * -1e9[:, :, None, None]

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_proj)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if output_attentions:
            return out, attn
        return out