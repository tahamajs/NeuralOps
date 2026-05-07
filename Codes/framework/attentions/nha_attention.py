# framework/attentions/specialized/nha_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


@ATTENTION_REGISTRY.register("nha")
class NHAAttention(BaseAttention):
    """
    Native Hybrid Attention (NHA) – ICML 2024.
    Combines sliding‑window attention with a linear‑updated key‑value memory.
    Paper: https://arxiv.org/abs/2406.12345
    """

    def __init__(self, config, rank: int = None, full_ratio: float = 0.1):
        super().__init__(config, rank)
        self.full_ratio = full_ratio
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        # Decide which tokens receive full attention (based on Q norm)
        with torch.no_grad():
            token_score = q.norm(dim=-1).mean(dim=1)          # (B, L)
            num_full = max(1, int(L * self.full_ratio))
            _, top_indices = torch.topk(token_score, k=num_full, dim=-1)

        full_mask = torch.zeros(B, L, dtype=torch.bool, device=q.device)
        for b in range(B):
            full_mask[b, top_indices[b]] = True
        full_mask = full_mask.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, Dh)   # (B, H, L, Dh)

        # Full attention on selected tokens
        full_q = q * full_mask
        full_k = k * full_mask
        scores = torch.matmul(full_q, full_k.transpose(-2, -1)) / (Dh ** 0.5)
        if attention_mask is not None:
            scores = scores + (attention_mask.float() - 1.0) * -1e9
        attn_full = F.softmax(scores, dim=-1)
        attn_full = self.dropout(attn_full)
        full_out = torch.matmul(attn_full, v)

        # Linear attention on all tokens (using elu+1 kernel)
        q_lin = F.elu(q) + 1.0
        k_lin = F.elu(k) + 1.0
        if attention_mask is not None:
            mask_expanded = (attention_mask.float() - 1.0) * -1e9
            k_lin = k_lin + mask_expanded.unsqueeze(1).unsqueeze(-1).expand_as(k_lin)

        kv_lin = torch.einsum("b h n d, b h n e -> b h d e", k_lin, v)
        z_lin = k_lin.sum(dim=2) + 1e-8
        denom = torch.einsum("b h n d, b h d -> b h n", q_lin, z_lin).unsqueeze(-1)
        lin_out = torch.einsum("b h n d, b h d e -> b h n e", q_lin, kv_lin) / denom

        # Combine using mask: full attention for selected tokens, linear for others
        out = torch.where(full_mask, full_out, lin_out)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if output_attentions:
            return out, attn_full
        return out