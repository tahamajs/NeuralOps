import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def entmax15(scores, dim=-1, n_iter=10, eps=1e-8):
    p = F.softmax(scores, dim=dim)
    for _ in range(n_iter):
        q = p.sqrt()
        tau = torch.sum(q * p, dim=dim, keepdim=True)
        p_1 = q / tau
        if torch.max(torch.abs(p_1 - p)) < eps:
            break
        p = p_1
    return p


@ATTENTION_REGISTRY.register("adasplash")
class AdaSplashAttention(BaseAttention):
    def __init__(self, config, rank=None, alpha=1.5, n_iter=10):
        super().__init__(config, rank)
        self.alpha = alpha
        self.n_iter = n_iter
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / (Dh**0.5)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            logits = logits + (1.0 - mask) * -1e9

        if self.alpha == 1.5:
            attn = entmax15(logits, dim=-1, n_iter=self.n_iter)
        else:
            raise NotImplementedError
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), None
