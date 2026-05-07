import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("tpa")
class TPAttention(nn.Module):
    def __init__(self, config, rank=8):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.rank = rank
        self.dim = config.hidden_size

        # Factorized projections
        self.q_proj1 = nn.Linear(self.dim, self.num_heads * self.rank)
        self.q_proj2 = nn.Linear(self.dim, self.num_heads * self.rank)
        self.k_proj1 = nn.Linear(self.dim, self.num_heads * self.rank)
        self.k_proj2 = nn.Linear(self.dim, self.num_heads * self.rank)
        self.v_proj = nn.Linear(self.dim, self.num_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, r, hd = self.num_heads, self.rank, self.head_dim

        q1 = self.q_proj1(hidden_states).view(B, L, H, r)
        q2 = self.q_proj2(hidden_states).view(B, L, H, r)
        k1 = self.k_proj1(hidden_states).view(B, L, H, r)
        k2 = self.k_proj2(hidden_states).view(B, L, H, r)
        v = self.v_proj(hidden_states).view(B, L, H, hd)

        # Tensor product: Q = q1 ⊗ q2  (outer product) -> (B, L, H, r, r)
        Q = torch.einsum('blhi,blhj->blhij', q1, q2).reshape(B, L, H, r*r)
        K = torch.einsum('blhi,blhj->blhij', k1, k2).reshape(B, L, H, r*r)

        Q = Q.transpose(1, 2).transpose(2, 3)  # (B, H, r*r, L)
        K = K.transpose(1, 2).transpose(2, 3)  # (B, H, r*r, L)
        v = v.transpose(1, 2)  # (B, H, L, hd)

        scores = torch.matmul(Q.transpose(-2, -1), K) / (r ** 0.5)  # (B, H, L, L)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, L, hd)
        out = out.transpose(1, 2).contiguous().view(B, L, H * hd)
        out = self.out_proj(out)
        return out, None