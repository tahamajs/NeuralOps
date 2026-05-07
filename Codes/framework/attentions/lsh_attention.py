import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("lsh")
class LSHAttention(nn.Module):
    def __init__(self, config, rounds=4, bucket_size=64):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.rounds = rounds
        self.bucket_size = bucket_size
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def _hash(self, vectors, proj):
        norm = F.normalize(vectors, dim=-1)
        dots = torch.matmul(norm, proj)
        bits = (dots > 0).to(torch.long)   # define bits here
        multipliers = (2 ** torch.arange(self.bucket_size, device=vectors.device)).to(torch.long)
        hash_val = torch.sum(bits * multipliers, dim=-1)
        return hash_val
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H = self.num_heads
        d = self.head_dim

        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        out = torch.zeros_like(V)
        scale = d ** -0.5

        for r in range(self.rounds):
            proj = torch.randn(d, self.bucket_size, device=Q.device)
            proj = F.normalize(proj, dim=0)
            q_hash = self._hash(Q, proj)  # (B, H, N)
            k_hash = self._hash(K, proj)

            # For each unique hash code, attend
            for b in range(B):
                for h in range(H):
                    qh = q_hash[b, h]
                    kh = k_hash[b, h]
                    unique_codes = torch.unique(torch.cat([qh, kh]))
                    for code in unique_codes:
                        q_mask = (qh == code)
                        k_mask = (kh == code)
                        if q_mask.sum() == 0 or k_mask.sum() == 0:
                            continue
                        Qs = Q[b, h][q_mask]  # (Nq, d)
                        Ks = K[b, h][k_mask]  # (Nk, d)
                        Vs = V[b, h][k_mask]  # (Nk, d)
                        scores = (Qs @ Ks.T) * scale
                        attn = F.softmax(scores, dim=-1)
                        attn = self.dropout(attn)
                        ctx = attn @ Vs
                        out[b, h][q_mask] += ctx

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), None