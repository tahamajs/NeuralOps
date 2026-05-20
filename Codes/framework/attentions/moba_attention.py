import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("moba")
class MoBAAttention(nn.Module):
    def __init__(self, config, block_size=64, top_k=4):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.block_size = block_size
        self.top_k = top_k
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        # Pad to block multiple
        pad = (self.block_size - L % self.block_size) % self.block_size
        if pad > 0:
            Q = F.pad(Q, (0, 0, 0, pad))
            K = F.pad(K, (0, 0, 0, pad))
            V = F.pad(V, (0, 0, 0, pad))
        Lp = L + pad
        nb = Lp // self.block_size

        Qb = Q.view(B, H, nb, self.block_size, d)
        Kb = K.view(B, H, nb, self.block_size, d)
        Vb = V.view(B, H, nb, self.block_size, d)

        # Block-level scores using mean
        Q_mean = Qb.mean(dim=3)
        K_mean = Kb.mean(dim=3)
        scale = d ** -0.5
        block_scores = torch.matmul(Q_mean, K_mean.transpose(-2, -1)) * scale  # (B, H, nb, nb)

        top_k = min(self.top_k, nb)
        _, top_idx = block_scores.topk(top_k, dim=-1)  # (B, H, nb, top_k)

        out_blocks = torch.zeros_like(Qb)
        for i in range(nb):
            # For each query block i, gather keys from top_k blocks
            idx = top_idx[:, :, i, :]  # (B, H, top_k)
            # Expand idx for gathering: (B, H, top_k) -> (B, H, top_k, 1, 1)
            idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.block_size, d)
            K_sel = torch.gather(Kb, 2, idx_exp)
            V_sel = torch.gather(Vb, 2, idx_exp)
            Q_i = Qb[:, :, i]  # (B, H, block_size, d)
            scores = torch.matmul(Q_i.unsqueeze(2), K_sel.transpose(-2, -1)) * scale  # (B, H, 1, block_size, top_k*block_size)
            scores = scores.squeeze(2)  # (B, H, block_size, top_k*block_size)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            # Compute context: (B, H, block_size, d)
            ctx = torch.matmul(attn.unsqueeze(2), V_sel).squeeze(2)
            out_blocks[:, :, i] = ctx

        out = out_blocks.view(B, H, Lp, d)[:, :, :L, :]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), None
    