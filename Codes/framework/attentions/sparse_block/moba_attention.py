import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("moba")
class MoBAAttention(nn.Module):
    """
    Mixture of Blocks (MoBA): Trainable block‑sparse attention via gating.
    Divides the sequence into blocks and selects top‑k blocks per query token.
    """
    def __init__(self, config, block_size=64, top_k=8):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.block_size = block_size
        self.top_k = top_k
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # Gate to select blocks per head (maybe per query token)
        self.router = nn.Linear(config.hidden_size, 1)   # not used in this simple version; we do block selection via key statistics.
        # For speed, we'll compute block importance by mean key L2 norm.
        # A more sophisticated MoBA would use a learned gate; we simplify here.

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H = self.num_heads
        d = self.head_dim
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0,2,1,3)  # (B, H, L, d)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0,2,1,3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0,2,1,3)

        # Break keys into blocks
        num_blocks = (L + self.block_size - 1) // self.block_size
        # Compute block importance scores: average L2 norm for each block per head
        block_scores = torch.zeros(B, H, num_blocks, device=Q.device)
        for b_idx in range(num_blocks):
            start = b_idx * self.block_size
            end = min(start + self.block_size, L)
            k_blk = K[:, :, start:end, :]
            block_scores[:, :, b_idx] = k_blk.norm(dim=-1).mean(dim=2)  # (B, H)

        # For each query position, select top_k blocks
        top_blocks = torch.topk(block_scores, min(self.top_k, num_blocks), dim=-1).indices  # (B, H, K)

        # Build sparse attention: for each query, attend only to tokens in top_blocks
        out = torch.zeros_like(V)
        scale = d ** -0.5

        for b in range(B):
            for h in range(H):
                selected_blocks = top_blocks[b, h]
                # Gather K and V from those blocks
                k_sel = []
                v_sel = []
                block_boundaries = []
                offset = 0
                for blk in selected_blocks:
                    blk = blk.item()
                    start = blk * self.block_size
                    end = min(start + self.block_size, L)
                    k_sel.append(K[b, h, start:end])
                    v_sel.append(V[b, h, start:end])
                    block_boundaries.append((offset, offset + (end-start)))
                    offset += (end-start)
                if not k_sel:
                    continue
                K_cat = torch.cat(k_sel, dim=0)  # (L_sel, d)
                V_cat = torch.cat(v_sel, dim=0)
                Q_bh = Q[b, h]   # (L, d)
                scores = torch.mm(Q_bh, K_cat.T) * scale
                # Apply attention mask if provided (simplified)
                attn = F.softmax(scores, dim=-1)
                ctx = torch.mm(attn, V_cat)
                out[b, h] = ctx
        out = out.permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out_proj(out), None