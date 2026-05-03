import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY


class UnifiedMoE(nn.Module):
    """Mixture of Experts with top‑k gating. Can be used for any projection."""
    def __init__(self, dim, num_experts=8, top_k=2, output_dim=None):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        out_dim = output_dim or dim
        self.out_dim = out_dim

        self.router = nn.Linear(dim, num_experts, bias=False)
        # Each expert is an FFN‑style block
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, out_dim * 2),
                nn.GELU(),
                nn.Linear(out_dim * 2, out_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, L, D = x.shape
        logits = self.router(x)                     # (B, L, E)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)   # (B, L, k)
        topk_vals = F.softmax(topk_vals, dim=-1)    # normalise the selected weights

        out = torch.zeros(B, L, self.out_dim, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            expert_idx = topk_idx[..., k]           # (B, L)
            scale = topk_vals[..., k:k+1]           # (B, L, 1)
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x)  # (B, L, out_dim)
                    out = out + mask.float().unsqueeze(-1) * scale * expert_out
        return out


@ATTENTION_REGISTRY.register("bam_umoe")
class BAMUMoEBlock(BaseAttention):
    """
    BAM!‑UMoE: Unified Mixture of Experts for Attention and FFN.
    - Shared experts produce Q, K, V projections (via an MoE that outputs 3*dim).
    - Standard multi‑head attention is computed on the resulting Q, K, V.
    - Another shared MoE (possibly a separate instance) acts as the feed‑forward network.
    """
    def __init__(self, config, rank=64, num_experts=8, top_k=2):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim

        # MoE that outputs Q, K, V concatenated (3*all_head_size)
        self.qkv_moe = UnifiedMoE(self.dim, num_experts, top_k, output_dim=3 * self.all_head_size)
        # MoE for the feed‑forward network (output same dim)
        self.ffn_moe = UnifiedMoE(self.dim, num_experts, top_k, output_dim=self.dim)

        self.output_proj = nn.Linear(self.all_head_size, self.dim)  # final projection
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # 1. Attention branch
        qkv = self.qkv_moe(x)                     # (B, L, 3*all_head_size)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, H, d).transpose(1, 2)    # (B, H, L, d)
        k = k.view(B, L, H, d).transpose(1, 2)
        v = v.view(B, L, H, d).transpose(1, 2)

        scale = d ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, L, L)

        if mask is not None:
            # Assume mask is (B, L) with 1 for keep, 0 for pad
            if mask.dim() == 2:
                mask = mask[:, None, None, :].to(dtype=attn_scores.dtype)
                attn_scores = attn_scores + (1.0 - mask) * -1e9
            else:
                attn_scores = attn_scores + mask.unsqueeze(1)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)    # (B, H, L, d)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)  # (B, L, D)
        attn_out = self.output_proj(context)

        # Residual + norm
        x = self.norm1(x + attn_out)

        # 2. FFN branch (unified MoE)
        ffn_out = self.ffn_moe(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights