import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class UnifiedMoE(nn.Module):
    """
    Mixture of Experts with top-k gating.
    Vectorized version (no slow python loops).
    """

    def __init__(self, dim, num_experts=8, top_k=2, output_dim=None):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        out_dim = output_dim or dim
        self.out_dim = out_dim

        self.router = nn.Linear(dim, num_experts, bias=False)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, out_dim * 2),
                    nn.GELU(),
                    nn.Linear(out_dim * 2, out_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        B, L, D = x.shape

        # Router → top‑k expert scores
        logits = self.router(x)  # (B, L, E)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # (B, L, k)

        # Softmax over selected experts
        scores = F.softmax(topk_vals, dim=-1)  # (B, L, k)

        # Compute all expert outputs in parallel
        # experts_out[e] = output of expert e
        experts_out = torch.stack(
            [exp(x) for exp in self.experts], dim=2
        )  # (B, L, E, out_dim)

        # Gather expert outputs for top‑k selections
        topk_out = torch.gather(
            experts_out,
            2,
            topk_idx.unsqueeze(-1).expand(B, L, self.top_k, self.out_dim),
        )  # (B, L, k, out_dim)

        # Weighted sum of top‑k
        out = torch.sum(topk_out * scores.unsqueeze(-1), dim=2)

        return out


@ATTENTION_REGISTRY.register("bam_umoe")
class BAMUMoEBlock(BaseAttention):
    def __init__(
        self,
        config,
        rank=64,
        num_experts=8,
        top_k=2,
        layer_id=None,
        num_layers=None,
        old_attn=None,
        comp_config=None,
        **kwargs
    ):
        super().__init__(config, rank)

        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.qkv_moe = UnifiedMoE(
            self.dim, num_experts, top_k, output_dim=3 * self.all_head_size
        )
        self.ffn_moe = UnifiedMoE(self.dim, num_experts, top_k, output_dim=self.dim)

        self.output_proj = nn.Linear(self.all_head_size, self.dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(config.dropout)

        # store for future if needed
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.old_attn = old_attn
        self.comp_config = comp_config

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        qkv = self.qkv_moe(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        proj = self.output_proj(v)
        h1 = self.norm1(hidden_states + self.dropout(proj))
        ffn = self.ffn_moe(h1)
        out = self.norm2(h1 + self.dropout(ffn))
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(-1)
            elif attention_mask.dim() == 4:
                mask = (attention_mask > -1).float().squeeze(1).squeeze(1).unsqueeze(-1)
            else:
                mask = attention_mask
            out = out * mask
        return out, None

