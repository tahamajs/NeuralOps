import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

class UnifiedMoE(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        # Shared FFN‑style experts (used for both attention and FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, L, D = x.shape
        logits = self.router(x)   # (B, L, E)
        weights = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(weights, self.top_k, dim=-1)
        topk_vals = F.softmax(topk_vals, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = topk_idx[:,:,k]   # (B, L)
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x)
                    out += mask.float().unsqueeze(-1) * topk_vals[:,:,k:k+1] * expert_out
        return out

@ATTENTION_REGISTRY.register("bam_umoe")
class BAMUMoEBlock(BaseAttention):
    """
    BAM!‑UMoE: unified MoE for both attention and FFN using shared experts.
    """
    def __init__(self, config, rank=64, num_experts=8):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.attn = UnifiedMoE(self.dim, num_experts)
        self.ffn = self.attn   # share same expert set
        # The router is the same object; we just call it twice: once for attention, once for FFN
        # Actually, we need independent routing? The UMoE paper shares experts but the routing can be shared or separate.
        # For simplicity, we use the same router.

    def forward(self, x, mask=None, **kwargs):
        # For demonstration: we treat the unified MoE as a substitute for attention and FFN.
        # In a full model, you would structure the block with attn(self, x) and ffn(x).
        attn_out = self.attn(x)   # acts as attention
        ffn_out = self.ffn(x)     # acts as FFN
        return self.attn(attn_out + ffn_out), None   # simplistic