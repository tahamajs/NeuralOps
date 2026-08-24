"""
BAM-UMoE: Unified Mixture of Experts Attention Block.

This module implements a Unified Mixture of Experts (UMoE) transformer layer with top-k routing
applied to multi-head projections and feed-forward transformation networks, with auxiliary load-balancing
loss and pretrained weight transfer.
"""

from typing import Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class UnifiedMoE(nn.Module):
    r"""
    Vectorized Mixture of Experts (MoE) layer with top-k gating and auxiliary load-balancing loss.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        output_dim: Optional[int] = None,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.num_experts: int = num_experts
        self.top_k: int = min(top_k, num_experts)
        self.out_dim: int = output_dim or dim
        self.noise_std: float = noise_std
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        # Router initialized with small uniform weights for balanced expert participation
        self.router_ln = nn.LayerNorm(dim if "dim" in locals() else (d_model if "d_model" in locals() else self.hidden_size))
        self.router = nn.Linear(dim, num_experts, bias=False)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.out_dim * 2),
                nn.GELU(),
                nn.Linear(self.out_dim * 2, self.out_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward execution pass for top-k Unified MoE with load balancing."""
        B, L, D = x.shape
        logits = self.router(self.router_ln(x))  # (B, L, E)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            routing_logits = logits + noise
        else:
            routing_logits = logits

        topk_vals, topk_idx = torch.topk(routing_logits, self.top_k, dim=-1)  # (B, L, k)
        scores = F.softmax(topk_vals, dim=-1)  # (B, L, k)

        # Auxiliary load balancing loss (Switch Transformer formulation)
        router_probs = F.softmax(logits / 1.0, dim=-1).view(-1, self.num_experts)  # (B*L, E)
        expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).float()  # (B, L, k, E)
        density = expert_mask.sum(dim=2).view(-1, self.num_experts).mean(dim=0)  # (E,)
        prob_mean = router_probs.mean(dim=0)  # (E,)
        self.aux_loss = self.num_experts * torch.sum(density * prob_mean)

        # Expert computation
        experts_out = torch.stack([exp(x) for exp in self.experts], dim=2)  # (B, L, E, out_dim)

        topk_out = torch.gather(
            experts_out,
            2,
            topk_idx.unsqueeze(-1).expand(B, L, self.top_k, self.out_dim),
        )  # (B, L, k, out_dim)

        out = torch.sum(topk_out * scores.unsqueeze(-1), dim=2)
        return out


@ATTENTION_REGISTRY.register("bam_umoe")
class BAMUMoEBlock(BaseAttention):
    r"""
    BAM Unified Mixture of Experts Attention Block.

    Replaces standard linear attention projections and feed-forward networks with
    parameter-efficient top-k conditional computation blocks with full self-attention
    and pretrained weight copying.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 64,
        num_experts: int = 8,
        top_k: int = 2,
        layer_id: Optional[int] = None,
        num_layers: Optional[int] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        self.dim: int = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads: int = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.head_dim: int = self.dim // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        # Pretrained Multi-Head Attention Projections
        self.q_lin = nn.Linear(self.dim, self.all_head_size)
        self.k_lin = nn.Linear(self.dim, self.all_head_size)
        self.v_lin = nn.Linear(self.dim, self.all_head_size)
        self.out_lin = nn.Linear(self.all_head_size, self.dim)

        # MoE Modulation Branch
        self.qkv_moe = UnifiedMoE(
            self.dim, num_experts, top_k, output_dim=self.all_head_size
        )
        self.ffn_moe = UnifiedMoE(self.dim, num_experts, top_k, output_dim=self.dim)

        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        self.layer_id: Optional[int] = layer_id
        self.num_layers: Optional[int] = num_layers
        self.comp_config: Optional[Any] = comp_config
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        if old_attn is not None:
            self.copy_weights(old_attn)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        r"""Copy weights from standard pretrained attention modules."""
        with torch.no_grad():
            q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", None)))
            k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", None)))
            v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", None)))
            out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o_proj", None))))

            if q_src is not None and hasattr(q_src, "weight"):
                self.q_lin.weight.copy_(q_src.weight)
                if q_src.bias is not None and self.q_lin.bias is not None:
                    self.q_lin.bias.copy_(q_src.bias)

            if k_src is not None and hasattr(k_src, "weight"):
                self.k_lin.weight.copy_(k_src.weight)
                if k_src.bias is not None and self.k_lin.bias is not None:
                    self.k_lin.bias.copy_(k_src.bias)

            if v_src is not None and hasattr(v_src, "weight"):
                self.v_lin.weight.copy_(v_src.weight)
                if v_src.bias is not None and self.v_lin.bias is not None:
                    self.v_lin.bias.copy_(v_src.bias)

            if out_src is not None and hasattr(out_src, "weight"):
                self.out_lin.weight.copy_(out_src.weight)
                if out_src.bias is not None and self.out_lin.bias is not None:
                    self.out_lin.bias.copy_(out_src.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape

        # Base QKV projections
        q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # MoE Value refinement
        moe_delta_v = self.qkv_moe(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v + 0.1 * moe_delta_v

        # Scaled dot-product attention (bidirectional)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, -10000.0)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask

        attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        proj = self.out_lin(attn_out)

        h1 = self.norm1(hidden_states + self.dropout(proj))
        ffn = self.ffn_moe(h1)
        out = self.norm2(h1 + self.dropout(ffn))

        self.aux_loss = self.qkv_moe.aux_loss + self.ffn_moe.aux_loss

        return out, (probs if output_attentions else None)
