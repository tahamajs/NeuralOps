"""
Expert Choice Mixture of Experts (MoE) Attention Block.

In Expert Choice routing (Zhou et al., 2022), experts select their top-k tokens
rather than tokens selecting top-k experts. This provides guaranteed perfect load balancing
across all experts, dynamic per-token expert allocation, and superior zero-shot transfer.
"""

from typing import Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class ExpertChoiceMoE(nn.Module):
    r"""
    Expert Choice Routing MoE layer.
    Each expert selects top-k tokens with the highest affinity scores.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        capacity_factor: float = 1.0,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        # Router matrix [d_model, n_experts] initialized uniformly for balanced initial gating
        self.router_ln = nn.LayerNorm(self.d_model)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward execution using Expert Choice routing."""
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # [T, D] where T = B * L
        T = x_flat.size(0)

        # Routing scores [T, E]
        logits = self.router(self.router_ln(x_flat))
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # Softmax over tokens per expert for expert-choice gating [T, E]
        gate_scores = F.softmax(logits, dim=0)

        # Expert capacity: number of tokens each expert processes
        k = max(1, min(T, int(T * self.capacity_factor / self.n_experts)))

        out_flat = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            # Expert e picks top-k tokens across the entire sequence/batch
            scores_e = gate_scores[:, e]
            topk_vals, topk_indices = torch.topk(scores_e, k, dim=0)

            selected_x = x_flat[topk_indices]
            expert_out = self.experts[e](selected_x)

            # Scatter back weighted by gating score
            out_flat.index_add_(0, topk_indices, expert_out * topk_vals.unsqueeze(-1))

        # Auxiliary entropy / balance loss
        prob_mean = F.softmax(logits / 1.0, dim=-1).mean(dim=0)
        self.aux_loss = self.n_experts * torch.sum(prob_mean ** 2)

        return out_flat.view(B, L, D)


@ATTENTION_REGISTRY.register("expert_choice_moe")
class ExpertChoiceMoEAttention(BaseAttention):
    r"""
    Expert Choice MoE Multi-Head Attention Module.
    Combines standard Multi-Head Attention with Expert Choice routing feed-forward transformation.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        n_experts: int = 8,
        capacity_factor: float = 1.0,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.dim = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.head_dim = self.dim // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.q_lin = nn.Linear(self.dim, self.all_head_size)
        self.k_lin = nn.Linear(self.dim, self.all_head_size)
        self.v_lin = nn.Linear(self.dim, self.all_head_size)
        self.out_lin = nn.Linear(self.all_head_size, self.dim)

        self.moe = ExpertChoiceMoE(self.dim, n_experts=n_experts, capacity_factor=capacity_factor)
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_p)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape

        q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

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
        moe_out = self.moe(h1)
        out = self.norm2(h1 + self.dropout(moe_out))
        self.aux_loss = self.moe.aux_loss

        return out, (probs if output_attentions else None)
