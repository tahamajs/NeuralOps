"""
REAP-MoE: Robust Expert Allocation and Pruning Mixture of Experts Attention Module.

Implements a vectorized Mixture-of-Experts block with load-balancing regularization,
SVD low-rank expert factorization, and multi-head attention weight transfer.
"""

from typing import Any, Optional, Tuple, Dict, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class SVDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: Optional[int] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank or max(1, min(in_features, out_features) // 4)
        self.U = nn.Parameter(torch.randn(out_features, self.rank) * 0.02)
        self.S = nn.Parameter(torch.ones(self.rank))
        self.V = nn.Parameter(torch.randn(in_features, self.rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.V) * self.S @ self.U.t() + self.bias


class REAPMoE(nn.Module):
    r"""
    Mixture-of-Experts with noisy gating, load-balancing loss, and SVD factorized experts.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_experts: int = 8,
        top_k: int = 2,
        noise_std: float = 0.05,
        use_svd: bool = True,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            top_k = getattr(config, "top_k", top_k)

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.noise_std = noise_std
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        # Router with uniform initialization
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)

        if use_svd:
            expert_layer = lambda: nn.Sequential(
                SVDLinear(d_model, d_model * 2, rank=max(1, min(d_model, d_model * 2) // 4)),
                nn.GELU(),
                SVDLinear(d_model * 2, d_model, rank=max(1, min(d_model * 2, d_model) // 4)),
            )
        else:
            expert_layer = lambda: nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )

        self.experts = nn.ModuleList([expert_layer() for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = x.shape

        logits = self.router(x)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            scores = logits + noise
        else:
            scores = logits

        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)  # [B, L, k]
        topk_weights = F.softmax(topk_weights, dim=-1)

        # Load balancing auxiliary loss
        router_probs = F.softmax(logits, dim=-1).view(-1, self.n_experts)
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_experts).float()
        density = expert_mask.sum(dim=2).view(-1, self.n_experts).mean(dim=0)
        prob_mean = router_probs.mean(dim=0)
        self.aux_loss = self.n_experts * torch.sum(density * prob_mean)

        # Expert output calculation
        experts_stack = torch.stack([exp(x) for exp in self.experts], dim=2)  # [B, L, E, D]
        topk_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        gathered = torch.gather(experts_stack, 2, topk_expanded)  # [B, L, k, D]

        out = torch.sum(gathered * topk_weights.unsqueeze(-1), dim=2)
        return out, None


@ATTENTION_REGISTRY.register("reap_moe")
class REAPMoEAttention(BaseAttention):
    r"""
    REAP-MoE Attention Block combining Multi-Head Attention with REAP MoE feed-forward transformation.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        n_experts: int = 8,
        top_k: int = 2,
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

        self.moe = REAPMoE(self.dim, n_experts=n_experts, top_k=top_k, config=config)
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
        output_attentions: bool = False,
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
        moe_out, _ = self.moe(h1)
        out = self.norm2(h1 + self.dropout(moe_out))
        self.aux_loss = self.moe.aux_loss

        return out, (probs if output_attentions else None)