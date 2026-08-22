# jamba.py
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

# فرض: mamba.py شامل SVDMamba و SVDLinear است
from framework.attentions.mamba import SVDMamba


class ExpertChoiceMoE(nn.Module):
    r"""
    Top-k noisy gating Mixture of Experts با تعادل بار.

    نوآوری‌ها:
    - نویز تصادفی در آموزش برای جلوگیری از فروپاشی روتر
    - aux_loss برای توزیع عادلانه توکن‌ها بین متخصص‌ها
    - هر متخصص از SVDLinear استفاده می‌کند (کم‌رتبه و منظم‌تر)
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        top_k: int = 2,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.aux_loss = 0.0

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(n_experts)
        ])

        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Gating با نویز در آموزش
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(self.router(x)) * self.noise_std
            scores = self.router(x) + noise
        else:
            scores = self.router(x)

        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)  # [B, L, k]
        topk_weights = F.softmax(topk_weights, dim=-1)                        # [B, L, k]

        # Load balancing loss
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_experts).float()  # [B, L, k, E]
        expert_counts = expert_mask.sum(dim=(0, 1, 2))  # [E]
        router_probs = F.softmax(self.router(x), dim=-1).mean(dim=(0, 1))  # [E]
        self.aux_loss = self.n_experts * torch.sum(expert_counts * router_probs) / (B * L)

        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = (topk_indices == e).any(dim=-1)  # [B, L]
            if mask.any():
                selected_weights = (topk_weights * (topk_indices == e).float()).sum(
                    dim=-1, keepdim=True
                )  # [B, L, 1]
                expert_input = x * mask.unsqueeze(-1)
                expert_output = self.experts[e](expert_input)
                out += selected_weights * expert_output

        return out


@ATTENTION_REGISTRY.register("jamba2")
class Jamba2Block(BaseAttention):
    r"""
    Jamba-2 Hybrid: SVD-Mamba + MHA + MoE با مدیریت ماسک.

    نوآوری:
    - از SVD-Mamba به‌جای Mamba ساده استفاده می‌کند.
    - MoE با تعادل بار و Gating نویزی.
    - ماسک به زیرلایه‌ها منتقل می‌شود.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int = 8,
        top_k: int = 2,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.d_model = d_model
        self.n_heads = n_heads

        self.mamba = SVDMamba(d_model, config=config)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe = ExpertChoiceMoE(d_model, n_experts, top_k)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Mamba branch
        mamba_out, _ = self.mamba(hidden_states, attention_mask=attention_mask)
        x = self.norm1(hidden_states + mamba_out)

        # Attention mask conversion
        if attention_mask is not None and attention_mask.dim() == 4:
            attn_mask = attention_mask[:, 0, 0, :]  # [B, L]
            attn_mask = (1.0 - attn_mask) * -10000.0
        elif attention_mask is not None and attention_mask.dim() == 2:
            attn_mask = (1.0 - attention_mask) * -10000.0
        else:
            attn_mask = None

        attn_out, attn_weights = self.attn(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=output_attentions,
        )
        x = self.norm2(x + attn_out)

        moe_out = self.moe(x)
        x = self.norm3(x + moe_out)

        self.aux_loss = self.moe.aux_loss

        return x, (attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("jamba2_inject")
def inject_jamba2(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> Jamba2Block:
    d_model = model_config.hidden_size
    n_heads = getattr(model_config, "num_attention_heads", 12)
    n_experts = getattr(comp_config, "n_experts", 8)
    top_k = getattr(comp_config, "top_k", 2)

    new_block = Jamba2Block(
        d_model=d_model,
        n_heads=n_heads,
        n_experts=n_experts,
        top_k=top_k,
        config=model_config,
    )
    layer.attention = new_block
    return new_block