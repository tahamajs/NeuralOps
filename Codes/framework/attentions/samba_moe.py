# samba_moe.py
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY

# توجه: این فایل به SAMBAM3Block از samba_m3.py نیاز دارد
from .samba_m3 import SAMBAM3Block


class SVDLinear(nn.Module):
    r"""
    لایه خطی فاکتورگیری‌شده SVD: W = U @ diag(S) @ V^T

    مزایا:
    - کاهش تعداد پارامترها
    - القای ساختار low-rank و regularization بهتر
    """
    def __init__(self, in_features: int, out_features: int, rank: Optional[int] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank or min(in_features, out_features) // 4

        self.U = nn.Parameter(torch.randn(out_features, self.rank) * 0.02)
        self.S = nn.Parameter(torch.ones(self.rank))
        self.V = nn.Parameter(torch.randn(in_features, self.rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.V) * self.S @ self.U.t() + self.bias


@ATTENTION_REGISTRY.register("reap_moe")
class REAPMoE(nn.Module):
    r"""
    Mixture-of-Experts با gating نویزی و تعادل بار.

    Args:
        d_model: بعد ویژگی D.
        n_experts: تعداد متخصص‌ها E.
        top_k: تعداد متخصص فعال برای هر توکن k.
        noise_std: انحراف معیار نویز در آموزش.
        use_svd: استفاده از SVDLinear در expertها.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_experts: int = 8,
        top_k: int = 2,
        noise_std: float = 0.1,
        use_svd: bool = True,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            top_k = getattr(config, "top_k", top_k)

        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.aux_loss = 0.0

        # ساخت expertها
        if use_svd:
            expert_layer = lambda: nn.Sequential(
                SVDLinear(d_model, d_model * 4, rank=min(d_model, d_model * 4) // 4),
                nn.GELU(),
                SVDLinear(d_model * 4, d_model, rank=min(d_model * 4, d_model) // 4),
            )
        else:
            expert_layer = lambda: nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )

        self.experts = nn.ModuleList([expert_layer() for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = x.shape

        # Gating با نویز در آموزش
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(self.router(x)) * self.noise_std
            scores = self.router(x) + noise
        else:
            scores = self.router(x)

        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)  # [B, L, k]
        topk_weights = F.softmax(topk_weights, dim=-1)                        # [B, L, k]

        # Load balancing loss (auxiliary)
        # expert_mask: [B, L, k, E] -> one-hot of selected experts
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_experts).float()
        # تعداد دفعات انتخاب هر expert
        expert_counts = expert_mask.sum(dim=(0, 1, 2))  # [E]
        # احتمال میانگین روتر برای هر expert
        router_probs = F.softmax(self.router(x), dim=-1).mean(dim=(0, 1))  # [E]
        # aux_loss = E * sum(counts * probs) / (B*L)
        self.aux_loss = self.n_experts * torch.sum(expert_counts * router_probs) / (B * L)

        # اجرای بهینه‌تر با gather/scatter
        out_flat = torch.zeros(B * L, D, device=x.device, dtype=x.dtype)
        x_flat = x.view(-1, D)

        for e in range(self.n_experts):
            # ماسک توکن‌هایی که این expert را انتخاب کرده‌اند
            mask = (topk_indices == e).any(dim=-1)          # [B, L]
            if mask.any():
                # وزن نرمال‌شده هر توکن برای این expert
                weights_e = (topk_weights * (topk_indices == e).float()).sum(dim=-1)  # [B, L]
                flat_mask = mask.view(-1)                    # [B*L]

                # انتخاب توکن‌های فعال
                selected_x = x_flat[flat_mask]               # [num_selected, D]
                selected_w = weights_e.view(-1)[flat_mask]   # [num_selected]

                # اجرای expert روی توکن‌های فعال
                expert_out = self.experts[e](selected_x)     # [num_selected, D]

                # افزودن وزن‌دار به خروجی
                out_flat[flat_mask] += expert_out * selected_w.unsqueeze(-1)

        out = out_flat.view(B, L, D)
        return out, None


@ATTENTION_REGISTRY.register("samba_moe")
class SambaMoEBlock(nn.Module):
    r"""
    بلوک ترکیبی SambaMoE: SAMBA-M3 + REAPMoE با مدیریت ماسک.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_experts: int = 8,
        window: int = 128,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", n_heads))
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            window = getattr(config, "window_size", window)

        self.swa_mamba = SAMBAM3Block(d_model, n_heads, window=window)
        self.moe = REAPMoE(d_model, n_experts, config=config)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # مرحله اول: SAMBA-M3
        x, attn_w = self.swa_mamba(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        # مرحله دوم: MoE
        moe_out = self.moe(x)
        if isinstance(moe_out, tuple):
            moe_out = moe_out[0]

        # ذخیره aux_loss برای استفاده خارجی
        self.aux_loss = self.moe.aux_loss

        x = self.norm(x + moe_out)
        return x, attn_w


@ATTENTION_REGISTRY.register("samba_moe_inject")
def inject_samba_moe(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> SambaMoEBlock:
    d_model = model_config.hidden_size
    n_heads = getattr(model_config, "num_attention_heads", 12)
    n_experts = getattr(comp_config, "n_experts", 8)
    window = getattr(comp_config, "window_size", 128)

    new_block = SambaMoEBlock(
        d_model=d_model,
        n_heads=n_heads,
        n_experts=n_experts,
        window=window,
        config=model_config,
    )
    layer.attention = new_block
    return new_block