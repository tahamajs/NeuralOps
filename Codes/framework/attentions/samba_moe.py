"""
Samba-MoE: Hybrid Selective State Space (SAMBA-M3) + REAP-MoE Block.
Combines local sliding-window attention, spectral Mamba SSM, and MoE routing.
"""

from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from .samba_m3 import SAMBAM3Block
from .reap_moe import REAPMoE


@ATTENTION_REGISTRY.register("samba_moe")
class SambaMoEBlock(BaseAttention):
    r"""
    SambaMoE: SAMBA-M3 + REAP-MoE Hybrid Architecture with pretrained weight copying.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_experts: int = 8,
        window: int = 128,
        rank: Optional[int] = 64,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model or 768))
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", n_heads or 12))
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            window = getattr(config, "window_size", window)
        else:
            d_model = d_model or 768
            n_heads = n_heads or 12

        self.d_model = d_model
        self.n_heads = n_heads
        self.swa_mamba = SAMBAM3Block(d_model, n_heads, window=window, config=config)
        self.moe = REAPMoE(d_model, n_experts, config=config)
        self.norm = nn.LayerNorm(d_model)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        if hasattr(self.swa_mamba, "copy_weights"):
            self.swa_mamba.copy_weights(pretrained_attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, attn_w = self.swa_mamba(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )

        moe_out, _ = self.moe(x)
        self.aux_loss = self.moe.aux_loss
        out = self.norm(x + moe_out)
        return out, attn_w
