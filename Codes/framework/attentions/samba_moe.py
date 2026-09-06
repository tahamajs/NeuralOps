"""
Samba-MoE: Hybrid Selective State Space (SAMBA-M3) + REAP-MoE Block.
Combines local sliding-window attention, spectral Mamba SSM, and MoE routing.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

import logging
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from .samba_m3 import SAMBAM3Block
from .reap_moe import REAPMoE

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("samba_moe", supports_causal=True, supports_gqa=True)
class SambaMoEBlock(BaseAttention):
    r"""SambaMoE: SAMBA-M3 + REAP-MoE Hybrid Architecture with pretrained weight copying."""

    supports_gqa: bool = True
    supports_causal: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_experts: int = 8,
        window: int = 128,
        rank: Optional[int] = 64,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
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

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.swa_mamba = SAMBAM3Block(
            config=config,
            d_model=self.d_model,
            n_heads=self.n_heads,
            window=window,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            is_decoder=is_decoder,
            **kwargs,
        )
        self.moe = REAPMoE(self.d_model, n_experts, config=config)
        self.norm = nn.LayerNorm(self.d_model)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        """Propagate weight transfer to internal SAMBA-M3 and SSM components."""
        if hasattr(self.swa_mamba, "copy_weights"):
            self.swa_mamba.copy_weights(pretrained_attn)

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
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        x, attn_w = self.swa_mamba(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if self.swa_mamba.is_nested_encoder:
            # Nested encoder bypass
            if x.shape[-1] != hidden_states.shape[-1]:
                x = F.pad(x, (0, hidden_states.shape[-1] - x.shape[-1])) if x.shape[-1] < hidden_states.shape[-1] else x[..., :hidden_states.shape[-1]]
            return (x, attn_w)

        moe_out, _ = self.moe(x)
        self.aux_loss = self.moe.aux_loss
        out = self.norm(x + moe_out)
        if out.shape[-1] != hidden_states.shape[-1]:
            out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1])) if out.shape[-1] < hidden_states.shape[-1] else out[..., :hidden_states.shape[-1]]
        return (out, attn_w)


@ATTENTION_REGISTRY.register("samba_moe_inject")
def inject_samba_moe(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[SambaMoEBlock]:
    """Safe injection hook for SambaMoEBlock."""
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break

        if src is None:
            return None

        d_model = 768
        n_heads = 12
        if src is not None:
            d_model = getattr(src, "hidden_size", getattr(src, "embed_dim", getattr(src, "all_head_size", 768)))
            if hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features"):
                d_model = src.q_proj.in_features
            n_heads = getattr(src, "num_heads", getattr(src, "num_attention_heads", 12))
        elif model_config is None and hasattr(layer, "config"):
            model_config = layer.config
        if model_config is not None:
            d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", d_model))
            n_heads = getattr(model_config, "num_attention_heads", n_heads)
        elif hasattr(layer, "config"):
            d_model = getattr(layer.config, "hidden_size", d_model)
            n_heads = getattr(layer.config, "num_attention_heads", n_heads)

        new_block = SambaMoEBlock(
            config=model_config,
            d_model=d_model,
            n_heads=n_heads,
            pretrained_attn=src,
            **kwargs,
        )

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_block)
                return new_block

        return new_block
    except Exception as e:
        logger.error("inject_samba_moe failed at layer %d: %s", layer_idx, e)
        return None
