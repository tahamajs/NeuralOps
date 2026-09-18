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

try:
    from .reap_moe import REAPMoE
except Exception:
    REAPMoE = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class MinimalMoEFallback(nn.Module):
    """Fallback MoE layer when REAPMoE cannot be instantiated."""

    def __init__(self, d_model: int = 768, n_experts: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(n_experts)
        ])
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_in = self.router.in_features
        if x.shape[-1] != router_in:
            x_in = F.pad(x, (0, router_in - x.shape[-1])) if x.shape[-1] < router_in else x[..., :router_in]
        else:
            x_in = x
        logits = self.router(x_in)
        weights = F.softmax(logits, dim=-1)
        out = torch.zeros_like(x_in)
        for i, exp in enumerate(self.experts):
            out = out + weights[..., i : i + 1] * exp(x_in)
        if out.shape[-1] != x.shape[-1]:
            out = F.pad(out, (0, x.shape[-1] - out.shape[-1])) if out.shape[-1] < x.shape[-1] else out[..., :x.shape[-1]]
        self.aux_loss = torch.tensor(0.0, device=x.device)
        return out, weights


@ATTENTION_REGISTRY.register("samba_moe", supports_causal=True, supports_gqa=True)
class SambaMoEBlock(BaseAttention):
    r"""SambaMoE: SAMBA-M3 + REAP-MoE Hybrid Architecture with pretrained weight copying."""

    supports_gqa: bool = True
    supports_causal: bool = True
    is_custom_attention: bool = True

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
        hidden_size = kwargs.get("d_model") or kwargs.get("hidden_size") or d_model
        if pretrained_attn is not None:
            cand = getattr(pretrained_attn, "hidden_size", getattr(pretrained_attn, "embed_dim", getattr(pretrained_attn, "all_head_size", None)))
            if cand is None and hasattr(pretrained_attn, "q_proj") and hasattr(pretrained_attn.q_proj, "in_features"):
                cand = pretrained_attn.q_proj.in_features
            elif cand is None and hasattr(pretrained_attn, "q_lin") and hasattr(pretrained_attn.q_lin, "in_features"):
                cand = pretrained_attn.q_lin.in_features
            if cand is not None:
                hidden_size = cand
        if hidden_size is None and config is not None:
            hidden_size = _cfg_get(config, "hidden_size", _cfg_get(config, "d_model", None))
        hidden_size = int(hidden_size or 768)

        if config is not None:
            n_heads = _cfg_get(config, "num_attention_heads", _cfg_get(config, "n_heads", n_heads or 12))
            n_experts = _cfg_get(config, "num_experts", _cfg_get(config, "n_experts", n_experts))
            window = _cfg_get(config, "window_size", window)
        else:
            n_heads = n_heads or 12

        self.d_model = hidden_size
        self.n_heads = int(n_heads)
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        self.swa_mamba = SAMBAM3Block(
            config=config,
            d_model=self.d_model,
            n_heads=self.n_heads,
            window=window,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            is_decoder=is_decoder,
            **kwargs,
        )

        if REAPMoE is not None:
            try:
                self.moe = REAPMoE(self.d_model, n_experts=n_experts)
            except Exception as e:
                logger.warning("REAPMoE construction failed: %s, using MinimalMoEFallback", e)
                self.moe = MinimalMoEFallback(self.d_model, n_experts=min(4, n_experts))
        else:
            self.moe = MinimalMoEFallback(self.d_model, n_experts=min(4, n_experts))

        self.norm = nn.LayerNorm(self.d_model)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def is_nested_encoder(self) -> bool:
        return getattr(self.swa_mamba, "is_nested_encoder", False)

    @is_nested_encoder.setter
    def is_nested_encoder(self, val: bool) -> None:
        if hasattr(self, "swa_mamba"):
            self.swa_mamba.is_nested_encoder = val

    @property
    def query(self) -> nn.Module:
        return getattr(self.swa_mamba, "q_lin", None)

    @property
    def key(self) -> nn.Module:
        return getattr(self.swa_mamba, "k_lin", None)

    @property
    def value(self) -> nn.Module:
        return getattr(self.swa_mamba, "v_lin", None)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        """Propagate weight transfer to internal SAMBA-M3 and SSM components."""
        src_cls_name = getattr(pretrained_attn, "__class__", type(pretrained_attn)).__name__
        has_out_proj = any(hasattr(pretrained_attn, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        if (
            src_cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (src_cls_name.endswith("SelfAttention") and not has_out_proj)
            or (hasattr(pretrained_attn, "query") and not has_out_proj)
            or (hasattr(pretrained_attn, "self") and hasattr(pretrained_attn, "output"))
        ):
            self.is_nested_encoder = True
            if hasattr(self, "swa_mamba"):
                self.swa_mamba.is_nested_encoder = True

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
        if encoder_hidden_states is not None:
            raise NotImplementedError("SambaMoEBlock does not support cross-attention")

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

        router_in = getattr(self.moe, "d_model", self.d_model)
        if hasattr(self.moe, "router") and hasattr(self.moe.router, "in_features"):
            router_in = self.moe.router.in_features

        if x.shape[-1] != router_in:
            x_moe = F.pad(x, (0, router_in - x.shape[-1])) if x.shape[-1] < router_in else x[..., :router_in]
        else:
            x_moe = x

        moe_out, _ = self.moe(x_moe)
        self.aux_loss = getattr(self.moe, "aux_loss", torch.tensor(0.0, device=x.device))

        if moe_out.shape[-1] != x.shape[-1]:
            moe_out = F.pad(moe_out, (0, x.shape[-1] - moe_out.shape[-1])) if moe_out.shape[-1] < x.shape[-1] else moe_out[..., :x.shape[-1]]

        if self.is_nested_encoder:
            # Nested encoder: bypass outer LayerNorm (self.norm) and double residual
            out = x + moe_out
        else:
            out = self.norm(x + moe_out)

        if out.shape[-1] != hidden_states.shape[-1]:
            out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1])) if out.shape[-1] < hidden_states.shape[-1] else out[..., :hidden_states.shape[-1]]
        return (out, attn_w)


@ATTENTION_REGISTRY.register("samba_moe_inject")
@ATTENTION_REGISTRY.register("inject_samba_moe")
def inject_samba_moe(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[SambaMoEBlock]:
    """Safe injection hook for SambaMoEBlock."""
    try:
        target = layer if layer is not None else model
        if target is None:
            target = kwargs.get("model", kwargs.get("layer", None))
        if target is None:
            return None

        # 1. Whole model traversal
        layer_list = None
        if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer
        elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
            layer_list = target.layers

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_samba_moe(
                    layer=lyr,
                    model_config=model_config,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

        # 2. Single layer injection
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break

        if src is None:
            return None

        if model_config is None and hasattr(layer, "config"):
            model_config = layer.config
        if model_config is None and hasattr(src, "config"):
            model_config = src.config

        d_model = 768
        n_heads = 12
        if src is not None:
            d_model = getattr(src, "hidden_size", getattr(src, "embed_dim", getattr(src, "all_head_size", 768)))
            if hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features"):
                d_model = src.q_proj.in_features
            elif hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features"):
                d_model = src.q_lin.in_features
            n_heads = getattr(src, "num_heads", getattr(src, "num_attention_heads", 12))

        if model_config is not None:
            d_model = _cfg_get(model_config, "hidden_size", _cfg_get(model_config, "dim", d_model))
            n_heads = _cfg_get(model_config, "num_attention_heads", n_heads)

        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
        lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 32.0))
        n_experts = kwargs.pop("n_experts", _cfg_get(comp_config, "n_experts", 8))

        new_block = SambaMoEBlock(
            config=model_config,
            d_model=d_model,
            n_heads=n_heads,
            n_experts=n_experts,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            pretrained_attn=src,
            **kwargs,
        )

        for attr in ("self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_block)
                return new_block
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_block
            else:
                layer.attention = new_block
            return new_block
        layer.attention = new_block
        return new_block
    except Exception as e:
        logger.error("inject_samba_moe failed at layer %d: %s", layer_idx, e)
        return None
