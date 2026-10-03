"""Gated DeltaNet Block with Optional Multi-Head Attention Bypass.

Implements an efficient hybrid linear-recurrent block switching dynamically between standard
multi-head self-attention and Bidirectional Gated DeltaNet with pretrained weight transfer.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.attentions.gated_delta_net import GatedDeltaNet
from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("gated_deltanet_block")
class GatedDeltaNetBlock(BaseAttention):
    r"""Gated DeltaNet Transformer Block with Pretrained Weight Transfer.

    Mathematical Formulation:
        $$\text{Output} = \begin{cases}
            \text{LayerNorm}(X + \text{MHA}(X, X, X)), & \text{if } \text{use\_full\_attention} \\
            \text{LayerNorm}(X + \text{GatedDeltaNet}(X)), & \text{otherwise}
        \end{cases}$$
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        use_full_attention: bool = False,
        comp_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
            else:
                d_model = kwargs.get("hidden_size", 768)

        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", 4)
            else:
                num_heads = kwargs.get("num_attention_heads", 4)

        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.use_full_attention: bool = use_full_attention
        self.comp_config = comp_config

        if use_full_attention:
            self.attn: nn.MultiheadAttention | None = nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
        else:
            self.attn = None

        self.deltanet = GatedDeltaNet(
            config=config,
            comp_config=comp_config,
            d_model=self.d_model,
            num_heads=self.num_heads,
            **kwargs,
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.is_nested_encoder: bool = getattr(
            self.deltanet, "is_nested_encoder", False
        )

    # Compatibility properties
    @property
    def query(self) -> Any:
        return self.deltanet.query

    @property
    def key(self) -> Any:
        return self.deltanet.key

    @property
    def value(self) -> Any:
        return self.deltanet.value

    @property
    def q_proj(self) -> Any:
        return self.deltanet.q_proj

    @property
    def k_proj(self) -> Any:
        return self.deltanet.k_proj

    @property
    def v_proj(self) -> Any:
        return self.deltanet.v_proj

    @property
    def out_proj(self) -> Any:
        return self.deltanet.out_proj

    @property
    def q_lin(self) -> Any:
        return self.deltanet.q_lin

    @property
    def k_lin(self) -> Any:
        return self.deltanet.k_lin

    @property
    def v_lin(self) -> Any:
        return self.deltanet.v_lin

    @property
    def out_lin(self) -> Any:
        return self.deltanet.out_lin

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained weights to either MHA or GatedDeltaNet."""
        if self.use_full_attention and self.attn is not None:
            try:
                adapter = UniversalAdapter()
                w_q, b_q = adapter.get_weight_bias(old_attn, "q")
                w_k, b_k = adapter.get_weight_bias(old_attn, "k")
                w_v, b_v = adapter.get_weight_bias(old_attn, "v")
                w_o, b_o = adapter.get_weight_bias(old_attn, "o")

                if w_o is None:
                    for cand in ("out_proj", "out_lin", "o_proj", "output", "proj"):
                        for target in (old_attn, getattr(old_attn, "self", None)):
                            if target is not None and hasattr(target, cand):
                                mod = getattr(target, cand)
                                if hasattr(mod, "dense"):
                                    mod = mod.dense
                                if hasattr(mod, "weight") and mod.weight is not None:
                                    w_o = mod.weight.data.clone()
                                    if w_o.ndim == 2 and w_o.shape[0] < w_o.shape[1]:
                                        w_o = w_o.t()
                                    b_o = getattr(mod, "bias", None)
                                    b_o = b_o.data.clone() if b_o is not None else None
                                    break
                        if w_o is not None:
                            break

                with torch.no_grad():
                    if w_q is not None and w_k is not None and w_v is not None:
                        qkv_w = torch.cat([w_q, w_k, w_v], dim=0)
                        if qkv_w.shape == self.attn.in_proj_weight.shape:
                            self.attn.in_proj_weight.copy_(qkv_w)
                        if (
                            b_q is not None
                            and b_k is not None
                            and b_v is not None
                            and self.attn.in_proj_bias is not None
                        ):
                            cat_b = torch.cat([b_q, b_k, b_v], dim=0)
                            if cat_b.shape == self.attn.in_proj_bias.shape:
                                self.attn.in_proj_bias.copy_(cat_b)
                    if w_o is not None:
                        if w_o.shape == self.attn.out_proj.weight.shape:
                            self.attn.out_proj.weight.copy_(w_o)
                        elif w_o.t().shape == self.attn.out_proj.weight.shape:
                            self.attn.out_proj.weight.copy_(w_o.t())
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(
                    "GatedDeltaNetBlock copy_weights to MultiheadAttention incomplete: %s",
                    e,
                )
        else:
            if hasattr(self.deltanet, "copy_weights"):
                self.deltanet.copy_weights(old_attn)
                self.is_nested_encoder = getattr(
                    self.deltanet, "is_nested_encoder", False
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for GatedDeltaNetBlock."""
        x = hidden_states
        if self.use_full_attention and self.attn is not None:
            key_padding_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    key_padding_mask = attention_mask == 0
                elif attention_mask.dim() == 4:
                    key_padding_mask = attention_mask[:, 0, 0, :] <= -10000.0

            attn_out, attn_weights = self.attn(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=output_attentions,
            )
            if attn_out.shape[-1] != x.shape[-1]:
                attn_out = (
                    F.pad(attn_out, (0, x.shape[-1] - attn_out.shape[-1]))
                    if attn_out.shape[-1] < x.shape[-1]
                    else attn_out[..., : x.shape[-1]]
                )
            if self.is_nested_encoder:
                return attn_out, (attn_weights if output_attentions else None)
            out = self.norm(x + attn_out)
            return out, (attn_weights if output_attentions else None)
        else:
            delta_out, _ = self.deltanet(
                x,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
            if delta_out.shape[-1] != x.shape[-1]:
                delta_out = (
                    F.pad(delta_out, (0, x.shape[-1] - delta_out.shape[-1]))
                    if delta_out.shape[-1] < x.shape[-1]
                    else delta_out[..., : x.shape[-1]]
                )
            if self.is_nested_encoder:
                return delta_out, None
            if getattr(self, "is_nested_encoder", False):
                return delta_out, None
            out = self.norm(x + delta_out)
            return out, None


@ATTENTION_REGISTRY.register("gated_deltanet_block_inject")
@ATTENTION_REGISTRY.register("inject_gated_deltanet_block")
def inject_gated_deltanet_block(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(
        target.layers, (torch.nn.ModuleList, list)
    ):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_gated_deltanet_block(
                layer=lyr,
                model_config=cfg,
                comp_config=(
                    comp_config
                    if "comp_config" in locals() or "comp_config" in kwargs
                    else None
                ),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""Injects GatedDeltaNetBlock into a transformer layer."""
    src_attn = old_attn
    if src_attn is None:
        if hasattr(layer, "attention"):
            src_attn = layer.attention
        elif hasattr(layer, "attn"):
            src_attn = layer.attn
        elif hasattr(layer, "self_attn"):
            src_attn = layer.self_attn
        elif hasattr(layer, "self"):
            src_attn = layer.self
        else:
            src_attn = layer

    d_model = 64
    num_heads = 4
    if src_attn is not None:
        cand = getattr(
            src_attn,
            "hidden_size",
            getattr(src_attn, "embed_dim", getattr(src_attn, "all_head_size", None)),
        )
        if (
            cand is None
            and hasattr(src_attn, "q_proj")
            and hasattr(src_attn.q_proj, "in_features")
        ):
            cand = src_attn.q_proj.in_features
        if cand is not None:
            d_model = cand
        num_heads = getattr(
            src_attn, "num_heads", getattr(src_attn, "num_attention_heads", num_heads)
        )
    elif model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    if model_config is not None:
        d_model = getattr(
            model_config, "hidden_size", getattr(model_config, "dim", d_model)
        )
        num_heads = getattr(model_config, "num_attention_heads", num_heads)
    elif hasattr(layer, "config"):
        d_model = getattr(layer.config, "hidden_size", d_model)
        num_heads = getattr(layer.config, "num_attention_heads", num_heads)

    try:
        use_full = (
            getattr(comp_config, "use_full_attention", False) if comp_config else False
        )
        new_block = GatedDeltaNetBlock(
            config=model_config,
            comp_config=comp_config,
            d_model=d_model,
            num_heads=num_heads,
            use_full_attention=use_full,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            **kwargs,
        )
        if src_attn is not None and hasattr(new_block, "copy_weights"):
            new_block.copy_weights(src_attn)

        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_block.is_nested_encoder = True
                new_block.deltanet.is_nested_encoder = True
                layer.attention.self = new_block
            else:
                new_block.is_nested_encoder = False
                new_block.deltanet.is_nested_encoder = False
                layer.attention = new_block
        elif hasattr(layer, "self"):
            new_block.is_nested_encoder = True
            new_block.deltanet.is_nested_encoder = True
            layer.self = new_block
        elif hasattr(layer, "attn"):
            new_block.is_nested_encoder = False
            new_block.deltanet.is_nested_encoder = False
            layer.attn = new_block
        elif hasattr(layer, "self_attn"):
            new_block.is_nested_encoder = False
            new_block.deltanet.is_nested_encoder = False
            layer.self_attn = new_block
        assert (
            new_block.is_nested_encoder == new_block.deltanet.is_nested_encoder
        ), "GatedDeltaNetBlock nested flag desynced"

        return new_block
    except Exception as e:
        logger.error(f"GatedDeltaNetBlock injection failed: {e}")
        return None
