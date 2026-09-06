r"""Gated DeltaNet Block with Optional Multi-Head Attention Bypass.

Implements an efficient hybrid linear-recurrent block switching dynamically between standard
multi-head self-attention and Bidirectional Gated DeltaNet with pretrained weight transfer.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.attentions.gated_delta_net import GatedDeltaNet
from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("gated_deltanet_block")
class GatedDeltaNetBlock(BaseAttention):
    r"""Gated DeltaNet Transformer Block with Pretrained Weight Transfer.

    Mathematical Formulation:
        $$\text{Output} = \begin{cases}
            \text{LayerNorm}(X + \text{MHA}(X, X, X)), & \text{if } \text{use\_full\_attention} \\
            \text{LayerNorm}(X + \text{GatedDeltaNet}(X)), & \text{otherwise}
        \end{cases}$$
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        num_heads: Optional[int] = None,
        use_full_attention: bool = False,
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

        if use_full_attention:
            self.attn: Optional[nn.MultiheadAttention] = nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
        else:
            self.attn = None

        self.deltanet = GatedDeltaNet(
            config=config, d_model=self.d_model, num_heads=self.num_heads, **kwargs
        )
        self.norm = nn.LayerNorm(self.d_model)

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
                                    mod = getattr(mod, "dense")
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
                        if b_q is not None and b_k is not None and b_v is not None and self.attn.in_proj_bias is not None:
                            cat_b = torch.cat([b_q, b_k, b_v], dim=0)
                            if cat_b.shape == self.attn.in_proj_bias.shape:
                                self.attn.in_proj_bias.copy_(cat_b)
                    if w_o is not None:
                        if w_o.shape == self.attn.out_proj.weight.shape:
                            self.attn.out_proj.weight.copy_(w_o)
                        elif w_o.t().shape == self.attn.out_proj.weight.shape:
                            self.attn.out_proj.weight.copy_(w_o.t())
                        if b_o is not None and self.attn.out_proj.bias is not None and b_o.shape == self.attn.out_proj.bias.shape:
                            self.attn.out_proj.bias.copy_(b_o)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass
        else:
            if hasattr(self.deltanet, "copy_weights"):
                self.deltanet.copy_weights(old_attn)

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
        """Forward pass for GatedDeltaNetBlock."""
        x = hidden_states
        if self.use_full_attention and self.attn is not None:
            key_padding_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    key_padding_mask = (attention_mask == 0)
                elif attention_mask.dim() == 4:
                    key_padding_mask = (attention_mask[:, 0, 0, :] <= -10000.0)

            attn_out, attn_weights = self.attn(
                x, x, x, key_padding_mask=key_padding_mask, need_weights=output_attentions
            )
            if attn_out.shape[-1] != x.shape[-1]:
                attn_out = F.pad(attn_out, (0, x.shape[-1] - attn_out.shape[-1])) if attn_out.shape[-1] < x.shape[-1] else attn_out[..., :x.shape[-1]]
            out = self.norm(x + attn_out)
            return out, (attn_weights if output_attentions else None)
        else:
            delta_out, _ = self.deltanet(
                x, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, **kwargs
            )
            if delta_out.shape[-1] != x.shape[-1]:
                delta_out = F.pad(delta_out, (0, x.shape[-1] - delta_out.shape[-1])) if delta_out.shape[-1] < x.shape[-1] else delta_out[..., :x.shape[-1]]
            out = self.norm(x + delta_out)
            return out, None


@ATTENTION_REGISTRY.register("gated_deltanet_block_inject")
def inject_gated_deltanet_block(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> GatedDeltaNetBlock:
    r"""Injects GatedDeltaNetBlock into a transformer layer."""
    d_model = 64
    num_heads = 4
    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None:
        cand = getattr(src_attn, "hidden_size", getattr(src_attn, "embed_dim", getattr(src_attn, "all_head_size", None)))
        if cand is None and hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features"):
            cand = src_attn.q_proj.in_features
        if cand is not None:
            d_model = cand
        num_heads = getattr(src_attn, "num_heads", getattr(src_attn, "num_attention_heads", num_heads))
    elif model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    if model_config is not None:
        d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", d_model))
        num_heads = getattr(model_config, "num_attention_heads", num_heads)
    elif hasattr(layer, "config"):
        d_model = getattr(layer.config, "hidden_size", d_model)
        num_heads = getattr(layer.config, "num_attention_heads", num_heads)

    use_full = getattr(comp_config, "use_full_attention", False) if comp_config else False
    new_block = GatedDeltaNetBlock(
        config=model_config, d_model=d_model, num_heads=num_heads, use_full_attention=use_full, **kwargs
    )
    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None and hasattr(new_block, "copy_weights"):
        new_block.copy_weights(src_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_block
    elif hasattr(layer, "self"):
        layer.self = new_block

    return new_block
