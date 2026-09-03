"""HGRN-2 Gated DeltaNet Hybrid Attention Module.

Combines bidirectional HGRN-2 element-wise linear recurrence with multi-head Bidirectional
Gated DeltaNet associative matrix memory for rich representational capacity.
"""

from typing import Any, Optional, Tuple, Union

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F

from framework.attentions.gated_delta_net import GatedDeltaNet
from framework.attentions.hgrn2_layer import HGRN2Layer
from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet")
class HGRN2GatedDeltaNet(BaseAttention):
    r"""HGRN-2 Gated DeltaNet Hybrid Architecture.

    Mathematical Mechanism:
        Combines two complementary linear recurrent mechanisms:
        1. **HGRN-2 Branch:** Fast element-wise linear recurrence with bounded forget gating.
        2. **Gated DeltaNet Branch:** Multi-head associative matrix state with Delta-rule error updates.
        3. **Dynamic Fusion:**
           $$Y = \gamma \odot Y_{\text{hgrn}} + (1 - \gamma) \odot Y_{\text{deltanet}}$$
           where $\gamma = \sigma(W_g X)$.

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot (D + H \cdot d_k \cdot d_v))$ strictly linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot (D + H \cdot d_k \cdot d_v))$ recurrent memory.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        num_heads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        
        # Resolve d_model
        if d_model is None:
            if isinstance(config, int):
                d_model = config
            elif config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "n_embd", getattr(config, "dim", None))))
        if d_model is None:
            d_model = kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))

        # Resolve num_heads
        if num_heads is None:
            if config is not None and not isinstance(config, int):
                num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", getattr(config, "n_heads", getattr(config, "n_head", None))))
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.d_model: int = int(d_model)
        self.num_heads: int = int(num_heads)

        self.hgrn = HGRN2Layer(config=config, rank=rank, d_model=self.d_model)
        self.deltanet = GatedDeltaNet(
            config=config, rank=rank, d_model=self.d_model, num_heads=self.num_heads
        )
        self.gate = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained Multi-Head Attention weights to both branches."""
        if hasattr(self.hgrn, "copy_weights"):
            self.hgrn.copy_weights(old_attn)
        if hasattr(self.deltanet, "copy_weights"):
            self.deltanet.copy_weights(old_attn)

        try:
            adapter = UniversalAdapter()
            w_o, b_o = adapter.get_weight_bias(old_attn, "o")
            with torch.no_grad():
                self.out_proj.weight.copy_(w_o)
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass

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
        """Forward propagation across hybrid HGRN-2 and Gated DeltaNet branches."""
        hgrn_out, _ = self.hgrn(
            hidden_states, attention_mask=attention_mask, head_mask=head_mask, **kwargs
        )
        delta_out, _ = self.deltanet(
            hidden_states, attention_mask=attention_mask, head_mask=head_mask, **kwargs
        )

        gamma = torch.sigmoid(self.gate(hidden_states))
        fused = gamma * hgrn_out + (1.0 - gamma) * delta_out
        out = self.out_proj(fused)

        return out, None


@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet_inject")
def inject_hgrn2_gated_deltanet(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> HGRN2GatedDeltaNet:
    r"""Injection hook for HGRN2GatedDeltaNet."""
    d_model = 768
    num_heads = 12
    if model_config is not None:
        d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
        num_heads = getattr(model_config, "num_attention_heads", getattr(model_config, "n_heads", 12))
    elif hasattr(layer, "config"):
        d_model = getattr(layer.config, "hidden_size", getattr(layer.config, "dim", 768))
        num_heads = getattr(layer.config, "num_attention_heads", getattr(layer.config, "n_heads", 12))

    new_net = HGRN2GatedDeltaNet(
        d_model=d_model, num_heads=num_heads, config=model_config, **kwargs
    )
    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None and hasattr(new_net, "copy_weights"):
        new_net.copy_weights(src_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_net
    elif hasattr(layer, "self"):
        layer.self = new_net

    return new_net
