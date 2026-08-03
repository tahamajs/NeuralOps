r"""
Gated DeltaNet Block with HGRN-2 and Optional Full Attention Routing.

Implements an efficient hybrid linear-recurrent block switching dynamically between standard
multi-head self-attention and Hierarchical Gated Recurrent Network v2 (HGRN-2).
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

from .hgrn2_layer import HGRN2Layer


@ATTENTION_REGISTRY.register("gated_deltanet_block")
class GatedDeltaNetBlock(nn.Module):
    r"""
    Gated DeltaNet Transformer Block.

    Mathematical Formulation:
        Given input activations $x \in \mathbb{R}^{B \times L \times D}$:
        $$\text{Output} = \begin{cases}
            \text{LayerNorm}(x + \text{MHA}(x, x, x)), & \text{if } \text{use\_full\_attention} \\
            \text{LayerNorm}(x + \text{HGRN2}(x)), & \text{otherwise}
        \end{cases}$$
        where $\text{HGRN2}(x)$ evaluates element-wise linear recurrence modulated by input and forget gates:
        $$h_t = \text{clip}\left( \sigma(W_f x_t) \odot h_{t-1} + \sigma(W_i x_t) \odot x_t, -C, C \right)$$

    Complexity:
        - Full Attention: $\mathcal{O}(B \cdot L^2 \cdot D)$ time, $\mathcal{O}(B \cdot L^2)$ attention memory.
        - HGRN-2 Mode: $\mathcal{O}(B \cdot L \cdot D^2)$ time, $\mathcal{O}(B \cdot D)$ recurrent state memory.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, L, D]$
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        d_model: Union[int, Any] = 768,
        n_heads: int = 12,
        use_full_attention: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if hasattr(d_model, "hidden_size"):
            config = d_model
            d_model = config.hidden_size
            n_heads = config.num_attention_heads

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.use_full_attention: bool = use_full_attention

        if use_full_attention:
            self.attn: Optional[nn.MultiheadAttention] = nn.MultiheadAttention(
                d_model, n_heads, batch_first=True
            )
        else:
            self.attn = None

        self.hgrn = HGRN2Layer(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for GatedDeltaNetBlock.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Optional attention mask $[B, L]$.
            output_attentions (bool): Whether to return intermediate attention weights.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed hidden activations of shape $[B, L, D]$.
                - None (or attention weights).
        """
        x = hidden_states
        if self.use_full_attention and self.attn is not None:
            key_padding_mask = (
                (attention_mask == 0)
                if attention_mask is not None and attention_mask.dim() == 2
                else None
            )
            attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=key_padding_mask)
            x = self.norm(x + attn_out)
            return x, attn_weights if output_attentions else None
        else:
            hgrn_out, _ = self.hgrn(x, attention_mask=attention_mask)
            x = self.norm(x + hgrn_out)
            return x, None


@ATTENTION_REGISTRY.register("gated_deltanet_block")
def inject_gated_deltanet_block(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> GatedDeltaNetBlock:
    r"""
    Injects GatedDeltaNetBlock into a transformer layer.

    Args:
        layer (nn.Module): Host layer to patch.
        model_config (Any): Model configuration object.
        comp_config (Any): Compression hyperparameter configuration.
        layer_idx (int): Current layer index.
        total_layers (int): Total transformer layers.
        old_attn (Optional[nn.Module]): Pretrained attention layer.

    Returns:
        GatedDeltaNetBlock: Initialized GatedDeltaNet block.
    """
    d_model = model_config.hidden_size
    n_heads = model_config.num_attention_heads
    use_full = getattr(comp_config, "use_full_attention", False)
    new_block = GatedDeltaNetBlock(d_model, n_heads, use_full)
    layer.attention = new_block
    return new_block
