from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.gated_deltanet_block import GatedDeltaNetBlock


@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet")
class HGRN2GatedDeltaNet(BaseAttention):
    r"""
    HGRN-2 Gated DeltaNet Hybrid Architecture.

    Mathematical Mechanism:
        Arranges a sequence of $N$ layers with a configurable hybrid ratio $R:1$ of
        linear HGRN-2 recurrence to dense Multihead Self-Attention:
        $$X^{(l+1)} = \begin{cases}
        \text{MHA-Block}(X^{(l)}), & \text{if } l \pmod{R+1} = R \\
        \text{HGRN2-Block}(X^{(l)}), & \text{otherwise}
        \end{cases}$$

    Computational Complexity:
        - HGRN-2 Steps: $\mathcal{O}(B \cdot L \cdot D)$ time, $\mathcal{O}(B \cdot D)$ memory per token.
        - Attention Steps: $\mathcal{O}(B \cdot L^2 \cdot D)$ time, $\mathcal{O}(B \cdot L^2)$ memory.
        - Overall Time: $\mathcal{O}\left(B \cdot N \cdot \left(\frac{R}{R+1} L \cdot D + \frac{1}{R+1} L^2 \cdot D\right)\right)$.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ratio: int = 3,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__(config, rank)
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers
        self.ratio: int = ratio

        self.layers: nn.ModuleList = nn.ModuleList()
        for i in range(n_layers):
            use_full = (i % (ratio + 1) == ratio)
            self.layers.append(GatedDeltaNetBlock(d_model, n_heads, use_full_attention=use_full))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward propagation across HGRN2 Gated DeltaNet stack.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Unused flag for interface compatibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple `(output, None)` with output of shape `[B, L, D]`.
        """
        x = hidden_states
        for layer in self.layers:
            x, _ = layer(x, attention_mask=attention_mask, **kwargs)
        return x, None


@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet")
def inject_hgrn2_gated_deltanet(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> HGRN2GatedDeltaNet:
    r"""
    Injection hook to instantiate and place HGRN2GatedDeltaNet into a transformer layer.

    Args:
        layer: Transformer layer module.
        model_config: Model configuration object.
        comp_config: Compression configuration object.
        layer_idx: 0-indexed position of layer.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module.

    Returns:
        Instantiated `HGRN2GatedDeltaNet` assigned to `layer.attention`.
    """
    d_model: int = model_config.hidden_size
    n_heads: int = getattr(model_config, "num_attention_heads", 12)
    n_layers: int = total_layers
    ratio: int = getattr(comp_config, "ratio", 3)
    new_net = HGRN2GatedDeltaNet(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ratio=ratio,
        config=model_config,
    )
    layer.attention = new_net
    return new_net
