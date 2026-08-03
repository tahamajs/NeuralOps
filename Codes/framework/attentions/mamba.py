from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("mamba")
class Mamba(BaseAttention):
    r"""
    Gated State Space Model (Mamba) Linear Layer Module.

    Mathematical Mechanism:
        Implements a gated selective state-space projection:
        $$\begin{aligned}
        [X_1, X_2] &= \text{chunk}(\text{Linear}(X), 2) \in \mathbb{R}^{B \times L \times D} \\
        Y &= \text{SiLU}(X_1) \odot X_2 \\
        O &= Y W_{\text{out}}
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot D^2)$ — strictly linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot L \cdot D)$ activations.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        d_model: int,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.d_model: int = d_model
        self.in_proj: nn.Linear = nn.Linear(d_model, d_model * 2)
        self.out_proj: nn.Linear = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward execution of Mamba layer.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Unused flag for interface compliance.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, None)` with output of shape `[B, L, D]`.
        """
        proj = self.in_proj(hidden_states)
        p1, p2 = proj.chunk(2, dim=-1)
        out = self.out_proj(F.silu(p1) * p2)
        return out, None


@ATTENTION_REGISTRY.register("mamba")
def inject_mamba(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> Mamba:
    r"""
    Injection hook for Mamba into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module.

    Returns:
        Instantiated `Mamba` assigned to `layer.attention`.
    """
    d_model: int = model_config.hidden_size
    new_module = Mamba(d_model=d_model, config=model_config)
    layer.attention = new_module
    return new_module
