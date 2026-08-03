from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hgrn2_outer")
class HGRN2OuterBlock(BaseAttention):
    r"""
    HGRN-2 Outer-Product Recurrent Linear Attention Block.

    Mathematical Mechanism:
        Updates a 2D associative recurrent matrix state $S_t \in \mathbb{R}^{D \times D}$ via convex
        interpolation between previous memory and current key-value outer product:
        $$\begin{aligned}
        f_t &= \sigma(W_f x_t) \in (0, 1)^D \\
        k_t &= W_k x_t \in \mathbb{R}^D, \quad v_t = W_v x_t \in \mathbb{R}^D \\
        S_t &= f_t \odot S_{t-1} + (1 - f_t) \odot (k_t \otimes v_t) \\
        y_t &= S_t x_t \in \mathbb{R}^D \\
        Y &= W_{\text{out}} [y_1, y_2, \dots, y_L]
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot D^2)$ — strictly linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot D^2)$ state matrix storage per sequence.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        dim: int,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = dim
        self.W_f: nn.Linear = nn.Linear(dim, dim)
        self.W_k: nn.Linear = nn.Linear(dim, dim)
        self.W_v: nn.Linear = nn.Linear(dim, dim)
        self.out_proj: nn.Linear = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        state: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Sequential forward recurrence over outer-product memory state.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Unused flag for interface compatibility.
            state: Optional initial recurrent state `[B, D, D]`.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, None)` with output of shape `[B, L, D]`.
        """
        B, L, D = hidden_states.shape
        if state is None:
            state = torch.zeros(B, D, D, device=hidden_states.device, dtype=hidden_states.dtype)
        outputs = []
        for t in range(L):
            xt = hidden_states[:, t]
            f = torch.sigmoid(self.W_f(xt))
            k = self.W_k(xt)
            v = self.W_v(xt)
            state = f.unsqueeze(-1) * state + (1.0 - f.unsqueeze(-1)) * (
                k.unsqueeze(-1) @ v.unsqueeze(1)
            )
            yt = (state @ xt.unsqueeze(-1)).squeeze(-1)
            outputs.append(yt)
        out = torch.stack(outputs, dim=1)
        return self.out_proj(out), None


@ATTENTION_REGISTRY.register("hgrn2_outer")
def inject_hgrn2_outer(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> HGRN2OuterBlock:
    r"""
    Injection hook to instantiate and place HGRN2OuterBlock into a transformer layer.

    Args:
        layer: Transformer layer module to patch.
        model_config: Model configuration with `hidden_size`.
        comp_config: Compression configuration.
        layer_idx: Layer index in transformer stack.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module being replaced.

    Returns:
        Instantiated `HGRN2OuterBlock` assigned to `layer.attention`.
    """
    dim: int = model_config.hidden_size
    new_block = HGRN2OuterBlock(dim=dim, config=model_config)
    layer.attention = new_block
    return new_block
