from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hgrn2_layer")
class HGRN2Layer(BaseAttention):
    r"""
    Hierarchically Gated Recurrent Network v2 (HGRN-2) Layer.

    Mathematical Mechanism:
        Implements element-wise linear recurrence with dynamic input-dependent forget and input gating:
        $$\begin{aligned}
        f_t &= \sigma(W_f x_t) \in (0, 1)^D \\
        i_t &= \sigma(W_i x_t) \in (0, 1)^D \\
        h_t &= \text{clip}\left(f_t \odot h_{t-1} + i_t \odot x_t, -C, C\right) \\
        y_t &= W_o h_t
        \end{aligned}$$
        where $C = 1.0$ is the residual state clipping threshold to ensure numerical stability.

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot D)$ — strictly linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot D)$ state memory.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        d_model: int,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        res_clip: float = 1.0,
    ) -> None:
        super().__init__(config, rank)
        self.d_model: int = d_model
        self.res_clip: float = res_clip

        self.forget_gate: nn.Linear = nn.Linear(d_model, d_model)
        self.input_gate: nn.Linear = nn.Linear(d_model, d_model)
        self.output: nn.Linear = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Sequential forward recurrence for HGRN-2.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Unused flag for interface compatibility.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, None)` with output tensor of shape `[B, L, D]`.
        """
        B, L, D = hidden_states.shape
        h = torch.zeros(B, D, device=hidden_states.device, dtype=hidden_states.dtype)
        outs = []
        for t in range(L):
            xt = hidden_states[:, t]
            f = torch.sigmoid(self.forget_gate(xt))
            i = torch.sigmoid(self.input_gate(xt))
            h = f * h + i * xt
            h = torch.clamp(h, -self.res_clip, self.res_clip)
            outs.append(self.output(h))
        out = torch.stack(outs, dim=1)
        return out, None


@ATTENTION_REGISTRY.register("hgrn2_layer")
def inject_hgrn2_layer(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> HGRN2Layer:
    r"""
    Injection hook to instantiate and place HGRN2Layer into a transformer layer.

    Args:
        layer: Transformer layer module to patch.
        model_config: Model configuration with `hidden_size`.
        comp_config: Compression configuration.
        layer_idx: Layer index in transformer stack.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module being replaced.

    Returns:
        Instantiated `HGRN2Layer` assigned to `layer.attention`.
    """
    d_model: int = model_config.hidden_size
    new_layer = HGRN2Layer(d_model=d_model, config=model_config)
    layer.attention = new_layer
    return new_layer
