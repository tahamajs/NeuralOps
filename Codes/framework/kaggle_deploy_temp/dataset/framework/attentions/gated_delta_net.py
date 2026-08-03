r"""
Gated DeltaNet & Hierarchical Gated Recurrent Network (HGRN2) Attention Block.

Combines input-dependent forgetting gates, input relevance modulation, bounded recurrent state tracking,
and optional full-attention bypass routing.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


class HGRN2Layer(nn.Module):
    r"""
    Hierarchically Gated Recurrent Network V2 (HGRN2) Layer.

    Mathematical Formulation:
        For input token activations $x_t \in \mathbb{R}^D$ at sequence step $t$:
        1. Gating Signals:
           $$f_t = \sigma(W_f x_t + b_f) \in [0, 1]^D \quad \text{(Forget Gate)}$$
           $$i_t = \sigma(W_i x_t + b_i) \in [0, 1]^D \quad \text{(Input Gate)}$$

        2. Recurrent Hidden State with Clamping:
           $$h_t = \text{clamp}\left( f_t \odot h_{t-1} + i_t \odot x_t, -\tau_{\text{clip}}, \tau_{\text{clip}} \right), \quad h_0 = \mathbf{0}$$

        3. Output Transformation:
           $$y_t = W_o h_t + b_o$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot D^2)$ linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot D)$ recurrent state footprint.

    Tensor Dimensions:
        - Input: $[B, L, D]$
        - Hidden State $h_t$: $[B, D]$
        - Output: $[B, L, D]$
    """

    def __init__(self, d_model: int, res_clip: float = 1.0) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.forget_gate = nn.Linear(d_model, d_model)
        self.input_gate = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.res_clip: float = res_clip

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Forward recurrence pass for HGRN2.

        Args:
            x (torch.Tensor): Input sequence activations $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor $[B, L]$.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            torch.Tensor: Recurrent transformed activations $[B, L, D]$.
        """
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            xt = x[:, t]
            f = torch.sigmoid(self.forget_gate(xt))
            i = torch.sigmoid(self.input_gate(xt))
            h = f * h + i * xt
            h = torch.clamp(h, -self.res_clip, self.res_clip)
            outs.append(self.output(h))
        return torch.stack(outs, dim=1)


class GatedDeltaNetBlock(nn.Module):
    r"""
    Hybrid Gated DeltaNet Block with Optional Multi-Head Attention Bypass.

    Mathematical Formulation:
        $$y = \begin{cases}
            \text{LayerNorm}(x + \text{MHA}(x, x, x)), & \text{if } \text{use\_full\_attention} \\
            \text{LayerNorm}(x + \text{HGRN2}(x)), & \text{otherwise}
        \end{cases}$$

    Complexity:
        - Full Attention Mode: $\mathcal{O}(B \cdot L^2 \cdot D)$.
        - Gated DeltaNet Mode: $\mathcal{O}(B \cdot L \cdot D^2)$ strictly linear in $L$.

    Tensor Dimensions:
        - Input: $[B, L, D]$
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

        self.hgrn = HGRN2Layer(d_model)
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
            hidden_states (torch.Tensor): Input representation $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor $[B, L]$.
            output_attentions (bool): Whether to return attention maps.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed activations $[B, L, D]$.
                - None.
        """
        x = hidden_states
        if self.use_full_attention and self.attn is not None:
            key_padding_mask = (
                (attention_mask == 0)
                if attention_mask is not None and attention_mask.dim() == 2
                else None
            )
            attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
            out = self.norm(x + attn_out)
        else:
            hgrn_out = self.hgrn(x, attention_mask=attention_mask)
            out = self.norm(x + hgrn_out)
        return out, None


@ATTENTION_REGISTRY.register("gated_delta_net")
def inject_gated_delta_net(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> GatedDeltaNetBlock:
    r"""
    Injects GatedDeltaNet module into a transformer layer.

    Args:
        layer (nn.Module): Host layer module.
        model_config (Any): Model configuration parameters.
        comp_config (Any): Compression hyperparameter configuration.
        layer_idx (int): Current layer index.
        total_layers (int): Total transformer layers.
        old_attn (Optional[nn.Module]): Pretrained attention layer.

    Returns:
        GatedDeltaNetBlock: Initialized GatedDeltaNet layer.
    """
    d_model = model_config.hidden_size
    n_heads = model_config.num_attention_heads
    use_full = getattr(comp_config, "use_full_attention", False)
    new_block = GatedDeltaNetBlock(d_model, n_heads, use_full)
    layer.attention = new_block
    return new_block
