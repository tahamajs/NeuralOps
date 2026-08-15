r"""
EAT: Efficient Attention Transformer with Progressive Token Pruning.

Implements a transformer block featuring standard multi-head self-attention, feed-forward sub-networks,
residual pre/post layer normalization, and optional progressive token pruning to reduce sequence length dynamically.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

from .progressive_token_pruning import ProgressiveTokenPruning


@ATTENTION_REGISTRY.register("eat")
class EATBlock(nn.Module):
    r"""
    Efficient Attention Transformer (EAT) Layer with Progressive Token Pruning.

    Mathematical Formulation:
        1. Multi-Head Self-Attention:
           $$A = \text{MHA}(X, X, X)$$
           $$X^{(1)} = \text{LayerNorm}(X + A)$$

        2. Position-wise Feed-Forward Network:
           $$F = W_2 \text{GELU}(W_1 X^{(1)} + b_1) + b_2$$
           $$X^{(2)} = \text{LayerNorm}(X^{(1)} + F)$$

        3. Optional Progressive Token Pruning:
           Given score function $S(X^{(2)})$, selects top-$k$ informative tokens $\mathcal{I}_{\text{keep}}$:
           $$X^{\text{pruned}} = X^{(2)}[:, \mathcal{I}_{\text{keep}}, :], \quad |\mathcal{I}_{\text{keep}}| = \lfloor \gamma \cdot L \rfloor$$

    Complexity:
        - Self-Attention: $\mathcal{O}(B \cdot L^2 \cdot D)$.
        - Feed-Forward: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{ffn}})$.
        - Pruning: $\mathcal{O}(B \cdot L \cdot D)$ reduction to length $L' \le L$.

    Tensor Dimensions:
        - Input: $[B, L, D]$
        - Pruned Output: $[B, L', D]$ where $L' \le L$
    """

    def __init__(
        self,
        d_model: Union[int, Any] = 768,
        n_heads: int = 12,
        prune_layer: Optional[ProgressiveTokenPruning] = None,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", n_heads))
        elif hasattr(d_model, "hidden_size"):
            config = d_model
            d_model = config.hidden_size
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.prune_layer: Optional[ProgressiveTokenPruning] = prune_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward execution pass for EATBlock.

        Args:
            hidden_states (torch.Tensor): Input activation tensor of shape $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor of shape $[B, L]$.
            token_indices (Optional[torch.Tensor]): Original token indices $[B, L]$ tracked across pruning layers.
            output_attentions (bool): Flag indicating whether to return attention weights.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed activations $[B, L', D]$ ($L' \le L$).
                - Updated token indices $[B, L']$ or None.
        """
        x = hidden_states
        attn_out, attn_weights = self.attn(
            x, x, x, key_padding_mask=(attention_mask == 0) if attention_mask is not None and attention_mask.dim() == 2 else None
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        if self.prune_layer is not None:
            x, new_indices = self.prune_layer(x)
            if token_indices is not None:
                token_indices = token_indices.gather(1, new_indices)
            return x, token_indices

        return x, token_indices if token_indices is not None else (attn_weights if output_attentions else None)
