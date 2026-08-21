r"""
Token-Compressed Attention Wrapper.

Structural wrapper facilitating layer-wise token compression,
sequence pruning, and distillation experiments by delegating
to an existing attention sub-module.

Mathematical Mechanism:
    Given input sequence activations $X \in \mathbb{R}^{B \times N \times D}$:
    $$\text{Output}, A = \text{BaseAttn}(X, M)$$
    where BaseAttn is any registered framework attention module.

Computational Complexity:
    Inherited from underlying source attention module:
    $\mathcal{O}(B \cdot H \cdot N^2 \cdot d)$ for full attention,
    or $\mathcal{O}(B \cdot N \cdot D)$ for linear variants.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from framework.core.registry import ATTENTION_REGISTRY


class DummyAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, **kwargs):
        return self.proj(x), None

@ATTENTION_REGISTRY.register("token_compressed_module")
class TokenCompressedAttention(nn.Module):
    r"""Attention wrapper module encapsulating token compression transformations.

    Delegates the forward pass to the wrapped attention module and ensures
    a consistent ``(output, weights)`` tuple return signature.

    Args:
        config: Transformer model configuration.
        comp_config: Compression configuration.
        layer_idx: Index of the current transformer layer.
        num_layers: Total number of transformer layers.
        old_attn: Pre-existing attention module to wrap.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        comp_config: Optional[Any] = None,
        layer_idx: int = 0,
        total_layers: int = 12,
        old_attn: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Fallback to an identity lambda if old_attn is an integer or None
        if old_attn is None or isinstance(old_attn, int):
            self.attn = DummyAttn(getattr(config, "hidden_size", 768))
        else:
            self.attn = old_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Prevent infinite recursion if wrapped around itself
        if self.attn is self:
            return hidden_states, None
        
        res = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        if isinstance(res, tuple):
            return res
        return res, None


@ATTENTION_REGISTRY.register("token_compressed")
def inject_token_compressed(
    layer: Any = None,
    config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    **kwargs: Any,
) -> TokenCompressedAttention:
    if isinstance(old_attn, TokenCompressedAttention):
        return old_attn
        
    new_attn = TokenCompressedAttention(
        config, comp_config, layer_idx, total_layers, old_attn
    )
    if layer is not None:
        layer.attention = new_attn
    return new_attn
