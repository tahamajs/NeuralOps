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
        config: Any,
        comp_config: Any,
        layer_idx: int,
        num_layers: int,
        old_attn: Any,
    ) -> None:
        super().__init__()
        self.attn: nn.Module = old_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Forward activations through the wrapped attention module.

        Args:
            hidden_states: Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional mask `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head pruning tensor.
            output_attentions: Return explicit attention weights if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax probability matrix or None.
        """
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
    layer: Any,
    config: Any,
    comp_config: Any,
    layer_idx: int,
    num_layers: int,
    old_attn: Any,
) -> TokenCompressedAttention:
    r"""Bridge helper injecting TokenCompressedAttention into a transformer layer.

    Args:
        layer: Target transformer layer module.
        config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Index of the current layer.
        num_layers: Total number of layers.
        old_attn: Existing attention module to wrap.

    Returns:
        Instantiated :class:`TokenCompressedAttention` assigned to ``layer.attention``.
    """
    new_attn = TokenCompressedAttention(
        config, comp_config, layer_idx, num_layers, old_attn
    )
    layer.attention = new_attn
    return new_attn
