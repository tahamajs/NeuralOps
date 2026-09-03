r"""
Token-Compressed Attention Wrapper (tt.py).

A minimal delegate wrapper around an existing attention sub-module to
enable transparent injection into transformer layer slots.

Mathematical Mechanism:
    Delegates to the wrapped sub-module:
    $$\text{Output} = \text{BaseAttn}(X, M)$$

Computational Complexity:
    Inherits the complexity of the wrapped attention module.
"""

from typing import Any, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn

from framework.core.registry import ATTENTION_REGISTRY


class TokenCompressedAttention(nn.Module):
    r"""Minimal wrapper delegating to an existing attention sub-module.

    Args:
        config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Index of current transformer layer.
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
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Delegate forward pass to wrapped attention module.

        Args:
            hidden_states: Input tensor `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (output tensor, None or attention weights).
        """
        res = self.attn(hidden_states, attention_mask=attention_mask, **kwargs)
        if isinstance(res, tuple):
            return res[0], (res[1] if len(res) > 1 else None)
        return res, None


# token_compressed handled by token_compressed.py
def inject_token_compressed(
    layer: Any,
    config: Any,
    comp_config: Any,
    layer_idx: int,
    num_layers: int,
    old_attn: Any,
) -> "TokenCompressedAttention":
    r"""Inject TokenCompressedAttention into a transformer layer.

    Returns:
        The newly instantiated :class:`TokenCompressedAttention`.
    """
    new_attn = TokenCompressedAttention(
        config, comp_config, layer_idx, num_layers, old_attn
    )
    layer.attention = new_attn
    return new_attn
