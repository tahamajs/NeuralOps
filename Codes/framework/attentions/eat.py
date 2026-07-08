import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

from .progressive_token_pruning import ProgressiveTokenPruning


@ATTENTION_REGISTRY.register("eat")
class EATBlock(nn.Module):
    """
    Efficient Attention Transformer block.
    Standard MultiheadAttention + FFN, with optional progressive pruning
    after the feed‑forward sub‑layer.
    """

    def __init__(
        self, d_model: int, n_heads: int, prune_layer: ProgressiveTokenPruning = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.prune_layer = prune_layer  # optional token pruner

    def forward(self, x: torch.Tensor, token_indices: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, hidden)
            token_indices: (batch, seq_len) original token ids – used only if pruning
        Returns:
            out: (batch, seq_len, hidden)
            updated_token_indices: if pruning was applied, reduced indices
        """
        # Multi‑head attention – no external mask for simplicity; you can add if needed
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        # Optional progressive pruning
        if self.prune_layer is not None:
            x, new_indices = self.prune_layer(x)
            if token_indices is not None:
                token_indices = token_indices.gather(1, new_indices)
            return x, token_indices
        return x, token_indices
