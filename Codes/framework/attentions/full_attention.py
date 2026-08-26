from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("full")
class FullAttention(nn.Module):
    r"""Standard Multi-Head Scaled Dot-Product Attention (Vaswani et al., 2017).

    Computes pairwise token interactions across multiple representation subspaces:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d)$
        - Memory: $\mathcal{O}(B \cdot H \cdot N^2)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - attention_mask: $[B, N]$, $[B, 1, 1, N]$, or $[B, 1, N, N]$.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, N])$.
    """

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim

        self.query: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, config.hidden_size)
        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of scaled dot-product multi-head attention."""
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q: torch.Tensor = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
        k: torch.Tensor = self.key(hidden_states).view(B, N, H, d).transpose(1, 2)
        v: torch.Tensor = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)

        scale: float = d**-0.5
        scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) * scale

        scores = apply_attention_mask(scores, attention_mask)

        attn: torch.Tensor = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        context: torch.Tensor = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out: torch.Tensor = self.out_proj(context)

        return out, attn if output_attentions else None
