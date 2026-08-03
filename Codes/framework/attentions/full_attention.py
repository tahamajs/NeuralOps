from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        - attention_mask: $[B, N]$ or $[B, 1, 1, N]$ binary or additive attention mask.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, N])$ representing projected contextualized states and attention maps.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.q_proj: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.k_proj: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.v_proj: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of scaled dot-product multi-head attention.

        Args:
            hidden_states (torch.Tensor): Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Binary or additive mask of shape `[Batch, SeqLen]` or `[Batch, 1, 1, SeqLen]`.
            **kwargs (Any): Additional model configuration parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Contextualized output tensor of shape `[Batch, SeqLen, HiddenDim]`.
                - Optional attention weights of shape `[Batch, NumHeads, SeqLen, SeqLen]`.
        """
        B, N, D = hidden_states.shape
        q: torch.Tensor = (
            self.q_proj(hidden_states)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k: torch.Tensor = (
            self.k_proj(hidden_states)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v: torch.Tensor = (
            self.v_proj(hidden_states)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9
        attn: torch.Tensor = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out: torch.Tensor = (
            torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        )
        return self.out_proj(out), None
