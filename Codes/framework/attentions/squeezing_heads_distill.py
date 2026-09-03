r"""
Squeezing Heads Attention Distillation Layer.

Multi-head attention wrapper designed for knowledge distillation and head capacity compression experiments.

Mathematical Mechanism:
    Given input $X \in \mathbb{R}^{B \times L \times D}$:
    1. **Multi-Head Projections:**
       $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \in \mathbb{R}^{B \times H \times L \times d}$$
    2. **Scaled Dot-Product Attention:**
       $$A = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) \in \mathbb{R}^{B \times H \times L \times L}$$
       $$O = (A V) W_O \in \mathbb{R}^{B \times L \times D}$$

Computational Complexity:
    - Linear projections: $\mathcal{O}(B \cdot L \cdot D^2)$.
    - Attention matrix products: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d)$.
    - Total per-layer complexity: $\mathcal{O}(B \cdot L \cdot D^2 + B \cdot H \cdot L^2 \cdot d)$.
"""

from typing import Any, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("squeezing_heads_distiller")
class SqueezingHeadsDistiller(nn.Module):
    r"""Attention layer wrapper for multi-head distillation and compression studies.

    Args:
        config: Transformer configuration specifying `hidden_size`, `num_attention_heads`, and `dropout`.
        rank: Optional compression rank parameter.
    """

    def __init__(self, config: Any, rank: Optional[int] = None) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        dropout_p: float = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
        )

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout_p)

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
        r"""Forward execution of distillation attention layer.

        Args:
            hidden_states: Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional mask `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head pruning mask.
            output_attentions: Return explicit attention probabilities if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax probability matrix or None.
        """
        x = hidden_states
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k   = self.k_proj(x).view(B, L, getattr(self, "num_kv_heads", getattr(self, "num_kv_heads", self.num_heads)), self.head_dim).transpose(1, 2)
        k = repeat_kv(k, getattr(self, 'kv_repeat', 1))
        v   = self.v_proj(x).view(B, L, getattr(self, "num_kv_heads", getattr(self, "num_kv_heads", self.num_heads)), self.head_dim).transpose(1, 2)
        v = repeat_kv(v, getattr(self, 'kv_repeat', 1))

        scale = self.head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask) * -10000.0
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1)
            else:
                scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        if head_mask is not None:
            attn = attn * head_mask

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(out)

        if output_attentions:
            return output, attn
        return output, None
