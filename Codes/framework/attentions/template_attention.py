r"""
Template Attention Mechanism.

Customizable template module demonstrating standard registry integration, base attention inheritance,
and optional low-rank latent modulation mechanisms.

Mathematical Mechanism:
    Given input sequence activations $X \in \mathbb{R}^{B \times N \times D}$:
    1. **Multi-Head Projections:**
       $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \in \mathbb{R}^{B \times H \times N \times d}$$
    2. **Optional Global Low-Rank Modulation:**
       $$h_{\text{pool}} = \frac{1}{N} \sum_{i=1}^N X_{:, i, :} \in \mathbb{R}^{B \times D}$$
       $$\gamma = \frac{1}{r} \sum_{j=1}^r \sigma(h_{\text{pool}} W_{\text{rank}})_j \in \mathbb{R}^{B \times 1 \times 1 \times 1}$$
    3. **Scaled Dot-Product Attention:**
       $$S = \left(\frac{Q K^T}{\sqrt{d}}\right) \odot \gamma + M \in \mathbb{R}^{B \times H \times N \times N}$$
       $$A = \operatorname{Softmax}(S), \quad O = (A V) W_O \in \mathbb{R}^{B \times N \times D}$$

Computational Complexity:
    - Linear projections: $\mathcal{O}(B \cdot N \cdot D^2)$.
    - Attention matrix products: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d)$.
    - Rank modulation (if enabled): $\mathcal{O}(B \cdot D \cdot r)$.
    - Total per-layer complexity: $\mathcal{O}(B \cdot N \cdot D^2 + B \cdot H \cdot N^2 \cdot d)$.
"""

from typing import Any, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("your_attention_name", supports_causal=True, supports_gqa=True)
class YourCustomAttention(BaseAttention):
    r"""Template attention layer with optional low-rank pooling modulation.

    Args:
        config: Transformer configuration specifying `hidden_size`, `num_attention_heads`, and `dropout`.
        rank: Optional compression bottleneck rank $r$.
        **kwargs: Extensible hyperparameter arguments.
    """

    supports_gqa: bool = True

    def __init__(self, config: Any, rank: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(config, rank)
        self.hidden_size: int = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads: int = getattr(config, "num_attention_heads", getattr(config, "num_heads", 4))
        self.head_dim: int = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads: int = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", self.num_heads))
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.dropout: float = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
        )

        self.custom_param: float = float(kwargs.get("custom_param", 0.1))

        self.query = nn.Linear(self.hidden_size, getattr(self, "q_out", self.num_heads * self.head_dim))
        self.key = nn.Linear(self.hidden_size, self.kv_head_size)
        self.value = nn.Linear(self.hidden_size, self.kv_head_size)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        if rank is not None and rank > 0:
            self.rank_proj: Optional[nn.Linear] = nn.Linear(self.hidden_size, rank, bias=False)
        else:
            self.rank_proj = None

        self._init_weights()

    def _init_weights(self) -> None:
        r"""Initialize projection layers with Xavier uniform weights and zero biases."""
        for module in [self.query, self.key, self.value, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _reshape(self, x: torch.Tensor, batch_size: int, heads: int) -> torch.Tensor:
        r"""Reshape `[Batch, SeqLen, Heads * HeadDim]` to `[Batch, Heads, SeqLen, HeadDim]`."""
        x = x.view(batch_size, -1, heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

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
        r"""Execute attention forward pass.

        Args:
            hidden_states: Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head pruning tensor.
            output_attentions: Return explicit attention probabilities if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax probability matrix or None.
        """
        batch_size, seq_len, _ = hidden_states.shape

        Q = self._reshape(self.query(hidden_states), batch_size, self.num_heads)
        K = self._reshape(self.key(hidden_states), batch_size, self.num_kv_heads)
        V = self._reshape(self.value(hidden_states), batch_size, self.num_kv_heads)
        if self.kv_repeat > 1:
            K = K.repeat_interleave(self.kv_repeat, dim=1)
            V = V.repeat_interleave(self.kv_repeat, dim=1)

        scale = self.head_dim**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if self.rank_proj is not None:
            H_pool = hidden_states.mean(dim=1)
            H_rank = self.rank_proj(H_pool)
            modulation = torch.sigmoid(H_rank).mean(dim=-1, keepdim=True)
            modulation = modulation.view(batch_size, 1, 1, 1)
            scores = scores * modulation

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask) * -10000.0
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1)
            else:
                scores = scores + attention_mask

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        context = torch.matmul(attn_probs, V)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        output = self.out_proj(context)

        if output_attentions:
            return output, attn_probs
        return output, None

    @property
    def complexity(self) -> str:
        r"""Return asymptotic time complexity description."""
        return (
            "quadratic"
            if self.rank_proj is None
            else "quadratic (with rank modulation)"
        )

    def extra_repr(self) -> str:
        r"""Return string representation of module configuration."""
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, rank={self.rank}"
