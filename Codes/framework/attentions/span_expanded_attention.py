r"""
Span-Expanded Attention with HyLoRA.

Extends attention receptive fields via fixed/cached key-value span expansion buffers,
complemented by parallel low-rank adaptation (HyLoRA) for parameter-efficient adaptation.

Mathematical Mechanism:
    Given queries $Q \in \mathbb{R}^{B \times H \times L \times d}$ and expanded span representations $(K, V) \in \mathbb{R}^{B \times H \times L_{\text{span}} \times d}$:
    1. **Span-Expanded Multi-Head Attention:**
       $$A = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) \in \mathbb{R}^{B \times H \times L \times L_{\text{span}}}$$
       $$O_{\text{attn}} = A V \in \mathbb{R}^{B \times H \times L \times d} \to \mathbb{R}^{B \times L \times D}$$
    2. **Parallel HyLoRA Residual Adaptation:**
       $$O_{\text{LoRA}} = X W_A W_B \in \mathbb{R}^{B \times L \times D}, \quad W_A \in \mathbb{R}^{D \times r}, \, W_B \in \mathbb{R}^{r \times D}$$
    3. **Fused Output Projection:**
       $$O = (O_{\text{attn}} + O_{\text{LoRA}}) W_O \in \mathbb{R}^{B \times L \times D}$$

Computational Complexity:
    - Span attention computation: $\mathcal{O}(B \cdot H \cdot L \cdot L_{\text{span}} \cdot d)$.
    - HyLoRA residual projection: $\mathcal{O}(B \cdot L \cdot D \cdot r)$.
    - Total per-layer complexity: $\mathcal{O}(B \cdot L \cdot (H \cdot L_{\text{span}} \cdot d + D \cdot r))$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("span_expanded")
class SpanExpandedAttention(BaseAttention):
    r"""Span-Expanded Attention module with parallel HyLoRA adaptation.

    Args:
        config: Transformer configuration specifying `hidden_size` and `num_attention_heads`.
        rank: Rank parameter (BaseAttention compatibility).
        expansion_span: Span cache window size.
        use_lora: Whether to incorporate parallel HyLoRA linear projections.
        lora_rank: Rank $r$ of the LoRA projection bottleneck.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        expansion_span: int = 1024,
        use_lora: bool = True,
        lora_rank: int = 8,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.dim // self.num_heads
        self.expansion_span: int = expansion_span
        self.use_lora: bool = use_lora

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

        if use_lora:
            self.lora_A = nn.Linear(self.dim, lora_rank, bias=False)
            self.lora_B = nn.Linear(lora_rank, self.dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute span-expanded attention forward pass.

        Args:
            hidden_states: Input activations `[Batch, SeqLen, HiddenDim]`.
            kv_cache: Optional tuple of `(k_cache, v_cache)` for expanded KV span.
            attention_mask: Optional attention mask `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head pruning tensor.
            output_attentions: Return explicit attention probabilities if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Attention matrix or None.
        """
        B, L, D = hidden_states.shape
        q = (
            self.W_q(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = k_cache.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            v = v_cache.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            k = (
                self.W_k(hidden_states)
                .view(B, L, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.W_v(hidden_states)
                .view(B, L, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

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
        if head_mask is not None:
            attn = attn * head_mask

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        if self.use_lora:
            lora_out = self.lora_B(self.lora_A(hidden_states))
            out = out + lora_out

        out = self.out_proj(out)

        if output_attentions:
            return out, attn
        return out, None
