r"""
Adaptive Span Attention Module.

This module implements Adaptive Attention Span mechanisms based on Sukhbaatar et al.
("Adaptive Attention Span in Transformers", NeurIPS 2019). It allows individual attention
heads to dynamically adjust their context window (span) during inference/training.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


class AdaptiveSpanAttention(nn.Module):
    r"""
    Adaptive Span Attention layer for Transformer architectures (e.g., DistilBERT).

    Each attention head $h \in \{1, \dots, H\}$ learns an individual receptive field size (span)
    $S_h \in [0, S_{\max}]$ parametrized dynamically via a sigmoid gating parameter:
    $$S_h = \sigma(\theta_h) \cdot S_{\max}$$
    where $\theta_h \in \mathbb{R}$ is a learnable logit and $S_{\max}$ denotes the maximum span.

    The relative positional distance between query index $i$ and key index $j$ is computed as:
    $$\Delta_{ij} = |i - j|$$

    A soft or hard masking function $M^{(h)} \in \mathbb{R}^{N \times N}$ is applied to the scaled dot-product:
    $$M^{(h)}_{ij} = \begin{cases} 0 & \text{if } \Delta_{ij} \le S_h \\ -\infty & \text{otherwise} \end{cases}$$

    The resulting attention output is computed as:
    $$\text{Attn}^{(h)}(Q_h, K_h, V_h) = \text{Softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_h}} + M^{(h)} + M_{\text{pad}}\right) V_h$$

    Complexity:
        - Time Complexity: $\mathcal{O}(B \cdot H \cdot N^2 + B \cdot N \cdot D)$ where $N$ is sequence length,
          $H$ is head count, $D$ is model hidden dimension, and $d_h = D / H$.
        - Space/Memory Complexity: $\mathcal{O}(B \cdot H \cdot N^2)$ for storing the dense attention logit matrix.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, $D$ is hidden dimension.
        - `attention_mask`: $[B, N]$ or $[B, 1, 1, N]$.
        - Projected Query/Key/Value: $[B, H, N, d_h]$.
        - Attention Scores / Weights: $[B, H, N, N]$.
        - Output: $[B, N, D]$.
    """

    def __init__(
        self,
        model_config: Any,
        rank: int = 32,
        dropout: float = 0.1,
        max_span: int = 128,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero-shot initialization")

        self.num_heads: int = model_config.num_attention_heads
        self.head_dim: int = model_config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = model_config.hidden_size
        self.max_span: int = max_span

        # Projections (copied from pretrained)
        self.q_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)

        # Copy pretrained weights
        with torch.no_grad():
            self.q_proj.weight.data.copy_(pretrained_attn.q_lin.weight.data)
            self.q_proj.bias.data.copy_(pretrained_attn.q_lin.bias.data)
            self.k_proj.weight.data.copy_(pretrained_attn.k_lin.weight.data)
            self.k_proj.bias.data.copy_(pretrained_attn.k_lin.bias.data)
            self.v_proj.weight.data.copy_(pretrained_attn.v_lin.weight.data)
            self.v_proj.bias.data.copy_(pretrained_attn.v_lin.bias.data)
            self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight.data)
            self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias.data)

        # Learnable per-head span parameters (logits)
        self.span_logits = nn.Parameter(torch.full((self.num_heads,), float(max_span)))
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, x: torch.Tensor, B: int) -> torch.Tensor:
        r"""
        Reshapes linear projection $[B, N, H \cdot d_h]$ into multi-head tensor $[B, H, N, d_h]$.

        Args:
            x (torch.Tensor): Projected tensor of shape $[B, N, H \cdot d_h]$.
            B (int): Batch size dimension.

        Returns:
            torch.Tensor: Multi-head formatted tensor of shape $[B, H, N, d_h]$.
        """
        return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for Adaptive Span Attention.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Binary or float mask of shape $[B, N]$ or $[B, 1, 1, N]$.
            head_mask (Optional[torch.Tensor]): Head mask tensor of shape $[H]$ or $[B, H, N, N]$.
            output_attentions (bool): Whether to return computed attention probabilities.
            **kwargs (Any): Additional model-specific keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output tensor of shape $[B, N, D]$.
                - Attention weights of shape $[B, H, N, N]$ if `output_attentions=True`, else None.
        """
        B, N, _ = hidden_states.shape
        _H, D = self.num_heads, self.head_dim

        Q = self._reshape(self.q_proj(hidden_states), B)  # (B, H, N, D)
        K = self._reshape(self.k_proj(hidden_states), B)
        V = self._reshape(self.v_proj(hidden_states), B)

        # Compute raw scores
        scale = D**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)

        # Adaptive span mask (soft, differentiable)
        span = torch.sigmoid(self.span_logits) * self.max_span  # (H,)
        pos = torch.arange(N, device=Q.device).float()
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (N, N)
        # Span mask: 1 if within span, 0 otherwise
        mask_span = (dist <= span.unsqueeze(1).unsqueeze(2)).float()
        # Apply mask (set scores outside span to -inf)
        scores = scores.masked_fill(mask_span == 0, -10000.0)

        # Standard attention masking (padding)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]  # (B, 1, 1, N)
            else:
                mask = attention_mask
            # Convert boolean mask to float (True -> 1.0, False -> 0.0)
            mask_float = mask.float()
            scores = scores + (1.0 - mask_float) * -10000.0

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)

        return output, attn_weights if output_attentions else None


# ----------------------------------------------------------------------
# Injection function – matches the framework's required signature
# ----------------------------------------------------------------------
@ATTENTION_REGISTRY.register("adaptive_span")
def inject_adaptive_span(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> AdaptiveSpanAttention:
    r"""
    Replaces standard Multi-Head Attention in a Transformer layer with AdaptiveSpanAttention.

    Args:
        layer (nn.Module): Transformer layer to inject attention into.
        model_config (Any): HuggingFace or architecture configuration object.
        comp_config (Any): Compression/experimental configuration object.
        layer_idx (int): Current layer index (0-indexed).
        total_layers (int): Total number of layers in the host network.
        old_attn (Optional[nn.Module]): Pretrained attention module whose projection weights are copied.

    Returns:
        AdaptiveSpanAttention: The newly initialized and injected attention module.
    """
    rank = getattr(comp_config, "adaptive_span_rank", 32)
    dropout = getattr(comp_config, "dropout", 0.1)
    max_span = getattr(comp_config, "max_span", 128)
    new_attn = AdaptiveSpanAttention(
        model_config=model_config,
        rank=rank,
        dropout=dropout,
        max_span=max_span,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
