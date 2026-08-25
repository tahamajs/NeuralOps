import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class LoRALinear(nn.Module):
    r"""
    Low-Rank Adaptation (LoRA) Linear Layer Wrapper.

    Mathematical Mechanism:
        Freezes the pretrained weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
        and introduces trainable rank-$r$ residual factor matrices $A$ and $B$:
        $$y = W_0 x + \frac{\alpha}{r} B A x$$
        where $A \sim \mathcal{U}\left(-\frac{1}{\sqrt{d_{\text{in}}}}, \frac{1}{\sqrt{d_{\text{in}}}}\right)$ and $B = 0$.

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot N \cdot (d_{\text{in}} \cdot d_{\text{out}} + r \cdot (d_{\text{in}} + d_{\text{out}})))$
        - Space: $\mathcal{O}(r \cdot (d_{\text{in}} + d_{\text{out}}))$ trainable parameters.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original_layer: nn.Linear = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        self.rank: int = rank
        self.alpha: float = alpha
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.lora_A: nn.Parameter = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B: nn.Parameter = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight matrix including LoRA residual adapter."""
        return self.original_layer.weight + (self.lora_B @ self.lora_A) * (self.alpha / self.rank)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.original_layer.bias

    @property
    def in_features(self) -> int:
        return self.original_layer.in_features

    @property
    def out_features(self) -> int:
        return self.original_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        orig_out = self.original_layer(x)
        lora_out = F.linear(self.dropout(x), self.lora_A)  # (..., rank)
        lora_out = F.linear(lora_out, self.lora_B)  # (..., out_features)
        lora_out = lora_out * (self.alpha / self.rank)
        return orig_out + lora_out


@ATTENTION_REGISTRY.register("lora_attn")
class LoRAAttention(BaseAttention):
    r"""
    LoRA Parameter-Efficient Fine-Tuning Attention Module.

    Mathematical Mechanism:
        Replaces standard query, key, value, and output linear projections with LoRA adapters:
        $$\begin{aligned}
        Q &= \text{LoRA}_Q(X), \quad K = \text{LoRA}_K(X), \quad V = \text{LoRA}_V(X) \\
        S &= \frac{Q K^\top}{\sqrt{d_k}} + M_{\text{mask}} \\
        P &= \text{softmax}(S, \text{dim}=-1) \\
        O &= \text{LoRA}_O(P V)
        \end{aligned}$$

    Computational Complexity:
        - Attention Map: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$ time, $\mathcal{O}(B \cdot H \cdot N^2)$ space.
        - LoRA Projections: $\mathcal{O}(B \cdot N \cdot (D^2 + 2 D r))$ time.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, N, D]`
        - Output: Tuple `(output, attn_weights)` where `output` is `[B, N, D]`.
    """

    def __init__(
        self,
        model_config: Any,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(model_config, rank)
        if pretrained_attn is None:
            class MockAttn:
                def __init__(self, d):
                    self.q_lin = torch.nn.Linear(d, d)
                    self.k_lin = torch.nn.Linear(d, d)
                    self.v_lin = torch.nn.Linear(d, d)
                    self.out_lin = torch.nn.Linear(d, d)
            hidden_size = getattr(model_config, "hidden_size", 768)
            pretrained_attn = MockAttn(hidden_size)

        self.num_heads: int = getattr(model_config, "num_attention_heads", 12)
        hidden_size: int = getattr(model_config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = hidden_size
        self.alpha: float = alpha

        # Keep original layers frozen, wrapped in LoRALinear
        self.q_proj: LoRALinear = LoRALinear(pretrained_attn.q_lin, rank, alpha, dropout)
        self.k_proj: LoRALinear = LoRALinear(pretrained_attn.k_lin, rank, alpha, dropout)
        self.v_proj: LoRALinear = LoRALinear(pretrained_attn.v_lin, rank, alpha, dropout)
        self.out_proj: LoRALinear = LoRALinear(pretrained_attn.out_lin, rank, alpha, dropout)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward execution of LoRAAttention.

        Args:
            hidden_states: Input tensor of shape `[B, N, D]`.
            attention_mask: Optional mask tensor of shape `[B, N]` or `[B, 1, N, N]`.
            head_mask: Optional head masking tensor.
            output_attentions: Boolean flag to return attention map.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, attn_weights)` with output of shape `[B, N, D]`.
        """
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)

        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask.to(scores.dtype)) * -10000.0

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)

        return output, (attn_weights if output_attentions else None)


def inject_lora_attn(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> LoRAAttention:
    r"""
    Injection hook for LoRAAttention into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module to wrap with LoRA adapters.

    Returns:
        Instantiated `LoRAAttention` assigned to `layer.attention`.
    """
    rank: int = getattr(comp_config, "lora_rank", 8)
    alpha: float = getattr(comp_config, "lora_alpha", 16.0)
    dropout: float = getattr(comp_config, "dropout", 0.1)
    new_attn = LoRAAttention(
        model_config=model_config,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
