"""Grouped-Query Attention (GQA).

Paper: https://arxiv.org/abs/2305.13245
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("gqa")
class GroupedQueryAttention(nn.Module):
    r"""Grouped-Query Attention (GQA) Mechanism (Ainslie et al., 2023).

    Interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA) by dividing
    $H_Q$ query heads into $G$ groups, where each group shares a single Key-Value head pair ($H_{KV} = G \le H_Q$):
    $$K_{\text{shared}} = \text{repeat\_interleave}(K, H_Q / H_{KV}, \text{dim}=\text{heads})$$
    $$\text{Attention}_{\text{GQA}}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\text{shared}}^\top}{\sqrt{d_k}} + M\right) V_{\text{shared}}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H_Q \cdot N^2 \cdot d_k)$
        - KV Cache Memory Reduction: Factor of $H_Q / H_{KV}$.

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, N])$.
    """

    def __init__(
        self,
        config: Any,
        num_kv_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        
        # Ensure num_kv_heads is an exact divisor of num_heads
        divisors = [d for d in range(1, self.num_heads + 1) if self.num_heads % d == 0]
        req_kv = getattr(config, "num_key_value_heads", num_kv_heads)
        if req_kv is not None and req_kv in divisors:
            self.num_kv_heads: int = req_kv
        else:
            # Pick a valid divisor <= num_heads // 2 (or 1)
            valid_divisors = [d for d in divisors if d <= max(1, self.num_heads // 2)]
            self.num_kv_heads: int = valid_divisors[-1] if valid_divisors else 1

        self.head_dim: int = config.hidden_size // self.num_heads
        self.hidden_size: int = config.hidden_size

        self.query: nn.Linear = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim
        )
        self.key: nn.Linear = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim
        )
        self.value: nn.Linear = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim
        )
        self.out_proj: nn.Linear = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size
        )
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

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Initialize GQA projections from standard full MHA module via group-averaging."""
        from framework.core.adapters import UniversalAdapter

        with torch.no_grad():
            try:
                w_q, b_q = UniversalAdapter.get_weight_bias(src, "q")
                self.query.weight.data.copy_(w_q)
                if b_q is not None and self.query.bias is not None:
                    self.query.bias.data.copy_(b_q)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

            try:
                w_o, b_o = UniversalAdapter.get_weight_bias(src, "o")
                self.out_proj.weight.data.copy_(w_o)
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.data.copy_(b_o)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

            repeat = self.num_heads // self.num_kv_heads
            for proj, target in [("k", self.key), ("v", self.value)]:
                try:
                    w, b = UniversalAdapter.get_weight_bias(src, proj)
                    # Reshape w: (num_heads, head_dim, hidden_size)
                    w_reshaped = w.view(self.num_heads, self.head_dim, self.hidden_size)
                    # Average heads across each group: (num_kv_heads, repeat, head_dim, hidden_size) -> (num_kv_heads, head_dim, hidden_size)
                    w_grouped = w_reshaped.view(self.num_kv_heads, repeat, self.head_dim, self.hidden_size).mean(dim=1)
                    target.weight.data.copy_(w_grouped.view(self.num_kv_heads * self.head_dim, self.hidden_size))
                    if b is not None and target.bias is not None:
                        b_reshaped = b.view(self.num_heads, self.head_dim)
                        b_grouped = b_reshaped.view(self.num_kv_heads, repeat, self.head_dim).mean(dim=1)
                        target.bias.data.copy_(b_grouped.view(self.num_kv_heads * self.head_dim))
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                    pass

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
        """Forward pass for Grouped-Query Attention with universal masking."""
        B, L, _ = hidden_states.shape
        q: torch.Tensor = self.query(hidden_states).view(
            B, L, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k: torch.Tensor = self.key(hidden_states).view(
            B, L, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v: torch.Tensor = self.value(hidden_states).view(
            B, L, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        repeat: int = self.num_heads // self.num_kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scale: float = self.head_dim**-0.5
        scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = apply_attention_mask(scores, attention_mask)

        attn: torch.Tensor = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        out: torch.Tensor = (
            torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, -1)
        )
        return self.out_proj(out), attn if output_attentions else None
