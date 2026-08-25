r"""
Universal Rotary Position Embedding Attention (UnifiedRoPE).

Applies consistent rotary positional encoding across attention query-key representations
in transformer architectures to guarantee positional consistency and relative position awareness.

Mathematical Formulation:
    1. Input projections:
       $$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$
    2. Rotary position embedding application:
       $$\mathbf{Q}_{\text{rot}} = \text{RoPE}(\mathbf{Q}), \quad \mathbf{K}_{\text{rot}} = \text{RoPE}(\mathbf{K})$$
    3. Scaled dot-product attention:
       $$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{rot}} \mathbf{K}_{\text{rot}}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$
       $$\mathbf{Y} = \mathbf{A} \mathbf{W}_O$$

Complexity:
    - Time: $\mathcal{O}(L^2 \cdot d + L \cdot d)$
    - Space: $\mathcal{O}(L \cdot d)$ for rotary frequency buffers.
"""

from __future__ import annotations
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("compression_unified_rope")
@ATTENTION_REGISTRY.register("unified_rope")
class UnifiedRoPE(BaseAttention):
    r"""Universal Rotary Position Embedding Attention module.

    Subclasses :class:`BaseAttention` and applies rotary position embeddings to query and
    key representations prior to dot-product attention computation.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        max_seq_len: int = 4096,
        base: int = 10000,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, rank=rank)

        if dim is None:
            dim = getattr(config, "hidden_size", getattr(config, "d_model", 768)) if config is not None else 768
        if num_heads is None:
            num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", 12)) if config is not None else 12

        self.hidden_size: int = dim
        self.num_heads: int = max(1, num_heads)
        self.head_dim: int = self.hidden_size // self.num_heads
        self.max_seq_len: int = max_seq_len
        self.base: int = base
        self.scale: float = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.dropout = nn.Dropout(dropout)

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _build_cache(self, seq_len: int) -> None:
        """Precomputes cosine and sine frequency tables up to seq_len."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates second half of hidden dimension for RoPE."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Applies RoPE rotation: [B, H, L, D_head]."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)  # [1, 1, L, D_head]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)

    def copy_weights(self, old_attn: Any) -> None:
        """Copies pretrained weights from standard transformer attention layers."""
        if old_attn is None:
            return
        with torch.no_grad():
            for src_name, dst_mod in [
                ("q_lin", self.q_proj), ("query", self.q_proj), ("q_proj", self.q_proj), ("q", self.q_proj),
                ("k_lin", self.k_proj), ("key", self.k_proj), ("k_proj", self.k_proj), ("k", self.k_proj),
                ("v_lin", self.v_proj), ("value", self.v_proj), ("v_proj", self.v_proj), ("v", self.v_proj),
                ("out_lin", self.out_proj), ("dense", self.out_proj), ("out_proj", self.out_proj), ("o", self.out_proj),
            ]:
                if hasattr(old_attn, src_name):
                    src_mod = getattr(old_attn, src_name)
                    if hasattr(src_mod, "weight") and src_mod.weight.shape == dst_mod.weight.shape:
                        dst_mod.weight.copy_(src_mod.weight)
                    if hasattr(src_mod, "bias") and src_mod.bias is not None and dst_mod.bias is not None:
                        dst_mod.bias.copy_(src_mod.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass computing RoPE attention, returning (output, attention_weights)."""
        if hidden_states.dim() == 4:
            B, H, L, Dh = hidden_states.shape
            hidden_states = hidden_states.view(B, L, H * Dh)

        B, L, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q_rot = self._apply_rope(q, L)
        k_rot = self._apply_rope(k, L)

        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_bias = (1.0 - attention_mask[:, None, None, :].to(scores.dtype)) * -10000.0
                scores = scores + attn_bias
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(context)

        return output, (attn_weights if output_attentions else None)
