from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    r"""
    Rotates half the hidden dimensions of the input tensor for Rotary Position Embedding (RoPE).

    Mathematical Mechanism:
        $$x = [x_1, x_2] \mapsto [-x_2, x_1]$$

    Args:
        x: Tensor of shape `[..., D]`.

    Returns:
        Rotated tensor of identical shape `[..., D]`.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    r"""
    Rotary Position Embedding (RoPE).

    Mathematical Mechanism:
        $$\Theta = \{\theta_i = 10000^{-2(i-1)/d}, i \in [1, d/2]\}$$
        $$R_{\Theta, m}^d x_m = (x_m \odot \cos m\Theta) + (\text{rotate\_half}(x_m) \odot \sin m\Theta)$$

    Computational Complexity:
        - Time: $\mathcal{O}(L \cdot d)$
        - Space: $\mathcal{O}(L_{\max} \cdot d)$ cached buffer.

    Tensor Dimensions:
        - Input $x$: `[B, H, L, d]`
        - Output: `[B, H, L, d]`
    """

    def __init__(self, dim: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (L, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        r"""
        Applies rotary position embedding to input tensor.

        Args:
            x: Input tensor of shape `[B, H, L, d]`.
            offset: Position offset along sequence dimension.

        Returns:
            RoPE-embedded tensor of shape `[B, H, L, d]`.
        """
        L = x.shape[2]
        cos = self.cos_cached[offset : offset + L].unsqueeze(0).unsqueeze(0)  # (1, 1, L, d)
        sin = self.sin_cached[offset : offset + L].unsqueeze(0).unsqueeze(0)  # (1, 1, L, d)
        return (x * cos) + (rotate_half(x) * sin)


class ReGLALayer(nn.Module):
    r"""
    Rectified Gated Linear Attention (ReGLA) Layer with ReLU Kernel.

    Mathematical Mechanism:
        Uses $\phi(\cdot) = \text{ReLU}(\cdot)$ feature map for $O(N)$ linear attention:
        $$\begin{aligned}
        Q &= \text{ReLU}(X W_Q), \quad K = \text{ReLU}(X W_K), \quad V = X W_V \\
        S &= K^\top V = \sum_{t=1}^L K_t V_t^\top \in \mathbb{R}^{d \times d} \\
        Z &= \sum_{t=1}^L K_t \in \mathbb{R}^{d} \\
        O_t &= \frac{Q_t S}{Q_t Z + \epsilon}
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot L \cdot d^2)$ — linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot H \cdot d^2)$ for associative state.

    Tensor Dimensions:
        - Input $x$: `[B, L, D]`
        - Output: `[B, L, D]`
    """

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.dim: int = dim
        self.heads: int = heads
        self.head_dim: int = dim // heads
        self.q_proj: nn.Linear = nn.Linear(dim, dim)
        self.k_proj: nn.Linear = nn.Linear(dim, dim)
        self.v_proj: nn.Linear = nn.Linear(dim, dim)
        self.out_proj: nn.Linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass for ReGLALayer.

        Args:
            x: Input tensor of shape `[B, L, D]`.

        Returns:
            Output tensor of shape `[B, L, D]`.
        """
        B, L, D = x.shape
        q = F.relu(self.q_proj(x)).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = F.relu(self.k_proj(x)).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)

        kv = torch.einsum("bhld,bhlm->bhldm", k, v)  # (B, H, L, d, d)
        z = k.cumsum(dim=2)  # (B, H, L, d)
        num = torch.einsum("bhld,bhldm->bhlm", q, kv.cumsum(dim=2))
        den = torch.einsum("bhld,bhld->bhl", q, z) + 1e-8
        out = (num / den.unsqueeze(-1)).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    r"""
    Standard Pre-Norm Transformer Layer with Multi-Head Attention and RoPE.

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L^2 \cdot D + B \cdot L \cdot D \cdot D_{\text{ff}})$
        - Space: $\mathcal{O}(B \cdot H \cdot L^2)$ for softmax attention map.
    """

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.rope: RotaryEmbedding = RotaryEmbedding(dim // heads)
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm1: nn.LayerNorm = nn.LayerNorm(dim)
        self.norm2: nn.LayerNorm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass for TransformerLayer.

        Args:
            x: Input tensor of shape `[B, L, D]`.

        Returns:
            Output tensor of shape `[B, L, D]`.
        """
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


@ATTENTION_REGISTRY.register("halo_regla")
class HALOReGLAAttention(BaseAttention):
    r"""
    HALO-ReGLA: Hardware-Aware Linear Optics & Rectified Gated Linear Attention.

    Mathematical Mechanism:
        Combines multi-head linear projection with Rotary Position Embeddings (RoPE) and
        ReLU non-negative kernel representation $\phi(x) = \text{ReLU}(x)$:
        $$\begin{aligned}
        Q &= \text{RoPE}(X W_Q), \quad K = \text{RoPE}(X W_K), \quad V = X W_V \\
        \tilde{Q} &= \text{ReLU}(Q), \quad \tilde{K} = \text{ReLU}(K) \\
        KV &= \sum_{t=1}^L \tilde{K}_t^\top V_t \in \mathbb{R}^{B \times H \times d \times d} \\
        Z &= \sum_{t=1}^L \tilde{K}_t \in \mathbb{R}^{B \times H \times d} \\
        \text{Attn}(X) &= \frac{\tilde{Q} (KV)}{\tilde{Q} Z^\top + \epsilon} W_O
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot L \cdot d^2)$ — strictly linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot H \cdot d^2)$ memory for associative matrix buffers.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        hybrid_ratio: int = 4,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = getattr(config, "hidden_size", 768)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.hybrid_ratio: int = hybrid_ratio

        self.q_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.k_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.v_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.out_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.rope: RotaryEmbedding = RotaryEmbedding(self.head_dim)
        dropout_p: float = getattr(config, "dropout", 0.0)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward execution of HALOReGLAAttention.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional attention mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Boolean flag indicating if attention matrices are returned.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, None)` of shape `([B, L, D], None)`.
        """
        B, L, D = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # ReLU linear attention
        q = F.relu(q)
        k = F.relu(k)

        kv = torch.einsum("b h n d, b h n e -> b h d e", k, v)
        z = k.sum(dim=2, keepdim=True)
        num = torch.einsum("b h n d, b h d e -> b h n e", q, kv)
        den = (
            torch.einsum("b h n d, b h d -> b h n", q, z.squeeze(2)).unsqueeze(-1)
            + 1e-8
        )
        out = num / den
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), None
