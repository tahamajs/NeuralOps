import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.simple_ssm import SimpleSSM


class UnifiedRoPE(nn.Module):
    r"""
    Unified Rotary Position Embedding (RoPE).

    Mathematical Mechanism:
        Applies 2D rotation matrices to adjacent feature pairs based on position $m$:
        $$\begin{aligned}
        \Theta &= \{\theta_i = b^{-2(i-1)/D}, i \in [1, D/2]\} \\
        R_{\Theta, m}^D x_m &= [x_{m, 2k} \cos(m\theta_k) - x_{m, 2k+1} \sin(m\theta_k), \; x_{m, 2k} \sin(m\theta_k) + x_{m, 2k+1} \cos(m\theta_k)]
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot D)$
        - Space: $\mathcal{O}(L_{\max} \cdot D)$ buffer.

    Tensor Dimensions:
        - Input: `[B, L, D]`
        - Output: `[B, L, D]`
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        self.base: float = base

        # Precompute frequency bands: shape (dim//2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer(
            "cos_cached", emb.cos().unsqueeze(0).unsqueeze(2), persistent=False
        )  # (1, seq_len, 1, dim)
        self.register_buffer(
            "sin_cached", emb.sin().unsqueeze(0).unsqueeze(2), persistent=False
        )

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        r"""
        Apply RoPE rotation to input tensor.

        Args:
            x: Input tensor `[B, L, D]`.
            offset: Starting offset for streaming autoregressive decoding.

        Returns:
            Positionally rotated tensor `[B, L, D]`.
        """
        batch, seq_len, dim = x.shape
        assert dim == self.dim, f"Expected input dim {self.dim}, got {dim}"

        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:, offset : offset + seq_len, :, :]  # (1, seq_len, 1, dim)
        sin = self.sin_cached[:, offset : offset + seq_len, :, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = (
            x_even * cos.squeeze(2)[..., 0::2] - x_odd * sin.squeeze(2)[..., 0::2]
        )
        rotated_odd = (
            x_odd * cos.squeeze(2)[..., 0::2] + x_even * sin.squeeze(2)[..., 0::2]
        )
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
        return rotated


@ATTENTION_REGISTRY.register("hymba_transx")
class HymbaTransXBlock(BaseAttention):
    r"""
    HymbaTransX: Hybrid Differential Transformer & State Space Model Block.

    Mathematical Mechanism:
        Combines Differential Attention with a parallel Mamba SSM stream modulated by learnable gating:
        $$\begin{aligned}
        X_{\text{rope}} &= \text{RoPE}(X) \\
        A_1 &= \text{softmax}\left(\frac{Q_1 K_1^\top}{\sqrt{d_k}}\right), \quad A_2 = \text{softmax}\left(\frac{Q_2 K_2^\top}{\sqrt{d_k}}\right) \\
        A_{\text{diff}} &= \text{ReLU}(A_1 - 0.5 A_2) \\
        Y_{\text{attn}} &= A_{\text{diff}} V \\
        Y_{\text{ssm}} &= \text{SimpleSSM}(X_{\text{rope}}) \\
        \alpha &= \sigma(\theta_\alpha) \\
        Y &= W_{\text{out}}\left(X + \alpha Y_{\text{attn}} + (1 - \alpha) Y_{\text{ssm}}\right)
        \end{aligned}$$

    Computational Complexity:
        - Differential Attention: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_k)$ time, $\mathcal{O}(B \cdot H \cdot L^2)$ space.
        - Mamba SSM: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$ time, $\mathcal{O}(B \cdot D \cdot d_{\text{state}})$ space.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, diff_attn)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 64,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = getattr(config, "hidden_size", 768)
        self.num_heads: int = num_heads or getattr(config, "num_attention_heads", 12)
        self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64

        self.rope: UnifiedRoPE = UnifiedRoPE(self.dim)
        self.W_q1: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k1: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.W_q2: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k2: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.mamba: SimpleSSM = SimpleSSM(self.dim)
        self.out_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.alpha: nn.Parameter = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation of HymbaTransXBlock.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional attention mask tensor `[B, L]` or `[B, 1, L, L]`.
            head_mask: Optional head mask tensor.
            output_attentions: Boolean flag indicating if attention weights are returned.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, diff_attn)` where `output` is `[B, L, D]`.
        """
        x = hidden_states
        B, L, D = x.shape
        x_rope = self.rope(x)

        q1 = self.W_q1(x_rope).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.W_k1(x_rope).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q2 = self.W_q2(x_rope).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.W_k2(x_rope).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x_rope).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (
                    (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                )  # (B, 1, 1, L)
                scores1 = scores1.masked_fill(pad_mask, float("-1e9"))
                scores2 = scores2.masked_fill(pad_mask, float("-1e9"))
        attn1 = F.softmax(scores1, dim=-1)
        attn2 = F.softmax(scores2, dim=-1)
        diff_attn = F.relu(attn1 - 0.5 * attn2)
        A_out = torch.matmul(diff_attn, v).transpose(1, 2).contiguous().view(B, L, D)

        M_out = self.mamba(x_rope, attention_mask=attention_mask)
        alpha = torch.sigmoid(self.alpha)
        out = self.out_proj(x + alpha * A_out + (1.0 - alpha) * M_out)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                out = out * attention_mask.unsqueeze(-1)
        return out, (diff_attn if output_attentions else None)


@ATTENTION_REGISTRY.register("hymba_transx")
def inject_hymba_transx(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> HymbaTransXBlock:
    r"""
    Injection hook for HymbaTransXBlock into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Existing attention module.

    Returns:
        Instantiated `HymbaTransXBlock` assigned to `layer.attention`.
    """
    new_attn = HymbaTransXBlock(
        config=model_config,
        rank=getattr(comp_config, "rank", 64),
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
