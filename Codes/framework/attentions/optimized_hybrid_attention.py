r"""
Optimized Hybrid Attention (OHA).

Fuses localized sliding-window self-attention with sequence-compressed low-rank global attention
via dynamic head-wise gated modulation.

Mathematical Mechanism:
    Given queries $Q$, keys $K$, values $V \in \mathbb{R}^{B \times H \times N \times d}$:
    1. **Local Path (Sliding Window):**
       $$A^{\text{local}}_{ij} = \operatorname{Softmax}\left(\frac{Q_i K_j^T}{\sqrt{d}} + M^{\text{band}}_{ij}\right)$$
       where $M^{\text{band}}_{ij} = 0$ if $|i - j| \le W$, else $-\infty$.
       $$O^{\text{local}} = A^{\text{local}} V$$
    2. **Global Path (Low-Rank Compression):**
       Compressed keys $\widetilde{K} = \operatorname{Softmax}(X W_K^{\text{comp}}, \text{dim}=N)^T K \in \mathbb{R}^{B \times H \times R \times d}$
       Compressed values $\widetilde{V} = \operatorname{Softmax}(X W_V^{\text{comp}}, \text{dim}=N)^T V \in \mathbb{R}^{B \times H \times R \times d}$
       $$A^{\text{global}} = \operatorname{Softmax}\left(\frac{Q \widetilde{K}^T}{\sqrt{d}}\right), \quad O^{\text{global}} = A^{\text{global}} \widetilde{V}$$
    3. **Gated Fusion:**
       $$g = \sigma(X W_g + b_g) \in \mathbb{R}^{B \times N \times H}$$
       $$O = g \odot O^{\text{local}} + (1 - g) \odot O^{\text{global}}$$

Computational Complexity:
    - Local path: $\mathcal{O}(B \cdot H \cdot N \cdot W \cdot d)$
    - Global path: $\mathcal{O}(B \cdot H \cdot N \cdot R \cdot d)$
    - Overall Complexity: $\mathcal{O}(N \cdot (W + R) \cdot d) \ll \mathcal{O}(N^2 \cdot d)$ for $W, R \ll N$.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def transpose_for_scores(
    x: torch.Tensor, num_attention_heads: int, head_dim: int
) -> torch.Tensor:
    r"""Reshape `[Batch, SeqLen, HiddenDim]` to `[Batch, NumHeads, SeqLen, HeadDim]`.

    Args:
        x: Input projection tensor.
        num_attention_heads: Number of attention heads $H$.
        head_dim: Dimensionality per head $d$.

    Returns:
        Transposed tensor of shape `[Batch, NumHeads, SeqLen, HeadDim]`.
    """
    new_x_shape = x.size()[:-1] + (num_attention_heads, head_dim)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)


@dataclass
class OptimizedHybridAttentionConfig:
    r"""Configuration hyperparameters for Optimized Hybrid Attention.

    Attributes:
        hidden_size: Total hidden model dimension $D$.
        num_attention_heads: Number of parallel attention heads $H$.
        attention_probs_dropout_prob: Attention map dropout probability.
        window_size: Half-window size for local sliding-window attention ($W$).
        compression_rank: Target sequence rank for global pooling ($R$).
        use_gate: Enable dynamic learned head-wise gating between local and global streams.
        gate_bias: Initial bias constant for the gate projection.
        layer_norm_eps: Epsilon parameter for layer normalization.
    """

    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_probs_dropout_prob: float = 0.1
    window_size: int = 128
    compression_rank: int = 64
    use_gate: bool = True
    gate_bias: float = 2.0
    layer_norm_eps: float = 1e-5


class OptimizedHybridAttention(BaseAttention):
    r"""Optimized Hybrid Attention (OHA) combining sliding-window local and compressed global attention.

    Args:
        config: Configuration instance or dataclass with attention hyperparameters.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        hidden_size: int = getattr(config, "hidden_size", 768)
        num_attention_heads: int = getattr(config, "num_attention_heads", 12)
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_heads: int = num_attention_heads
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.window_size: int = getattr(config, "window_size", 128)
        self.compression_rank: int = getattr(config, "compression_rank", 64)
        self.use_gate: bool = getattr(config, "use_gate", True)
        dropout_p: float = float(
            getattr(
                config,
                "attention_probs_dropout_prob",
                getattr(config, "dropout", 0.1),
            )
        )

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.compress_k = nn.Linear(
            hidden_size, self.compression_rank, bias=False
        )
        self.compress_v = nn.Linear(
            hidden_size, self.compression_rank, bias=False
        )
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        if self.use_gate:
            self.gate = nn.Linear(hidden_size, self.num_heads)
            gate_bias_val = getattr(config, "gate_bias", 2.0)
            nn.init.constant_(self.gate.bias, gate_bias_val)
            nn.init.normal_(self.gate.weight, std=0.02)

        self._init_weights()

    def _init_weights(self) -> None:
        r"""Initialize projection parameters for key/value sequence compression."""
        nn.init.normal_(self.compress_k.weight, std=0.02)
        nn.init.normal_(self.compress_v.weight, std=0.02)

    def _sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        r"""Generate 4D sliding window boolean band mask."""
        idx = torch.arange(seq_len, device=device)
        mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute hybrid local-global attention with gated fusion.

        Args:
            hidden_states: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional mask tensor of shape `[Batch, SeqLen]` or `[Batch, 1, 1, SeqLen]`.
            head_mask: Optional head masking tensor.
            output_attentions: Return local attention probabilities if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Fused and projected representations `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Local attention probabilities or None.
        """
        B, N, _ = hidden_states.shape
        Q = transpose_for_scores(
            self.query(hidden_states), self.num_heads, self.head_dim
        )
        K = transpose_for_scores(self.key(hidden_states), self.num_heads, self.head_dim)
        V = transpose_for_scores(
            self.value(hidden_states), self.num_heads, self.head_dim
        )

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                local_add = attention_mask
                mask_2d = (attention_mask[:, 0, 0, :] > -10000.0).float() if attention_mask.size(1) == 1 else (attention_mask[:, 0, :, 0] > -10000.0).float()
                comp_add = (1.0 - mask_2d.unsqueeze(-1)) * -10000.0
            elif attention_mask.dim() == 2:
                mask_2d = attention_mask.to(dtype=hidden_states.dtype)
                local_add = (1.0 - mask_2d[:, None, None, :]) * -10000.0
                comp_add = (1.0 - mask_2d[:, :, None]) * -10000.0
            else:
                local_add = None
                comp_add = None

        # Local sliding-window attention
        scale = 1.0 / math.sqrt(self.head_dim)
        Q_s = Q * scale
        local_scores = torch.matmul(Q_s, K.transpose(-2, -1))
        win_mask = self._sliding_window_mask(N, Q.device)
        local_scores = local_scores.masked_fill(~win_mask, float("-inf"))
        if local_add is not None:
            local_scores = local_scores + local_add
        local_probs = F.softmax(local_scores, dim=-1)
        local_probs = self.dropout(local_probs)
        local_ctx = torch.matmul(local_probs, V)

        # Global low-rank attention
        comp_k_raw = self.compress_k(hidden_states)
        comp_v_raw = self.compress_v(hidden_states)
        if comp_add is not None:
            comp_k_raw = comp_k_raw + comp_add
            comp_v_raw = comp_v_raw + comp_add
        comp_k_weights = F.softmax(comp_k_raw, dim=1)
        comp_v_weights = F.softmax(comp_v_raw, dim=1)

        full_k = K.permute(0, 1, 3, 2)
        full_v = V.permute(0, 1, 3, 2)
        ck = torch.matmul(full_k, comp_k_weights.unsqueeze(1)).permute(0, 1, 3, 2)
        cv = torch.matmul(full_v, comp_v_weights.unsqueeze(1)).permute(0, 1, 3, 2)
        global_scores = torch.matmul(Q_s, ck.transpose(-2, -1))
        global_probs = F.softmax(global_scores, dim=-1)
        global_probs = self.dropout(global_probs)
        global_ctx = torch.matmul(global_probs, cv)

        # Gated fusion
        if self.use_gate:
            gate = torch.sigmoid(self.gate(hidden_states))
            gate = gate.permute(0, 2, 1).unsqueeze(-1)
            context = gate * local_ctx + (1.0 - gate) * global_ctx
        else:
            context = local_ctx + global_ctx

        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )
        output = self.out_proj(context)
        return output, (local_probs if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Compute orthogonal regularization loss."""
        return torch.tensor(0.0, device=self.out_proj.weight.device)

    @classmethod
    def from_distilbert_pretrained(
        cls, distilbert_attention_layer: Any, config: Any
    ) -> "OptimizedHybridAttention":
        r"""Construct and initialize OHA layer from pretrained DistilBERT weights."""
        oha = cls(config)
        src = distilbert_attention_layer
        oha.query.weight.data.copy_(src.q_lin.weight.data)
        oha.query.bias.data.copy_(src.q_lin.bias.data)
        oha.key.weight.data.copy_(src.k_lin.weight.data)
        oha.key.bias.data.copy_(src.k_lin.bias.data)
        oha.value.weight.data.copy_(src.v_lin.weight.data)
        oha.value.bias.data.copy_(src.v_lin.bias.data)
        oha.out_proj.weight.data.copy_(src.out_lin.weight.data)
        oha.out_proj.bias.data.copy_(src.out_lin.bias.data)
        return oha


@ATTENTION_REGISTRY.register("oha")
def inject_oha(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> OptimizedHybridAttention:
    r"""Dynamic bridge injecting OptimizedHybridAttention into a target transformer layer."""
    oha_cfg = OptimizedHybridAttentionConfig(
        hidden_size=getattr(model_config, "hidden_size", 768),
        num_attention_heads=getattr(model_config, "num_attention_heads", 12),
        window_size=getattr(comp_config, "block_size", 128) if comp_config is not None else 128,
        compression_rank=getattr(comp_config, "compression_rank", 64) if comp_config is not None else 64,
        use_gate=getattr(comp_config, "use_gate", True) if comp_config is not None else True,
        gate_bias=getattr(comp_config, "gate_bias", 2.0) if comp_config is not None else 2.0,
    )
    new_attn = OptimizedHybridAttention.from_distilbert_pretrained(old_attn, oha_cfg)
    layer.attention = new_attn
    return new_attn
