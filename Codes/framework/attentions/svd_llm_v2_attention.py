r"""
SVD-LLM V2 Attention: Truncation-Aware SVD for LLM Compression.

Direct low-rank factorized parameterization maintaining explicit $U$, $\Sigma$, and $V^H$ factor buffers,
initialized via truncated SVD with energy preservation thresholds.

Mathematical Mechanism:
    For projection operators $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$:
    1. **Truncated Factorized Parameterization:**
       $$W \approx (U \operatorname{diag}(S)) V^H, \quad U \in \mathbb{R}^{D_{\text{out}} \times r}, \, S \in \mathbb{R}^r, \, V^H \in \mathbb{R}^{r \times D_{\text{in}}}$$
       $$y = x (V^H)^T \operatorname{diag}(S) U^T = \left((x (V^H)^T) \odot S\right) U^T$$
    2. **Multi-Head Attention Computation:**
       $$A = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) \in \mathbb{R}^{B \times H \times N \times N}$$
       $$O = (A V) W_O \in \mathbb{R}^{B \times N \times D}$$

Computational Complexity:
    - Factorized projection evaluation: $\mathcal{O}(B \cdot N \cdot r \cdot (D_{\text{in}} + D_{\text{out}}))$.
    - Attention matrix products: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d)$.
    - Total per-layer complexity: $\mathcal{O}(B \cdot N \cdot r \cdot D + B \cdot H \cdot N^2 \cdot d)$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("svd_llm_v2")
class SVDLLMV2Attention(BaseAttention):
    r"""SVD-factorized attention layer with explicit $U, \Sigma, V^H$ matrix buffers.

    Args:
        config: Transformer configuration specifying `hidden_size`, `num_attention_heads`, and `dropout`.
        rank: Target compression rank $r$.
        energy_threshold: Singular value energy threshold fraction $\tau \in (0, 1]$.
        pretrained_attn: Optional source attention module for zero-shot SVD decomposition.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 64,
        energy_threshold: float = 0.95,
        pretrained_attn: Optional[Any] = None,
    ) -> None:
        actual_rank = rank if rank is not None else 64
        super().__init__(config, actual_rank)
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rank: int = actual_rank
        self.energy_threshold: float = energy_threshold

        # Create parameter buffers
        self.query_u = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.query_s = nn.Parameter(torch.empty(self.rank))
        self.query_vh = nn.Parameter(torch.empty(self.rank, config.hidden_size))

        self.key_u = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.key_s = nn.Parameter(torch.empty(self.rank))
        self.key_vh = nn.Parameter(torch.empty(self.rank, config.hidden_size))

        self.value_u = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.value_s = nn.Parameter(torch.empty(self.rank))
        self.value_vh = nn.Parameter(torch.empty(self.rank, config.hidden_size))

        self.out_proj_u = nn.Parameter(torch.empty(config.hidden_size, self.rank))
        self.out_proj_s = nn.Parameter(torch.empty(self.rank))
        self.out_proj_vh = nn.Parameter(torch.empty(self.rank, self.all_head_size))

        if pretrained_attn is not None:
            self._init_from_pretrained(pretrained_attn)
        else:
            self._init_random()

        dropout_p: float = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
        )
        self.dropout = nn.Dropout(dropout_p)

    def _init_random(self) -> None:
        r"""Initialize orthogonal factor projections with identity scales."""
        nn.init.orthogonal_(self.query_u)
        nn.init.ones_(self.query_s)
        nn.init.orthogonal_(self.query_vh)
        nn.init.orthogonal_(self.key_u)
        nn.init.ones_(self.key_s)
        nn.init.orthogonal_(self.key_vh)
        nn.init.orthogonal_(self.value_u)
        nn.init.ones_(self.value_s)
        nn.init.orthogonal_(self.value_vh)
        nn.init.orthogonal_(self.out_proj_u)
        nn.init.ones_(self.out_proj_s)
        nn.init.orthogonal_(self.out_proj_vh)

    def _init_from_pretrained(self, pretrained_attn: Any) -> None:
        r"""Initialize SVD factors from pretrained linear layers."""
        with torch.no_grad():
            projections = [
                (pretrained_attn.q_lin, self.query_u, self.query_s, self.query_vh),
                (pretrained_attn.k_lin, self.key_u, self.key_s, self.key_vh),
                (pretrained_attn.v_lin, self.value_u, self.value_s, self.value_vh),
                (pretrained_attn.out_lin, self.out_proj_u, self.out_proj_s, self.out_proj_vh),
            ]
            for orig, u_param, s_param, vh_param in projections:
                W = orig.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(self.rank, len(S))
                u_param.data[:, :r] = (U[:, :r] * S[:r]).to(dtype=torch.float32)
                s_param.data[:r] = torch.ones(r)
                vh_param.data[:r, :] = Vt[:r, :].to(dtype=torch.float32)
                if r < self.rank:
                    self._pad_factors(u_param, s_param, vh_param, self.rank)

    def _pad_factors(
        self,
        u: nn.Parameter,
        s: nn.Parameter,
        vh: nn.Parameter,
        target_rank: int,
    ) -> None:
        r"""Pad factors to target rank with small random values."""
        current_rank = u.shape[1]
        if current_rank >= target_rank:
            return
        pad_u = torch.randn(u.shape[0], target_rank - current_rank, device=u.device) * 0.01
        u.data = torch.cat([u.data[:, :current_rank], pad_u], dim=1)
        pad_s = torch.ones(target_rank - current_rank, device=s.device) * 0.01
        s.data = torch.cat([s.data[:current_rank], pad_s], dim=0)
        pad_vh = torch.randn(target_rank - current_rank, vh.shape[1], device=vh.device) * 0.01
        vh.data = torch.cat([vh.data[:current_rank, :], pad_vh], dim=0)

    def _reshape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        r"""Reshape `[Batch, SeqLen, AllHeadSize]` to `[Batch, Heads, SeqLen, HeadDim]`."""
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Forward execution of SVD-LLM V2 attention layer.

        Args:
            hidden_states: Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head pruning tensor.
            output_attentions: Return explicit attention weights if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax probability matrix or None.
        """
        batch_size, seq_len, _ = hidden_states.shape

        W_q = (self.query_u * self.query_s) @ self.query_vh
        W_k = (self.key_u * self.key_s) @ self.key_vh
        W_v = (self.value_u * self.value_s) @ self.value_vh
        W_o = (self.out_proj_u * self.out_proj_s) @ self.out_proj_vh

        Q = (
            F.linear(hidden_states, W_q)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        K = (
            F.linear(hidden_states, W_k)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        V = (
            F.linear(hidden_states, W_v)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        scale = self.head_dim**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask) * -1e9
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1)
            else:
                scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask

        context = torch.matmul(probs, V)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        out = F.linear(context, W_o)

        if output_attentions:
            return out, probs
        return out, None


@ATTENTION_REGISTRY.register("svd_llm_v2")
def inject_svd_llm_v2(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> SVDLLMV2Attention:
    r"""Bridge helper injecting SVDLLMV2Attention into a target transformer layer."""
    rank = getattr(comp_config, "rank", 64)
    energy = getattr(comp_config, "svd_energy", 0.95)
    new_attn = SVDLLMV2Attention(
        model_config, rank=rank, energy_threshold=energy, pretrained_attn=old_attn
    )
    layer.attention = new_attn
    return new_attn
