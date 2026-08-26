import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class LowRankCache(nn.Module):
    r"""
    Three-Memory System Cache for LoLA Attention.

    Mathematical Mechanism:
        Maintains three complementary partitions of key-value historical tokens:
        1. **Recent Cache**: $\mathcal{C}_{\text{recent}} = \{k_{t}, v_{t}\}_{t=L-W+1}^L$ (sliding window).
        2. **Sparse Cache**: $\mathcal{C}_{\text{sparse}} = \{k_{4i}, v_{4i}\}$ (subsampled historical tokens).
        3. **Low-Rank Cache**: $\mathcal{C}_{\text{lr}} = \text{SVD-TopK}(\mathcal{C}_{\text{recent}}, k_{\text{rank}})$.

    Computational Complexity:
        - Cache Update: $\mathcal{O}(L \cdot D + W \cdot D \cdot k_{\text{rank}})$
        - Space: $\mathcal{O}((W + W_{\text{sparse}} + k_{\text{rank}}) \cdot D)$ bounded constant memory.
    """

    def __init__(self, max_len: int = 128, low_rank_k: int = 16) -> None:
        super().__init__()
        self.max_len: int = max_len
        self.low_rank_k: int = low_rank_k
        self.register_buffer("recent_k", torch.zeros(0), persistent=False)
        self.recent_k: torch.Tensor
        self.register_buffer("recent_v", torch.zeros(0), persistent=False)
        self.recent_v: torch.Tensor
        self.register_buffer("sparse_k", torch.zeros(0), persistent=False)
        self.sparse_k: torch.Tensor
        self.register_buffer("sparse_v", torch.zeros(0), persistent=False)
        self.sparse_v: torch.Tensor
        self.register_buffer("low_rank_k_buf", torch.zeros(0), persistent=False)
        self.low_rank_k_buf: torch.Tensor
        self.register_buffer("low_rank_v_buf", torch.zeros(0), persistent=False)
        self.low_rank_v_buf: torch.Tensor

    def update(self, K: torch.Tensor, V: torch.Tensor) -> None:
        r"""Updates the triple memory cache with newly computed key/value activations."""
        B, H, N, D = K.shape
        k_flat = K[0, 0].detach()  # (N, D)
        v_flat = V[0, 0].detach()

        # Recent cache
        if self.recent_k.numel() == 0:
            self.recent_k = k_flat[-self.max_len :]
            self.recent_v = v_flat[-self.max_len :]
        else:
            self.recent_k = torch.cat([self.recent_k, k_flat], dim=0)[-self.max_len :, :]
            self.recent_v = torch.cat([self.recent_v, v_flat], dim=0)[-self.max_len :, :]

        # Sparse cache
        sparse_idx = torch.arange(0, N, 4, device=K.device)
        sparse_k = k_flat[sparse_idx]
        sparse_v = v_flat[sparse_idx]
        if self.sparse_k.numel() == 0:
            self.sparse_k = sparse_k[-self.max_len // 2 :]
            self.sparse_v = sparse_v[-self.max_len // 2 :]
        else:
            self.sparse_k = torch.cat([self.sparse_k, sparse_k], dim=0)[
                -self.max_len // 2 :, :
            ]
            self.sparse_v = torch.cat([self.sparse_v, sparse_v], dim=0)[
                -self.max_len // 2 :, :
            ]

        # Low-rank cache via truncated SVD
        if self.recent_k.size(0) > self.low_rank_k:
            U, S, Vt = torch.linalg.svd(self.recent_k.float(), full_matrices=False)
            low_k = (
                U[:, : self.low_rank_k]
                @ torch.diag(S[: self.low_rank_k])
                @ Vt[: self.low_rank_k, :]
            ).to(self.recent_k.dtype)
            self.low_rank_k_buf = low_k

            Uv, Sv, Vtv = torch.linalg.svd(self.recent_v.float(), full_matrices=False)
            low_v = (
                Uv[:, : self.low_rank_k]
                @ torch.diag(Sv[: self.low_rank_k])
                @ Vtv[: self.low_rank_k, :]
            ).to(self.recent_v.dtype)
            self.low_rank_v_buf = low_v

    def get_cached(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""Aggregates and returns the combined cached (K, V) keys and values."""
        if self.recent_k.numel() == 0:
            return None, None
        tensors_k = [self.recent_k]
        tensors_v = [self.recent_v]
        if self.sparse_k.numel() > 0:
            tensors_k.append(self.sparse_k)
            tensors_v.append(self.sparse_v)
        if self.low_rank_k_buf.numel() > 0:
            tensors_k.append(self.low_rank_k_buf)
            tensors_v.append(self.low_rank_v_buf)

        K_cached = torch.cat(tensors_k, dim=0)
        V_cached = torch.cat(tensors_v, dim=0)
        return K_cached, V_cached


class LowRankLinear(nn.Module):
    r"""
    Factorized Low-Rank Linear Transformation: $y = x V^\top U^\top + b$.

    Mathematical Mechanism:
        $$W \approx U V, \quad U \in \mathbb{R}^{d_{\text{out}} \times r}, \; V \in \mathbb{R}^{r \times d_{\text{in}}}$$
    """

    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        self.U: nn.Parameter = nn.Parameter(torch.empty(out_features, rank))
        self.V: nn.Parameter = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.U, self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U)
        if self.bias is not None:
            out = out + self.bias
        return out


@ATTENTION_REGISTRY.register("lola_attn")
@ATTENTION_REGISTRY.register("lola_attention")
@ATTENTION_REGISTRY.register("lola")
class LoLAAttention(BaseAttention):
    r"""
    LoLA: Low-Rank Linear Attention with Sparse Caching.

    Mathematical Mechanism:
        Combines factorized low-rank projections with selective multi-system cached retrieval:
        $$\begin{aligned}
        Q &= \text{LowRankLinear}_Q(X), \quad K = \text{LowRankLinear}_K(X), \quad V = \text{LowRankLinear}_V(X) \\
        S_{\text{lr}} &= \gamma_{\text{lr}} \cdot \frac{Q K^\top}{\sqrt{d_k}} \\
        S_{\text{cache}} &= \gamma_{\text{cache}} \cdot \frac{Q K_{\text{cache}}^\top}{\sqrt{d_k}} \\
        S &= [S_{\text{lr}} \;\|\; S_{\text{cache}}] \in \mathbb{R}^{B \times H \times N \times (N + C)} \\
        V_{\text{comb}} &= [V \;\|\; V_{\text{cache}}] \in \mathbb{R}^{B \times H \times (N + C) \times d_k} \\
        \text{Attn}(X) &= \text{softmax}(S) V_{\text{comb}} W_{\text{out}}
        \end{aligned}$$

    Computational Complexity:
        - Low-Rank Projections: $\mathcal{O}(B \cdot N \cdot D \cdot r)$
        - Attention Computation: $\mathcal{O}(B \cdot H \cdot N \cdot (N + C) \cdot d_k)$
        - Memory Footprint: $\mathcal{O}(B \cdot H \cdot N \cdot (N + C) + C_{\max} \cdot D)$ bounded cache size.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, N, D]`
        - Output: Tuple `(output, attn_weights)` where `output` is `[B, N, D]`.
    """

    def __init__(
        self,
        model_config: Any,
        rank: int = 32,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        actual_rank = getattr(
            model_config,
            "svd_rank",
            rank if rank is not None else getattr(model_config, "rank", 32),
        )
        super().__init__(model_config, actual_rank)
        self.num_heads: int = getattr(model_config, "num_attention_heads", 12)
        hidden_size: int = getattr(model_config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = hidden_size
        self.rank: int = actual_rank

        self.q_proj: LowRankLinear = LowRankLinear(self.hidden_size, self.all_head_size, self.rank, bias=True)
        self.k_proj: LowRankLinear = LowRankLinear(self.hidden_size, self.all_head_size, self.rank, bias=True)
        self.v_proj: LowRankLinear = LowRankLinear(self.hidden_size, self.all_head_size, self.rank, bias=True)
        self.out_proj: LowRankLinear = LowRankLinear(self.all_head_size, self.hidden_size, self.rank, bias=True)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

        max_seq_len: int = getattr(comp_config, "max_seq_len", 128) if comp_config is not None else 128
        low_rank_cache_size: int = getattr(comp_config, "low_rank_cache_size", 16) if comp_config is not None else 16
        self.cache: LowRankCache = LowRankCache(max_len=max_seq_len, low_rank_k=low_rank_cache_size)
        self.LR_scale: nn.Parameter = nn.Parameter(torch.tensor(0.5))
        self.sparse_gate: nn.Parameter = nn.Parameter(torch.tensor(0.5))

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _set_lowrank_factors(self, lowrank: LowRankLinear, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, U.size(1), Vt.size(0))
            sqrt_s = torch.sqrt(S[:r])
            lowrank.U.data[:, :r] = (U[:, :r] * sqrt_s).to(lowrank.U.dtype)
            lowrank.V.data[:r, :] = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(lowrank.V.dtype)
            if lowrank.bias is not None and bias is not None:
                lowrank.bias.data.copy_(bias)

    def copy_weights(self, src: Any) -> None:
        """Decompose and copy weights from any HuggingFace or PyTorch MHA module."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [
            ("q", self.q_proj),
            ("k", self.k_proj),
            ("v", self.v_proj),
            ("o", self.out_proj),
        ]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                self._set_lowrank_factors(target, w, b)
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
        r"""
        Forward pass for LoLA attention.

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
        scores_lr = torch.matmul(Q, K.transpose(-2, -1)) * scale * self.LR_scale

        K_cached, V_cached = self.cache.get_cached()
        if K_cached is not None:
            C = K_cached.size(0)
            K_cache_exp = K_cached.unsqueeze(0).unsqueeze(0).expand(B, H, C, D).to(Q.device, Q.dtype)
            V_cache_exp = V_cached.unsqueeze(0).unsqueeze(0).expand(B, H, C, D).to(V.device, V.dtype)
            scores_cache = (
                torch.matmul(Q, K_cache_exp.transpose(-2, -1))
                * scale
                * self.sparse_gate
            )
            scores = torch.cat([scores_lr, scores_cache], dim=-1)
            V_combined = torch.cat([V, V_cache_exp], dim=2)
        else:
            scores = scores_lr
            V_combined = V

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask = (attention_mask[:, 0, 0, :] > -10000.0).float()
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, 0, :]
            else:
                mask = attention_mask.float()
                
            if K_cached is not None:
                cache_mask = torch.ones(
                    B, C, device=mask.device, dtype=mask.dtype
                )
                mask = torch.cat([mask, cache_mask], dim=-1)
            
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L+C]
            scores = scores + (1.0 - mask.to(scores.dtype)) * -10000.0

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V_combined)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)

        self.cache.update(K, V)
        return output, (attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("lola_attn_inject")
def inject_lola_attn(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
    **kwargs) -> LoLAAttention:
    r"""
    Injection hook for LoLAAttention into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module.

    Returns:
        Instantiated `LoLAAttention` assigned to `layer.attention`.
    """
    rank: int = getattr(comp_config, "rank", 32)
    dropout: float = getattr(comp_config, "dropout", 0.1)
    new_attn = LoLAAttention(
        model_config=model_config,
        rank=rank,
        dropout=dropout,
        pretrained_attn=old_attn,
        comp_config=comp_config,
    )
    layer.attention = new_attn
    return new_attn
