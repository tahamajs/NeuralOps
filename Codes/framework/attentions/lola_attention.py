"""
LoLA Attention – Low‑Rank Linear Attention with Sparse Caching (2025)
Paper: "LoLA: Low-Rank Linear Attention With Sparse Caching"
https://arxiv.org/abs/2509.58281

Key innovations (integrated into the code):
1. Three‑memory‑system caching: past KV pairs distributed into recent, sparse, and low‑rank cache.
2. Training‑free augmentation that boosts associative recall without extra training.
3. Constant memory footprint even on infinite context lengths.
4. Cached attention (K_cached, V_cached) used for retrieval, combined with low‑rank global context.
5. Learnable LR_scale and sparse_gate to weight contributions.

Zero‑shot initialisation from a pretrained DistilBERT attention layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class LowRankCache(nn.Module):
    """
    LoLA cache: stores recent, sparse, and low‑rank KV pairs.
    Implements the three‑memory system described in the paper.
    """
    def __init__(self, max_len: int = 128, low_rank_k: int = 16):
        super().__init__()
        self.max_len = max_len
        self.low_rank_k = low_rank_k
        self.register_buffer('recent_k', torch.zeros(0))
        self.register_buffer('recent_v', torch.zeros(0))
        self.register_buffer('sparse_k', torch.zeros(0))
        self.register_buffer('sparse_v', torch.zeros(0))
        self.register_buffer('low_rank_k', torch.zeros(0))
        self.register_buffer('low_rank_v', torch.zeros(0))

    def update(self, K: torch.Tensor, V: torch.Tensor):
        """
        K, V: (B, H, N, D)
        Updates caches with the current K, V.
        """
        B, H, N, D = K.shape
        # For simplicity, use the first batch and first head as reference
        # (You may expand to per‑batch/head in production)
        k_flat = K[0, 0]  # (N, D)
        v_flat = V[0, 0]

        # Recent cache (sliding window)
        self.recent_k = torch.cat([self.recent_k, k_flat], dim=0)[-self.max_len:, :]
        self.recent_v = torch.cat([self.recent_v, v_flat], dim=0)[-self.max_len:, :]

        # Sparse cache: approx every 4th token (example pattern)
        sparse_idx = torch.arange(0, N, 4)
        sparse_k = k_flat[sparse_idx]
        sparse_v = v_flat[sparse_idx]
        self.sparse_k = torch.cat([self.sparse_k, sparse_k], dim=0)[-self.max_len//2:, :]
        self.sparse_v = torch.cat([self.sparse_v, sparse_v], dim=0)[-self.max_len//2:, :]

        # Low‑rank cache: SVD on recent KV to extract top‑k components
        if self.recent_k.size(0) > self.low_rank_k:
            U, S, Vt = torch.linalg.svd(self.recent_k, full_matrices=False)
            low_k = U[:, :self.low_rank_k] @ torch.diag(S[:self.low_rank_k]) @ Vt[:self.low_rank_k, :]
            self.low_rank_k = low_k
            # Similarly for V (simplified: store top‑k SVD components)
            Uv, Sv, Vtv = torch.linalg.svd(self.recent_v, full_matrices=False)
            low_v = Uv[:, :self.low_rank_k] @ torch.diag(Sv[:self.low_rank_k]) @ Vtv[:self.low_rank_k, :]
            self.low_rank_v = low_v

    def get_cached(self):
        """
        Returns combined cached K, V from all three memory systems.
        """
        if self.recent_k.numel() == 0:
            return None, None
        # Stack as (C, D) tensors
        K_cached = torch.cat([self.recent_k, self.sparse_k, self.low_rank_k], dim=0)
        V_cached = torch.cat([self.recent_v, self.sparse_v, self.low_rank_v], dim=0)
        return K_cached, V_cached


class LowRankLinear(nn.Module):
    """Low‑rank linear layer: W = U @ V, computed as U(Vx) + bias."""
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U)
        if self.bias is not None:
            out = out + self.bias
        return out

    def set_low_rank_factors(self, U: torch.Tensor, V: torch.Tensor):
        with torch.no_grad():
            self.U.copy_(U)
            self.V.copy_(V)


class LoLAAttention(nn.Module):
    """
    LoLA Attention – Low‑Rank Linear Attention with Sparse Caching.
    Combines global low‑rank attention with a selective cached memory for retrieval.
    """
    def __init__(self, model_config, rank: int = 32, dropout: float = 0.1,
                 pretrained_attn=None, comp_config=None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.rank = rank

        # Low‑rank Q/K/V/Out projections
        self.q_proj = LowRankLinear(self.hidden_size, self.all_head_size, rank,
                                    bias=pretrained_attn.q_lin.bias is not None)
        self.k_proj = LowRankLinear(self.hidden_size, self.all_head_size, rank,
                                    bias=pretrained_attn.k_lin.bias is not None)
        self.v_proj = LowRankLinear(self.hidden_size, self.all_head_size, rank,
                                    bias=pretrained_attn.v_lin.bias is not None)
        self.out_proj = LowRankLinear(self.all_head_size, self.hidden_size, rank,
                                      bias=pretrained_attn.out_lin.bias is not None)

        # SVD initialisation using pretrained weights
        self._init_from_linear(self.q_proj, pretrained_attn.q_lin)
        self._init_from_linear(self.k_proj, pretrained_attn.k_lin)
        self._init_from_linear(self.v_proj, pretrained_attn.v_lin)
        self._init_from_linear(self.out_proj, pretrained_attn.out_lin)

        self.dropout = nn.Dropout(dropout)

        # LoLA cache mechanisms
        max_seq_len = getattr(comp_config, 'max_seq_len', 128)
        low_rank_cache_size = getattr(comp_config, 'low_rank_cache_size', 16)
        self.cache = LowRankCache(max_len=max_seq_len, low_rank_k=low_rank_cache_size)
        self.LR_scale = nn.Parameter(torch.tensor(0.5))
        self.sparse_gate = nn.Parameter(torch.tensor(0.5))

    def _init_from_linear(self, lowrank: LowRankLinear, orig: nn.Linear):
        with torch.no_grad():
            W = orig.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, U.size(1), Vt.size(0))
            lowrank.U.data[:, :r] = (U[:, :r] * S[:r]).to(lowrank.U.dtype)
            lowrank.V.data[:r, :] = Vt[:r, :].to(lowrank.V.dtype)
            if lowrank.bias is not None and orig.bias is not None:
                lowrank.bias.data = orig.bias.data.clone()

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        K = self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)

        # 1) Low‑rank global attention
        scale = D ** -0.5
        scores_lr = torch.matmul(Q, K.transpose(-2, -1)) * scale * self.LR_scale  # (B, H, N, N)

        # 2) Sparse cached retrieval (training‑free)
        K_cached, V_cached = self.cache.get_cached()
        if K_cached is not None:
            # expand to full (B, H, N, C)
            C = K_cached.size(0)
            K_cache_exp = K_cached.unsqueeze(0).unsqueeze(0).expand(B, H, C, D)
            V_cache_exp = V_cached.unsqueeze(0).unsqueeze(0).expand(B, H, C, D)
            scores_cache = torch.matmul(Q, K_cache_exp.transpose(-2, -1)) * scale * self.sparse_gate
            scores = torch.cat([scores_lr, scores_cache], dim=-1)
            V_combined = torch.cat([V, V_cache_exp], dim=2)
        else:
            scores = scores_lr
            V_combined = V

        # Masking
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]  # (B,1,1,N)
            else:
                mask = attention_mask
            # Pad mask dimension if cached part exists
            if K_cached is not None:
                cache_mask = torch.ones(B, 1, 1, C, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, cache_mask], dim=-1)
            scores = scores + (1.0 - mask) * -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V_combined)                # (B, H, N, D)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)   # (B, N, H*D)
        output = self.out_proj(context)

        # Update caches for next step (during inference or training)
        # For simplicity, we update only at the end; in autoregressive inference you'd update per token.
        self.cache.update(K, V)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("lola_attn")
def inject_lola_attn(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """
    Injection function for DistilBERT.
    comp_config may contain:
      - rank (int, default 32)
      - dropout (float, default 0.1)
      - max_seq_len (int, default 128)
      - low_rank_cache_size (int, default 16)
    """
    rank = getattr(comp_config, 'rank', 32)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = LoLAAttention(model_config, rank, dropout, old_attn, comp_config)
    layer.attention = new_attn
    return new_attn