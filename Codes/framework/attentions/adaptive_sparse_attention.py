"""
Adaptive Sparse Attention Module.
Implements dynamic top-k and threshold-based adaptive sparse attention.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union
import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter
from framework.utils.compat import repeat_kv, maybe_apply_rotary, causal_mask_from_attention_mask


@ATTENTION_REGISTRY.register("adaptive_sparse_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("adaptive_sparse", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class AdaptiveSparseAttention(BaseAttention):
    supports_causal = True
    supports_bidirectional = True
    supports_gqa = True
    supports_rope = True
    r"""
    Adaptive Sparse Multi-Head Attention.
    
    Mathematical Formulation:
        1. Query-Key Attention Scores:
           $$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}}$$
        
        2. Adaptive Dynamic Sparsification:
           Per-head learnable sparsity ratio $\gamma_h = \sigma(\text{logit}_h)$.
           Top-$k$ active key positions are retained per query:
           $$k_h = \max\left(1, \lfloor N \cdot (1 - \gamma_h) \rfloor\right)$$
           $$\mathcal{M}_{ij}^{(h)} = \begin{cases} 0 & \text{if } S_{ij}^{(h)} \ge \text{TopK-Threshold}_h \\ -\infty & \text{otherwise} \end{cases}$$
        
        3. Context Aggregation:
           $$A = \operatorname{Softmax}(S + \mathcal{M} + M_{\text{pad}})$$
           $$O = (A V) W_O$$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        sparsity_ratio: float = 0.5,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.adapter = ModelAdapter(config)
        self.num_heads = self.adapter.num_heads
        self.num_key_value_heads = self.adapter.num_key_value_heads
        hidden_size = self.adapter.hidden_size
        self.head_dim = self.adapter.head_dim
        self.q_out, self.kv_out, self.head_dim, self.kv_repeat = get_qkv_projection_dims(config)
        self.all_head_size = self.q_out
        self.all_kv_head_size = getattr(self, 'kv_out', getattr(self, 'all_head_size', 768))
        self.num_kv_heads = getattr(config, 'num_key_value_heads', getattr(config, 'num_kv_heads', getattr(self, 'num_heads', 1)))
        self.supports_gqa = True
        self.kv_head_size = self.adapter.kv_proj_out
        self.default_sparsity: float = float(sparsity_ratio)

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.kv_head_size)
        self.v_proj = nn.Linear(hidden_size, self.kv_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)

        # Learnable per-head sparsity logit initialized to target sparsity
        init_logit = math.log(max(1e-4, min(0.9999, sparsity_ratio)) / (1.0 - max(1e-4, min(0.9999, sparsity_ratio))))
        self.sparsity_logits = nn.Parameter(torch.full((self.num_heads,), fill_value=init_logit))

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", dropout),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        self.is_decoder: bool = getattr(config, "is_decoder", False)
        self.has_relative_attention_bias: bool = False
        self.relative_attention_bias: Optional[nn.Embedding] = None
        self.relative_attention_num_buckets: int = getattr(config, "relative_attention_num_buckets", 32)
        self.relative_attention_max_distance: int = getattr(config, "relative_attention_max_distance", 128)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _relative_position_bucket(
        self, relative_position: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.relative_attention_bias is not None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            relative_position = memory_position - context_position
            relative_position_bucket = self._relative_position_bucket(
                relative_position,
                bidirectional=(not self.is_decoder),
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            values = self.relative_attention_bias(relative_position_bucket)
            values = values.permute([2, 0, 1]).unsqueeze(0)
            return values
        return torch.zeros((1, self.num_heads, query_length, key_length), device=device)

    def copy_weights(self, src: Any) -> None:
        """Universal adapter weight copying with T5 relative position bias preservation."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if hasattr(src, "relative_attention_bias") and src.relative_attention_bias is not None:
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

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
        # Check for cross-attention
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2)
        K = self.k_proj(kv_input).view(B, N_k, self.num_key_value_heads, d).transpose(1, 2)
        V = self.v_proj(kv_input).view(B, N_k, self.num_key_value_heads, d).transpose(1, 2)
        Q, K = maybe_apply_rotary(Q, K, self.adapter, self.rotary_emb, kwargs.get("position_ids"))
        K, V = repeat_kv(K, self.adapter.kv_repeat), repeat_kv(V, self.adapter.kv_repeat)

        scale = d ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N_q, N_k]

        # Handle T5 position bias
        position_bias = kwargs.get("position_bias", None)
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(N_q, N_k, device=device)

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if position_bias is not None:
            if mask is not None and mask.dim() == 4:
                position_bias = position_bias + mask.to(scores.dtype)
            scores = scores + position_bias.to(scores.dtype)
        elif mask is not None:
            if mask.dim() == 4:
                scores = scores + mask.to(scores.dtype)
            else:
                valid_m = get_token_valid_mask(mask, B, N_k, device)
                if valid_m is not None:
                    scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)

        # 1. Explicit Causal Masking (CRITICAL for GPT-2 / Qwen before Top-K)
        if self.is_decoder or getattr(self.config, "is_decoder", False) or getattr(self.config, "model_type", "") == "gpt2":
            causal_tril = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_tril, -10000.0)

        # 2. Vectorized Top-K Thresholding (NO loops, NO .item() calls)
        # We detach the logits for the index calculation since dynamic 'k' indexing is non-differentiable
        sparsity_per_head = torch.sigmoid(self.sparsity_logits.detach()).clamp(0.0, 0.95)  # [H]
        k_per_head = torch.clamp(((1.0 - sparsity_per_head) * N_k).long(), min=1, max=N_k)  # [H]

        # Sort scores along the key dimension
        sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)  # [B, H, N_q, N_k]
        
        # Broadcast k_per_head to gather the threshold value
        k_indices = (k_per_head - 1).view(1, H, 1, 1).expand(B, H, N_q, 1)
        
        # Gather the exact threshold score for each head efficiently
        topk_thresh = torch.gather(sorted_scores, dim=-1, index=k_indices)  # [B, H, N_q, 1]

        # 3. Apply the Sparse Mask
        scores = scores.masked_fill(scores < topk_thresh, -10000.0)

        attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        output = self.out_proj(context)

        if position_bias is not None:
            return output, position_bias
        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("adaptive_sparse_attention_inject")
@ATTENTION_REGISTRY.register("adaptive_sparse_inject")
def inject_adaptive_sparse(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
    **kwargs) -> AdaptiveSparseAttention:
    sparsity_ratio = getattr(comp_config, "adaptive_sparsity_ratio", 0.5)
    dropout = getattr(comp_config, "dropout", 0.1)
    new_attn = AdaptiveSparseAttention(
        config=model_config,
        sparsity_ratio=sparsity_ratio,
        dropout=dropout,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
