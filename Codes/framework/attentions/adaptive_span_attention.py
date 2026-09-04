
from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("adaptive_span", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class AdaptiveSpanAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    r"""
    Adaptive Span Attention (Sukhbaatar et al., 2019).

    Mathematical Formulation:
        1. Query-Key Attention Scores:
           $$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}}$$
        
        2. Dynamic Per-Head Span Masking:
           Let $z_h = \sigma(\text{logit}_h) \cdot S_{\max}$.
           $$M_{ij}^{(h)} = \begin{cases} 0 & \text{if } |i - j| \le z_h \\ -10000 & \text{otherwise} \end{cases}$$
        
        3. Context:
           $$A = \operatorname{Softmax}(S + M + M_{\text{pad}})$$
           $$O = A V W_O$$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        dropout: float = 0.1,
        max_span: int = 128,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        cfg = getattr(config, "text_config", config)
        self.num_heads: int = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", 4))
        hidden_size: int = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 768))
        self.head_dim: int = getattr(cfg, "head_dim", hidden_size // max(1, self.num_heads))
        self.all_head_size: int = getattr(cfg, "q_proj_out", self.num_heads * self.head_dim)
        self.num_kv_heads: int = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_kv_heads", self.num_heads))
        self.kv_head_size: int = getattr(cfg, "kv_proj_out", self.num_kv_heads * self.head_dim)
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.max_span: int = getattr(cfg, "max_span", max_span)
        self.is_decoder: bool = getattr(cfg, "is_decoder", False)
        self.has_relative_attention_bias: bool = False
        self.relative_attention_bias: Optional[nn.Embedding] = None
        self.relative_attention_num_buckets: int = getattr(cfg, "relative_attention_num_buckets", 32)
        self.relative_attention_max_distance: int = getattr(cfg, "relative_attention_max_distance", 128)
        self.supports_rope: bool = True
        self.supports_gqa: bool = True

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.kv_head_size)
        self.v_proj = nn.Linear(hidden_size, self.kv_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)

        # Initialize to 5.0 so span starts fully open (sigmoid(5.0) ~ 0.993) preserving full context
        self.span_logits = nn.Parameter(torch.full((self.num_heads,), 5.0, dtype=torch.float32))

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
        r"""Universal adapter weight copying with T5 relative position bias preservation."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    if target.weight.shape == w.shape:
                        target.weight.data.copy_(w)
                    elif target.weight.shape == w.t().shape:
                        target.weight.data.copy_(w.t())
                    else:
                        print(f"CRITICAL WARNING: Shape mismatch copying {proj}: target {target.weight.shape} vs src {w.shape}")
                        min_r = min(target.weight.shape[0], w.shape[0])
                        min_c = min(target.weight.shape[1], w.shape[1])
                        target.weight.data[:min_r, :min_c].copy_(w[:min_r, :min_c])
                if b is not None and getattr(target, "bias", None) is not None:
                    if target.bias.shape == b.shape:
                        target.bias.data.copy_(b)
                    else:
                        min_b = min(target.bias.shape[0], b.shape[0])
                        target.bias.data[:min_b].copy_(b[:min_b])
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to copy {proj} weights: {e}")

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
        K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
        V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)

        # Apply RoPE if available in kwargs (Crucial for Qwen3.5/LLaMA/Mistral)
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            def _rotate_half(x):
                return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
            if cos.dim() == 3:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            while cos.dim() < Q.dim():
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            if cos.shape[-2] != Q.shape[-2]:
                cos, sin = cos[..., :Q.shape[-2], :], sin[..., :Q.shape[-2], :]
            rotary_dim = cos.shape[-1]
            if rotary_dim <= Q.shape[-1]:
                q_rot, q_pass = Q[..., :rotary_dim], Q[..., rotary_dim:]
                k_rot, k_pass = K[..., :rotary_dim], K[..., rotary_dim:]
                q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
                k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
                Q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
                K = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
            else:
                cos_cut, sin_cut = cos[..., :Q.shape[-1]], sin[..., :Q.shape[-1]]
                Q = (Q * cos_cut.to(Q.dtype)) + (_rotate_half(Q) * sin_cut.to(Q.dtype))
                K = (K * cos_cut.to(K.dtype)) + (_rotate_half(K) * sin_cut.to(K.dtype))

        if self.kv_repeat > 1:
            K = K.unsqueeze(2).expand(-1, -1, self.kv_repeat, -1, -1).reshape(B, H, N_k, d)
            V = V.unsqueeze(2).expand(-1, -1, self.kv_repeat, -1, -1).reshape(B, H, N_k, d)

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

        # Adaptive span mask (applied for self-attention)
        if not is_cross_attention and N_q == N_k:
            span = torch.sigmoid(self.span_logits) * self.max_span   # [H]
            pos = torch.arange(N_q, device=device, dtype=torch.float32)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()       # [N, N]
            span_bool = dist.unsqueeze(0) <= span.unsqueeze(1).unsqueeze(2)  # [H, N, N]
            # --- BRUTE-FORCE PERISCOPE LOGIC FOR CAUSAL & BIDIRECTIONAL DECODERS ---
            # If causal decoder, enforce causal mask first
            if self.is_decoder or getattr(self.config, "is_decoder", False) or getattr(self.config, "model_type", "") == "gpt2":
                causal_tril = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(~causal_tril, -10000.0)

            # Open periscope visibility for the last 15 tokens to bypass right-padding blindness
            span_bool = span_bool.clone()
            span_bool[..., -min(15, N_q):, :] = True

            scores = scores.masked_fill(~span_bool.to(scores.device).unsqueeze(0), -10000.0)
            # ------------------------------------------------------------------------

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


@ATTENTION_REGISTRY.register("adaptive_span_inject")
def inject_adaptive_span(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
    **kwargs) -> AdaptiveSpanAttention:
    rank = getattr(comp_config, "adaptive_span_rank", 32)
    dropout = getattr(comp_config, "dropout", 0.1)
    max_span = getattr(comp_config, "max_span", 128)
    new_attn = AdaptiveSpanAttention(
        config=model_config,
        rank=rank,
        dropout=dropout,
        max_span=max_span,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn