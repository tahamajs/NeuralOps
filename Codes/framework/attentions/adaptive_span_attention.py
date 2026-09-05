r"""
Adaptive Span Attention Module (Sukhbaatar et al., 2019).

Mathematical Formulation:
    1. Query-Key Attention Scores:
       $$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}}$$

    2. Dynamic Per-Head Span Masking:
       Let $z_h = \sigma(\text{logit}_h) \cdot S_{\max}$.
       $$M_{ij}^{(h)} = \begin{cases} 0 & \text{if } |i - j| \le z_h \\ -10000 & \text{otherwise} \end{cases}$$

    3. Context Aggregation:
       $$A = \operatorname{Softmax}(S + M + M_{\text{pad}})$$
       $$O = (A V) W_O$$
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register("adaptive_span", supports_causal=True, supports_bidirectional=False, supports_gqa=True)
@ATTENTION_REGISTRY.register("adaptive_span_attention", supports_causal=True, supports_bidirectional=False, supports_gqa=True)
class AdaptiveSpanAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = False
    supports_gqa: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        dropout: float = 0.1,
        max_span: int = 128,
        lora_rank: int = 16,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.adapter = ModelAdapter(config)
        self.num_heads: int = self.adapter.num_heads
        self.num_kv_heads: int = self.adapter.num_key_value_heads
        self.hidden_size: int = self.adapter.hidden_size
        self.head_dim: int = self.adapter.head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = self.adapter.kv_repeat
        self.max_span: int = getattr(config, "max_span", max_span)
        self.is_decoder: bool = getattr(config, "is_decoder", False)
        self.is_nested_encoder: bool = False

        self.has_relative_attention_bias: bool = False
        self.relative_attention_bias: Optional[nn.Embedding] = None
        self.relative_attention_num_buckets: int = getattr(config, "relative_attention_num_buckets", 32)
        self.relative_attention_max_distance: int = getattr(config, "relative_attention_max_distance", 128)

        self.lora_rank = getattr(config, "lora_rank", lora_rank)

        self.q_proj = LoRALinear(self.hidden_size, self.all_head_size, bias=True, lora_rank=self.lora_rank, freeze_base=True)
        self.k_proj = LoRALinear(self.hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, freeze_base=True)
        self.v_proj = LoRALinear(self.hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.hidden_size, bias=True, lora_rank=self.lora_rank, freeze_base=True)

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
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

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
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. HuggingFace / standard UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            weight = src.c_attn.weight.data
            bias = getattr(src.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            # Conv1D weights are stored as [in_features, out_features]
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t()
            split_size = weight.shape[0] // 3
            q_w = weight[:split_size, :]
            k_w = weight[split_size : 2 * split_size, :]
            v_w = weight[2 * split_size :, :]
            q_b = bias_data[:split_size] if bias_data is not None else None
            k_b = bias_data[split_size : 2 * split_size] if bias_data is not None else None
            v_b = bias_data[2 * split_size :] if bias_data is not None else None
            extracted["q"] = (q_w, q_b)
            extracted["k"] = (k_w, k_b)
            extracted["v"] = (v_w, v_b)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            w_o = src.c_proj.weight.data
            if w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size:
                w_o = w_o.t()
            b_o = getattr(src.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(src, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense") and extracted["o"][0] is None:
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"AdaptiveSpanAttention missing required weights for projections: {missing}")

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    raise RuntimeError(f"AdaptiveSpanAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                raise RuntimeError(f"AdaptiveSpanAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        # Copy into projection modules
        with torch.no_grad():
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w.shape
                    target_w.data[:min(out_d, src_out), :min(in_d, src_in)].copy_(w[:min(out_d, src_out), :min(in_d, src_in)])
                    if b is not None and target.bias is not None:
                        b_val = b.data if isinstance(b, nn.Parameter) else b
                        target.bias.data[:min(len(target.bias), len(b_val))].copy_(b_val[:min(len(target.bias), len(b_val))])
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

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

        # Apply RoPE
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))

        # GQA KV repetition
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

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

        # Explicit Causal Masking for autoregressive decoders
        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )
        if is_decoder:
            causal_tril = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_tril, -10000.0)

        # Adaptive span mask (applied for self-attention)
        if not is_cross_attention and N_q == N_k:
            span = torch.sigmoid(self.span_logits) * self.max_span   # [H]
            pos = torch.arange(N_q, device=device, dtype=torch.float32)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()       # [N, N]
            span_bool = dist.unsqueeze(0) <= span.unsqueeze(1).unsqueeze(2)  # [H, N, N]
            scores = scores.masked_fill(~span_bool.to(scores.device).unsqueeze(0), -10000.0)

        attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(1)
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
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
    **kwargs: Any,
) -> Optional[AdaptiveSpanAttention]:
    try:
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
    except Exception:
        return None