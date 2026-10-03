"""
Zero-Mean Linear Attention (ZeroS Attention).

Removes the mean from key representations and applies normalized linear attention with ReLU-based non-negative
feature maps, with multi-architecture weight transfer, trainable LoRA residual adapters, GQA, RoPE, and causal decoding.

Hardened for production across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.
target_attention_only: Replaces self-attention modules only, preserving all FFN/MLP blocks.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions: [-x2, x1].
    Robust to odd head_dim by rotating only the even slice 2 * (d // 2).
    """
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    if rotary_dim == 0:
        return x
    x_rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings safely handling even or odd dimensions."""
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

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos_cut = cos[..., :rotary_dim].to(q.dtype)
    sin_cut = sin[..., :rotary_dim].to(q.dtype)

    q_rot_embed = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot_embed = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed

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


@ATTENTION_REGISTRY.register("zeros", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("zeros_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class ZeroSAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        lora_rank: int = 16,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if config is not None and not isinstance(config, int):
            self.adapter = ModelAdapter(config)
            self.num_heads: int = self.adapter.num_heads
            self.num_kv_heads: int = self.adapter.num_key_value_heads
            self.hidden_size: int = self.adapter.hidden_size
            self.head_dim: int = self.adapter.head_dim
            self.all_head_size: int = self.num_heads * self.head_dim
            self.kv_head_size: int = self.num_kv_heads * self.head_dim
            self.kv_repeat: int = self.adapter.kv_repeat
            self.is_decoder: bool = getattr(config, "is_decoder", False)
        else:
            self.adapter = None
            self.hidden_size = _cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768))
            self.num_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_heads", 12))
            self.num_kv_heads = _cfg_get(config, "num_key_value_heads", self.num_heads)
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
            self.is_decoder = bool(_cfg_get(config, "is_decoder", kwargs.get("is_decoder", False)))

        # Prioritize explicit kwargs
        if "hidden_size" in kwargs:
            self.hidden_size = kwargs["hidden_size"]
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
        if "num_heads" in kwargs:
            self.num_heads = kwargs["num_heads"]
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.is_nested_encoder: bool = False
        self.lora_rank = _cfg_get(config, "lora_rank", lora_rank)

        self.W_q = LoRALinear(self.hidden_size, self.all_head_size, bias=True, lora_rank=self.lora_rank, freeze_base=True)
        self.W_k = LoRALinear(self.hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, freeze_base=True)
        self.W_v = LoRALinear(self.hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, freeze_base=True)
        self.out = LoRALinear(self.all_head_size, self.hidden_size, bias=True, lora_rank=self.lora_rank, freeze_base=True)

        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            weight = src.c_attn.weight.data
            bias = getattr(src.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t().contiguous()
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
                w_o = w_o.t().contiguous()
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
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out"],
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
        has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj", "out"))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # Copy base weights into projection modules
        with torch.no_grad():
            for proj_key, target in [("q", self.W_q), ("k", self.W_k), ("v", self.W_v), ("o", self.out)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w_oriented.shape
                    target_w.data[:min(out_d, src_out), :min(in_d, src_in)].copy_(w_oriented[:min(out_d, src_out), :min(in_d, src_in)])
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

        # Preserve source LoRA adapters if present on source module
        source_submodules = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj_key, target in [("q", self.W_q), ("k", self.W_k), ("v", self.W_v), ("o", self.out)]:
            for name in source_submodules[proj_key]:
                if hasattr(src, name):
                    sub = getattr(src, name)
                    if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
                        src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
                        src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
                        if hasattr(src_lA, "data") and target.lora_A is not None:
                            if src_lA.shape[0] == target.lora_A.shape[0] and src_lA.shape[1] == target.lora_A.shape[1]:
                                target.lora_A.data.copy_(src_lA.data)
                                target.lora_A.requires_grad = True
                            else:
                                print(f"Warning: lora_A shape mismatch for {proj_key}; keeping zero-init.")
                        if hasattr(src_lB, "data") and target.lora_B is not None:
                            if src_lB.shape[0] == target.lora_B.shape[0] and src_lB.shape[1] == target.lora_B.shape[1]:
                                target.lora_B.data.copy_(src_lB.data)
                                target.lora_B.requires_grad = True
                            else:
                                print(f"Warning: lora_B shape mismatch for {proj_key}; keeping zero-init.")
                        if hasattr(sub, "lora_scaling"):
                            target.lora_scaling = sub.lora_scaling
                        break

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

        B, L, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device

        Q = self.W_q(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        K = self.W_k(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2).contiguous()
        V = self.W_v(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2).contiguous()

        # Apply RoPE: Priority 1 = position_embeddings, Priority 2 = rotary_emb fallback
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                rotary_out = self.rotary_emb(hidden_states, position_ids=kwargs.get("position_ids"))
                if isinstance(rotary_out, tuple) and len(rotary_out) >= 2:
                    cos, sin = rotary_out[0], rotary_out[1]
                    Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))
            except Exception:
                try:
                    from framework.utils.compat import maybe_apply_rotary
                    Q, K = maybe_apply_rotary(Q, K, getattr(self, "adapter", None), self.rotary_emb, kwargs.get("position_ids"))
                except Exception as e:
                    print(f'copy_weights failed: {e}')
                    logger.warning(f"copy_weights failed: {e}")

        # GQA repeat
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )

        eps = 1e-6
        attn_map = None
        if is_decoder:
            valid_m = get_token_valid_mask(attention_mask, B, N_k, device)
            if valid_m is not None:
                mask_f = valid_m.unsqueeze(1).unsqueeze(-1).to(K.dtype)
                K = K * mask_f
                V = V * mask_f
                k_cum_sum = torch.cumsum(K, dim=2)
                count = torch.cumsum(mask_f, dim=2).clamp(min=1.0)
                k_running_mean = k_cum_sum / count
            else:
                step = torch.arange(1, N_k + 1, device=device, dtype=K.dtype).view(1, 1, N_k, 1)
                k_cum_sum = torch.cumsum(K, dim=2)
                k_running_mean = k_cum_sum / step

            k_centered = K - k_running_mean

            q_hat = F.relu(Q) + eps  # [B, H, L, d]
            k_hat = F.relu(k_centered) + eps  # [B, H, N_k, d]

            # Causal linear attention via cumulative states
            kv = torch.einsum("b h j d, b h j e -> b h j d e", k_hat, V)  # [B, H, N_k, d, d]
            kv_cum = torch.cumsum(kv, dim=2)  # [B, H, N_k, d, d]
            k_cum = torch.cumsum(k_hat, dim=2)  # [B, H, N_k, d]

            sim = None
            if L == N_k:
                num = torch.einsum("b h l d, b h l d e -> b h l e", q_hat, kv_cum)  # [B, H, L, d]
                den = (q_hat * k_cum).sum(dim=-1, keepdim=True)  # [B, H, L, 1]
            elif L <= N_k:
                kv_cum_slice = kv_cum[:, :, -L:, :, :]
                k_cum_slice = k_cum[:, :, -L:, :]
                num = torch.einsum("b h l d, b h l d e -> b h l e", q_hat, kv_cum_slice)  # [B, H, L, d]
                den = (q_hat * k_cum_slice).sum(dim=-1, keepdim=True)  # [B, H, L, 1]
            else:
                sim = torch.matmul(q_hat, k_hat.transpose(-1, -2).contiguous())  # [B, H, L, N_k]
                causal_mask = torch.tril(torch.ones(L, N_k, device=device, dtype=torch.bool))
                sim = sim.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
                num = torch.matmul(sim, V)  # [B, H, L, d]
                den = sim.sum(dim=-1, keepdim=True)  # [B, H, L, 1]

            context = num / den.clamp(min=eps)  # [B, H, L, d]
            context = self.dropout(context)

            if output_attentions:
                if sim is not None:
                    attn_map = sim / den.clamp(min=eps)
                else:
                    attn_map = None
        else:
            # Bidirectional Mean-Centering
            valid_m = get_token_valid_mask(attention_mask, B, N_k, device)
            if valid_m is not None:
                mask_f = valid_m.unsqueeze(1).unsqueeze(-1).to(K.dtype)
                K = K * mask_f
                k_mean = K.sum(dim=2, keepdim=True) / mask_f.sum(dim=2, keepdim=True).clamp(min=1.0)
            else:
                k_mean = K.mean(dim=2, keepdim=True)

            k_centered = K - k_mean
            q_hat = F.relu(Q) + eps
            k_hat = F.relu(k_centered) + eps

            S_kv = torch.matmul(k_hat.transpose(-2, -1).contiguous(), V)  # [B, H, d, d]
            num = torch.matmul(q_hat, S_kv)                  # [B, H, L, d]
            den = torch.matmul(q_hat, k_hat.sum(dim=2, keepdim=True).transpose(-2, -1).contiguous()) + eps
            context = num / den.clamp(min=eps)
            context = self.dropout(context)

            if output_attentions:
                attn_map = torch.matmul(q_hat, k_hat.transpose(-1, -2).contiguous()) / den.clamp(min=eps)

        # Head mask broadcasting across 1D/2D/3D/4D
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            elif head_mask.dim() == 4:
                if head_mask.shape[-1] == head_mask.shape[-2] and head_mask.shape[-1] != context.shape[-1]:
                    head_mask = head_mask.mean(dim=-1, keepdim=True)
            context = context * head_mask.to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out(context)

        return output, attn_map


@ATTENTION_REGISTRY.register("zeros_attention_inject")
@ATTENTION_REGISTRY.register("zeros_inject")
def inject_zeros_attention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_zeros_attention(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    """
    Injection hook for ZeroSAttention.
    Replaces self-attention with Zero-Mean Linear Attention.
    target_attention_only: Replaces self-attention modules only, preserving all FFN/MLP blocks.
    """
    if old_attn is None:
        return None

    if model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    if model_config is None and hasattr(old_attn, "config"):
        model_config = old_attn.config

    hidden_size = (
        _cfg_get(model_config, "hidden_size")
        or _cfg_get(model_config, "d_model")
        or _cfg_get(model_config, "dim")
        or getattr(old_attn, "hidden_size", None)
        or getattr(old_attn, "embed_dim", None)
        or (old_attn.q_proj.in_features if hasattr(old_attn, "q_proj") and hasattr(old_attn.q_proj, "in_features") else None)
        or (old_attn.q_lin.in_features if hasattr(old_attn, "q_lin") and hasattr(old_attn.q_lin, "in_features") else None)
        or (old_attn.query.in_features if hasattr(old_attn, "query") and hasattr(old_attn.query, "in_features") else None)
        or 768
    )
    num_heads = (
        _cfg_get(model_config, "num_attention_heads")
        or _cfg_get(model_config, "num_heads")
        or _cfg_get(model_config, "n_heads")
        or getattr(old_attn, "num_heads", None)
        or getattr(old_attn, "num_attention_heads", None)
        or 12
    )

    if model_config is None:
        import types
        model_config = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=getattr(old_attn, "num_key_value_heads", num_heads),
            is_decoder=getattr(old_attn, "is_decoder", False),
        )

    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

    try:
        new_attn = ZeroSAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=old_attn,
            **kwargs,
        )

        for attr in ("self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_attn)
                return new_attn
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
            return new_attn
        layer.attention = new_attn
        return new_attn
    except Exception:
        return None
