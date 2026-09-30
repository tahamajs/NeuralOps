"""
LoRA Parameter-Efficient Fine-Tuning Attention Module (v2, Production-Hardened).

Replaces standard Q/K/V/O projections with frozen pretrained base weights plus
trainable low-rank residual adapters, with multi-architecture weight transfer,
GQA, RoPE, and causal decoding support.

This is intentionally an attention-swap module, NOT a compressor:
  - Base weights are frozen.
  - Only LoRA A/B are trainable.
  - Parameters are NOT reduced (compression_rate stays ~1.0).
  - Its purpose is PEFT, not compression.

Fixes (v2):
  LORA2-FIX-01  `actual_rank` is now purely the LoRA rank. `rank` (compression
                rank) is no longer conflated with LoRA rank.
  LORA2-FIX-02  Removed duplicated `if hasattr(layer, 'attention')` block.
  LORA2-FIX-03  Removed unreachable `elif hasattr(layer, 'attention')`.
  LORA2-FIX-04  Causal mask applied only for self-attention.
  LORA2-FIX-05  Robust `cfg = None` handling.
  LORA2-FIX-06  Proper `is_nested_encoder` propagation.
  LORA2-FIX-07  Idempotency guard in injector.
  LORA2-FIX-08  Storage accounting (`orig_storage_bytes`, `comp_storage_bytes`).
  LORA2-FIX-09  LoRA init: kaiming A, zeros B, dtype-safe.
  LORA2-FIX-10  `position_embeddings` accepted as explicit param + kwargs.
  LORA2-FIX-11  Strict mask handling with normalized valid-mask shape.
  LORA2-FIX-12  Deterministic injector recursion.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE (dtype-safe)
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
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


# ===========================================================================
# LoRALinear (dtype-safe)
# ===========================================================================
class LoRALinear(nn.Module):
    """
    Linear layer with frozen base weights and trainable LoRA adapters.

    Forward:
        y = x W^T + b + (alpha / r) * x A^T B^T
    where W, b are frozen, and A, B are trainable.
    """
    is_custom_attention: bool = False

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
        fk = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0
        self.freeze_base = bool(freeze_base)

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((self.lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.zeros((out_features, self.lora_rank), **fk))
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

        # LORA2-FIX-09: A kaiming, B zeros (dtype-safe)
        if self.lora_rank > 0 and self.lora_A is not None:
            if self.lora_A.dtype in (torch.float16, torch.bfloat16):
                tmp = torch.empty_like(self.lora_A, dtype=torch.float32, device="cpu")
                nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
                self.lora_A.data.copy_(tmp.to(self.lora_A.dtype))
            else:
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
        # dtype-safe
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        out = F.linear(x, w, b)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            lora_out = F.linear(F.linear(x, a), bb) * self.lora_scaling
            out = out + lora_out
        return out


# ===========================================================================
# Main Attention
# ===========================================================================
@ATTENTION_REGISTRY.register("lora_attn",
                              supports_causal=True,
                              supports_bidirectional=True,
                              supports_gqa=True)
@ATTENTION_REGISTRY.register("lora_attention",
                              supports_causal=True,
                              supports_bidirectional=True,
                              supports_gqa=True)
class LoRAAttention(BaseAttention):
    """
    LoRA attention wrapper.

    NOTE: This module is deliberately a PEFT swap, NOT a compressor.
    `compression_rate` remains ~1.0; `num_compressed_layers` remains 0.
    Its role is to provide a trainable LoRA residual on top of frozen
    pretrained attention weights.
    """
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        model_config: Optional[Any] = None,
        rank: Optional[int] = None,           # accepted for compat, ignored
        lora_rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config if config is not None else model_config
        # LORA2-FIX-01: LoRA rank is what actually matters. `rank` from
        # the sweep (compression rank) is not used for LoRA.
        actual_lora_rank = int(lora_rank) if lora_rank is not None else 16

        super().__init__(cfg, actual_lora_rank)
        # LORA2-FIX-05: cfg may be None
        if cfg is None:
            cfg = getattr(pretrained_attn, "config", None)

        self.adapter = ModelAdapter(cfg)
        self.num_heads = int(self.adapter.num_heads)
        self.num_kv_heads = int(self.adapter.num_key_value_heads)
        self.hidden_size = int(self.adapter.hidden_size)
        self.head_dim = int(self.adapter.head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = int(self.adapter.kv_repeat)
        self.is_decoder = bool(getattr(cfg, "is_decoder", False)) if cfg is not None else False
        self.is_nested_encoder = False

        self.lora_rank = actual_lora_rank
        self.alpha = float(getattr(cfg, "lora_alpha", alpha)) if cfg is not None else float(alpha)

        self.q_proj = LoRALinear(self.hidden_size, self.all_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.alpha,
                                  freeze_base=True)
        self.k_proj = LoRALinear(self.hidden_size, self.kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.alpha,
                                  freeze_base=True)
        self.v_proj = LoRALinear(self.hidden_size, self.kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.alpha,
                                  freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.hidden_size, bias=True,
                                    lora_rank=self.lora_rank, lora_alpha=self.alpha,
                                    freeze_base=True)

        dropout_p = float(
            getattr(cfg, "dropout",
                    getattr(cfg, "hidden_dropout_prob",
                            getattr(cfg, "attention_probs_dropout_prob", dropout)))
            if cfg is not None else dropout
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        # LORA2-FIX-08: storage accounting
        self._orig_storage_bytes: int = 0
        self._comp_storage_bytes: int = 0

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def dense(self): return self.out_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        # Track original bytes
        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None),
            "v": (None, None), "o": (None, None),
        }

        # 1. UniversalAdapter
        for proj in ("q", "k", "v", "o"):
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        # 2. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and any(extracted[p][0] is None for p in ("q", "k", "v")):
            c_attn = src.c_attn
            base_c = getattr(c_attn, "base_layer", c_attn)
            cw = getattr(base_c, "weight", getattr(c_attn, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                is_conv = (c_attn.__class__.__name__ == "Conv1D"
                           or base_c.__class__.__name__ == "Conv1D"
                           or (cw.ndim == 2 and cw.shape[0] < cw.shape[1]))
                if is_conv:
                    cw = cw.t().contiguous()
                split = cw.size(0) // 3
                bmod = getattr(base_c, "bias", getattr(c_attn, "bias", None))
                cb = bmod.data.clone() if bmod is not None else None
                extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
                extracted["k"] = (cw[split:2*split], cb[split:2*split] if cb is not None else None)
                extracted["v"] = (cw[2*split:], cb[2*split:] if cb is not None else None)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj = src.c_proj
            base_o = getattr(c_proj, "base_layer", c_proj)
            ow = getattr(base_o, "weight", getattr(c_proj, "weight", None))
            if ow is not None:
                ow = ow.data.clone()
                is_conv_o = (c_proj.__class__.__name__ == "Conv1D"
                             or base_o.__class__.__name__ == "Conv1D"
                             or (ow.ndim == 2 and ow.shape[0] < ow.shape[1]))
                if is_conv_o:
                    ow = ow.t().contiguous()
                bo = getattr(base_o, "bias", getattr(c_proj, "bias", None))
                extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # 3. Nested BERT / RoBERTa
        if hasattr(src, "self") and any(extracted[p][0] is None for p in ("q", "k", "v")):
            s = getattr(src, "self")
            if hasattr(s, "query") and extracted["q"][0] is None:
                m = s.query
                extracted["q"] = (m.weight.data, m.bias.data if m.bias is not None else None)
            if hasattr(s, "key") and extracted["k"][0] is None:
                m = s.key
                extracted["k"] = (m.weight.data, m.bias.data if m.bias is not None else None)
            if hasattr(s, "value") and extracted["v"][0] is None:
                m = s.value
                extracted["v"] = (m.weight.data, m.bias.data if m.bias is not None else None)

        if (hasattr(src, "output") and hasattr(getattr(src, "output"), "dense")
                and extracted["o"][0] is None):
            d = src.output.dense
            extracted["o"] = (d.weight.data, d.bias.data if d.bias is not None else None)

        # 4. Modern decoders and flat modules
        named_map = {
            "q": ("q_proj", "q_lin", "query"),
            "k": ("k_proj", "k_lin", "key"),
            "v": ("v_proj", "v_lin", "value"),
            "o": ("o_proj", "out_proj", "out_lin", "dense"),
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        if hasattr(layer, "weight"):
                            b = getattr(layer, "bias", None)
                            extracted[proj] = (
                                layer.weight.data,
                                b.data if b is not None else None,
                            )
                            break

        # Detect nested encoder
        cls_name = src.__class__.__name__
        has_out_proj = any(hasattr(src, attr) for attr in
                           ("dense", "out_proj", "c_proj", "out_lin",
                            "o_proj", "wo", "o", "proj"))
        is_nested = False
        if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            is_nested = True
        elif (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("LoRAAttention copy_weights: missing %s", missing)

        # Frobenius norm checks
        for k in ("q", "k", "v"):
            w = extracted[k][0]
            if w is not None:
                n = torch.linalg.norm(w.float()).item()
                if n < 1e-4:
                    logger.warning("LoRAAttention: '%s' near-zero norm (%.2e)", k, n)
        if extracted["o"][0] is not None:
            n = torch.linalg.norm(extracted["o"][0].float()).item()
            if n < 1e-4:
                logger.warning("LoRAAttention: 'o' near-zero norm (%.2e)", n)

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for key, target in (("q", self.q_proj), ("k", self.k_proj),
                                 ("v", self.v_proj), ("o", self.out_proj)):
                w, b = extracted[key]
                if w is not None:
                    in_dim = self.all_head_size if key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    out_d, in_d = target.weight.shape
                    src_out, src_in = w_oriented.shape
                    # K/V: GQA-safe copy with mean-pooling if needed
                    if key in ("k", "v") and src_out != out_d:
                        if src_out > out_d and src_out % out_d == 0:
                            rep = src_out // out_d
                            w_pooled = w_oriented.view(out_d, rep, -1).mean(dim=1)
                            target.weight.data.copy_(w_pooled)
                            if b is not None and target.bias is not None:
                                bp = b.view(out_d, rep).mean(dim=1)
                                target.bias.data.copy_(bp)
                        elif src_out < out_d and out_d % src_out == 0:
                            rep = out_d // src_out
                            target.weight.data.copy_(w_oriented.repeat(rep, 1))
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b.repeat(rep))
                        else:
                            m0, m1 = min(out_d, src_out), min(in_d, src_in)
                            target.weight.data.zero_()
                            target.weight.data[:m0, :m1].copy_(w_oriented[:m0, :m1])
                            if b is not None and target.bias is not None:
                                mb = min(target.bias.shape[0], b.shape[0])
                                target.bias.data.zero_()
                                target.bias.data[:mb].copy_(b[:mb])
                    else:
                        m0, m1 = min(out_d, src_out), min(in_d, src_in)
                        target.weight.data.zero_()
                        target.weight.data[:m0, :m1].copy_(w_oriented[:m0, :m1])
                        if b is not None and target.bias is not None:
                            mb = min(target.bias.shape[0], b.shape[0])
                            target.bias.data.zero_()
                            target.bias.data[:mb].copy_(b[:mb])
                elif key == "o" and self.is_nested_encoder:
                    md = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:md, :md].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        # Track compressed bytes
        try:
            for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    # ------------------------------------------------------------------
    # Mask helper (LORA2-FIX-11)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_valid_mask(attention_mask, B, N_kv, device):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                m = attention_mask
            elif attention_mask.dtype.is_floating_point:
                m = attention_mask > -1e4
            else:
                m = attention_mask > 0
            return m if m.shape == (B, N_kv) else None
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            m = v.any(dim=-2).squeeze(1)
            return m if m.shape == (B, N_kv) else None
        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross = key_value_states is not None
        kv_input = key_value_states if is_cross else hidden_states

        B, N_q, _ = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # RoPE
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if (position_embeddings is not None and not is_cross
                and isinstance(position_embeddings, (tuple, list))
                and len(position_embeddings) >= 2):
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))

        # GQA repeat
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = d ** -0.5
        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        # Attention mask
        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        valid_m = self._normalize_valid_mask(mask, B, N_k, device)
        if valid_m is not None:
            scores = scores.masked_fill(~valid_m[:, None, None, :], -1e4)

        # LORA2-FIX-04: causal only for self-attention
        is_decoder = bool(
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in
               ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )
        if is_decoder and not is_cross:
            causal = torch.tril(torch.ones(N_q, N_k, device=device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        if head_mask is not None:
            hm = head_mask.to(attn.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            attn = attn * hm

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        return output, (attn if output_attentions else None)


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("lora_attention_inject")
@ATTENTION_REGISTRY.register("inject_lora_attention")
def inject_lora_attention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_lora_attention: recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_lora_attention(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    # Single-layer injection
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attn", "attention", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None:
        src_attn = target

    # LORA2-FIX-07: idempotency guard
    if isinstance(src_attn, LoRAAttention) or "LoRAAttention" in src_attn.__class__.__name__:
        return None

    if cfg is None:
        cfg = getattr(src_attn, "config", None) or getattr(target, "config", None)

    try:
        # LORA2-FIX-01: lora_rank is the meaningful knob for LoRA.
        # `rank` from sweep is ignored.
        lora_rank = int(kwargs.pop("lora_rank", 16))
        if comp_config is not None:
            lora_rank = int(getattr(comp_config, "lora_rank", lora_rank))
        alpha = float(kwargs.pop("alpha", 32.0))
        if comp_config is not None:
            alpha = float(getattr(comp_config, "lora_alpha", alpha))

        new_attn = LoRAAttention(
            config=cfg,
            lora_rank=lora_rank,
            alpha=alpha,
            pretrained_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )
        p = next(src_attn.parameters(), None) if src_attn is not target else None
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        # LORA2-FIX-02/03: single, correct splice branch
        if hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
        else:
            logger.error("inject_lora_attention: no writable attention attribute on %s",
                          target.__class__.__name__)
            return None

        # Propagate training mode (LORA2-FIX-06)
        if src_attn is not target:
            new_attn.train(src_attn.training)
        return new_attn

    except Exception as e:
        logger.exception("inject_lora_attention failed at layer %d: %s", layer_idx, e)
        return None


__all__ = [
    "LoRAAttention",
    "LoRALinear",
    "inject_lora_attention",
    "apply_rotary_pos_emb",
]