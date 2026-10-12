"""
Grouped-Query Attention (GQA) — rank-aware, KV-cache aware, SDPA-accelerated.

Interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA)
by dividing H_Q query heads into G groups, where each group shares a single
Key-Value head pair (H_KV = G <= H_Q).
"""

from __future__ import annotations
from neuralops.core.common import _cfg_get
from neuralops.core.common import _unpack_weight_bias

from neuralops.core.attention_utils import (
    LoRALinear,
)

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half, apply_rotary_pos_emb

logger = logging.getLogger(__name__)




# =============================================================================
# Grouped-Query Attention
# =============================================================================


@ATTENTION_REGISTRY.register(
    "gqa_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
)
class GroupedQueryAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        dim: int | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        src_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---- hidden_size ----
        if dim is None and config is not None:
            dim = (
                _cfg_get(config, "hidden_size", None)
                or _cfg_get(config, "d_model", None)
                or _cfg_get(config, "n_embd", None)
                or _cfg_get(config, "dim", None)
            )
        if dim is None and src_attn is not None:
            dim = getattr(src_attn, "hidden_size", getattr(src_attn, "embed_dim", None))
            if dim is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(src_attn, cand, None)
                    if m is not None:
                        w, _ = _unpack_weight_bias(m)
                        if w is not None and w.dim() == 2:
                            dim = w.shape[1]
                            break
        self.hidden_size = int(dim or kwargs.get("hidden_size", 768))

        # ---- num_heads ----
        if num_heads is None and config is not None:
            num_heads = (
                _cfg_get(config, "num_attention_heads", None)
                or _cfg_get(config, "num_heads", None)
                or _cfg_get(config, "n_head", None)
            )
        if num_heads is None and src_attn is not None:
            num_heads = getattr(
                src_attn,
                "num_heads",
                getattr(src_attn, "num_attention_heads", getattr(src_attn, "n_head", None)),
            )
        self.num_heads = int(
            num_heads or kwargs.get("num_attention_heads", kwargs.get("num_heads", 8))
        )

        # ---- head_dim ----
        self.head_dim = int(
            kwargs.get("head_dim", getattr(config, "head_dim", None) if config else None)
            or (self.hidden_size // max(1, self.num_heads))
        )
# ==================================================================
        # تفکیک دقیق: rank مال LoRA است، num_kv_heads فشرده‌سازی GQA است
        # ==================================================================
        resolved_kv_heads = None

        if num_kv_heads is not None:
            resolved_kv_heads = int(num_kv_heads)
        elif config is not None:
            cfg_kv = _cfg_get(config, "num_key_value_heads", None) or _cfg_get(config, "num_kv_heads", None)
            if cfg_kv is not None:
                resolved_kv_heads = int(cfg_kv)
        
        if resolved_kv_heads is None:
            kwargs_kv = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads"))
            if kwargs_kv is not None:
                resolved_kv_heads = int(kwargs_kv)

        # اگر مشخص نشده بود یا برابر با کل هدها بود، به نسبت ۴ برابر فشرده شود
        if resolved_kv_heads is None or resolved_kv_heads >= self.num_heads:
            resolved_kv_heads = max(1, self.num_heads // 4)

        self.num_kv_heads = int(resolved_kv_heads)
        self._num_kv_heads_source = "gqa_compressed"

        # ---- sizes ----
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))



        # ---- decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            model_type = str(_cfg_get(config, "model_type", "")).lower()
            is_dec = _cfg_get(
                config,
                "is_decoder",
                _cfg_get(
                    config,
                    "is_causal",
                    kwargs.get("is_causal", kwargs.get("is_decoder", False)),
                ),
            )
            _dec_pfx = (
                "gpt",
                "qwen",
                "llama",
                "mistral",
                "gemma",
                "phi",
                "opt",
                "bloom",
                "falcon",
                "olmo",
                "mpt",
                "stablelm",
            )
            if any(model_type.startswith(p) for p in _dec_pfx):
                is_dec = True
            if src_attn is not None:
                if getattr(src_attn, "is_decoder", False) or hasattr(src_attn, "c_attn"):
                    is_dec = True
            self.is_decoder = bool(is_dec)
        self.is_nested_encoder = False

        # ---- LoRA ----
        comp_config = getattr(config, "compression_config", None) if config else None
        self.lora_rank = int(
            _cfg_get(
                comp_config,
                "lora_rank",
                _cfg_get(config, "lora_rank", kwargs.get("lora_rank", lora_rank)),
            )
        )
        self.lora_alpha = float(
            _cfg_get(
                comp_config,
                "lora_alpha",
                _cfg_get(config, "lora_alpha", kwargs.get("lora_alpha", lora_alpha)),
            )
        )
        self.freeze_base = self.lora_rank > 0
        # ---- projections ----
        self.query = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.key = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.value = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(float(dropout_p))
        self.rotary_emb = getattr(src_attn, "rotary_emb", None)

        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if src_attn is not None:
            self.copy_weights(src_attn)

    @property
    def q_proj(self) -> LoRALinear:
        return self.query

    @property
    def k_proj(self) -> LoRALinear:
        return self.key

    @property
    def v_proj(self) -> LoRALinear:
        return self.value

    @property
    def q_lin(self) -> LoRALinear:
        return self.query

    @property
    def k_lin(self) -> LoRALinear:
        return self.key

    @property
    def v_lin(self) -> LoRALinear:
        return self.value

    @property
    def out_lin(self) -> LoRALinear:
        return self.out_proj

    @property
    def dense(self) -> LoRALinear:
        return self.out_proj

    @property
    def o_proj(self) -> LoRALinear:
        return self.out_proj

    @property
    def c_proj(self) -> LoRALinear:
        return self.out_proj

    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        for proj in ("q", "k", "v", "o"):
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter failed for %s: %s", proj, e)

        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            weight, bias_data = _unpack_weight_bias(src.c_attn)
            if weight is not None:
                split_size = weight.shape[0] // 3
                extracted["q"] = (
                    weight[:split_size, :],
                    bias_data[:split_size] if bias_data is not None else None,
                )
                extracted["k"] = (
                    weight[split_size : 2 * split_size, :],
                    bias_data[split_size : 2 * split_size] if bias_data is not None else None,
                )
                extracted["v"] = (
                    weight[2 * split_size :, :],
                    bias_data[2 * split_size :] if bias_data is not None else None,
                )

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            w_o, b_o = _unpack_weight_bias(src.c_proj)
            extracted["o"] = (w_o, b_o)

        if hasattr(src, "self") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            s = src.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = _unpack_weight_bias(s.query)
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = _unpack_weight_bias(s.key)
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = _unpack_weight_bias(s.value)

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            extracted["o"] = _unpack_weight_bias(src.output.dense)

        named_map = {
            "q": ("q_proj", "q_lin", "query"),
            "k": ("k_proj", "k_lin", "key"),
            "v": ("v_proj", "v_lin", "value"),
            "o": ("o_proj", "out_proj", "out_lin", "dense"),
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for n in names:
                    if hasattr(src, n):
                        w, b = _unpack_weight_bias(getattr(src, n))
                        if w is not None:
                            extracted[proj] = (w, b)
                            break

        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, a)
            for a in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj")
        )
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        def _orient(w: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
            if w.dim() != 2:
                return w
            if w.shape == target_shape:
                return w
            if w.shape == (target_shape[1], target_shape[0]):
                return w.t().contiguous()
            return w

        # GQA-aware head selection / pooling (Head-Dimension Safe)
# GQA-aware head selection (Variance-Preserving)
        def _pool_gqa(w: torch.Tensor, target_heads: int, head_dim: int) -> torch.Tensor:
            in_feat = w.shape[1] if w.dim() == 2 else 1
            src_heads = w.shape[0] // head_dim

            if src_heads == target_heads:
                return w

            w_reshaped = w.view(src_heads, head_dim, in_feat)

            if src_heads > target_heads and src_heads % target_heads == 0:
                group_size = src_heads // target_heads
                # میانگین‌گیری روی هدهای هر گروه
                w_pooled = w_reshaped.view(target_heads, group_size, head_dim, in_feat).mean(dim=1)
            elif src_heads > target_heads:
                indices = torch.linspace(0, src_heads - 1, target_heads).long()
                w_pooled = w_reshaped.index_select(0, indices)
            else:
                rep = target_heads // src_heads
                w_pooled = w_reshaped.repeat_interleave(rep, dim=0)

            return w_pooled.reshape(target_heads * head_dim, in_feat)

        with torch.no_grad():
            for key, target in (
                ("q", self.query),
                ("k", self.key),
                ("v", self.value),
                ("o", self.out_proj),
            ):
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        min_d = min(target.weight.shape[0], target.weight.shape[1])
                        target.weight.data.zero_()
                        target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                        if target.bias is not None:
                            target.bias.data.zero_()
                    target.freeze_base_weights()
                    continue

                if key in ("k", "v"):
                    w = _pool_gqa(w, self.num_kv_heads, self.head_dim)
                    if b is not None:
                        b = _pool_gqa(b.unsqueeze(-1), self.num_kv_heads, self.head_dim).squeeze(-1)

                w = _orient(w, target.weight.shape)
                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data.zero_()
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mn = min(target.bias.numel(), b.numel())
                    target.bias.data.zero_()
                    target.bias.data[:mn].copy_(b[:mn].to(target.bias.device, target.bias.dtype))
                elif target.bias is not None:
                    target.bias.data.zero_()

                target.freeze_base_weights()

        try:
            for m in (self.query, self.key, self.value, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

    @staticmethod
    def _prepare_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int,
        L: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        def _to_bool(t: torch.Tensor) -> torch.Tensor:
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if attention_mask.dim() == 2:
            m = _to_bool(attention_mask)
            if m.shape == (B, L):
                return m
        elif attention_mask.dim() == 3:
            m = _to_bool(attention_mask)
            if m.shape[0] == B and m.shape[-1] == L:
                return m[:, -1, :]
        elif attention_mask.dim() == 4:
            m = _to_bool(attention_mask)
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m.squeeze(1)
            if m.shape == (B, L):
                return m
        return None

    @staticmethod
    def _read_cache(
        past_key_value: Any,
        layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        if past_key_value is None:
            return None, None, 0

        if isinstance(past_key_value, (tuple, list)):
            if len(past_key_value) == 0:
                return None, None, 0
            first = past_key_value[0]
            if isinstance(first, (tuple, list)):
                if layer_idx < len(past_key_value):
                    entry = past_key_value[layer_idx]
                    if len(entry) < 2 or entry[0] is None:
                        return None, None, 0
                    k, v = entry[0], entry[1]
                    return k, v, k.shape[-2]
                return None, None, 0
            k, v = past_key_value[0], past_key_value[1] if len(past_key_value) > 1 else None
            if k is None:
                return None, None, 0
            return k, v, k.shape[-2]

        if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
            if layer_idx < len(past_key_value.key_cache):
                k = past_key_value.key_cache[layer_idx]
                if k is None:
                    return None, None, 0
                v = None
                if hasattr(past_key_value, "value_cache"):
                    if layer_idx < len(past_key_value.value_cache):
                        v = past_key_value.value_cache[layer_idx]
                return k, v, k.shape[-2]

        return None, None, 0

    @staticmethod
    def _write_cache(
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: Any,
        layer_idx: int,
    ) -> Any:
        if past_key_value is not None and hasattr(past_key_value, "key_cache"):
            if layer_idx < len(past_key_value.key_cache):
                past_key_value.key_cache[layer_idx] = k
                if hasattr(past_key_value, "value_cache"):
                    past_key_value.value_cache[layer_idx] = v
            else:
                past_key_value.key_cache.append(k)
                if hasattr(past_key_value, "value_cache"):
                    past_key_value.value_cache.append(v)
            return past_key_value

        return (k, v)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if encoder_hidden_states is not None:
            return hidden_states, None
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        layer_idx = int(kwargs.get("layer_idx", 0))

        key_value_states = kwargs.get("key_value_states")
        is_cross = key_value_states is not None or encoder_hidden_states is not None
        kv_input = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else (key_value_states if key_value_states is not None else hidden_states)
        )

        B, N_q, D = hidden_states.shape
        N_k_new = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        past_k, past_v, past_len = (None, None, 0)
        if not is_cross and past_key_value is not None:
            past_k, past_v, past_len = self._read_cache(past_key_value, layer_idx)

        Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K = (
            self.key(kv_input)
            .view(B, N_k_new, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            self.value(kv_input)
            .view(B, N_k_new, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            position_ids = kwargs.get("position_ids")
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

        present_key_value: Any | None = None
        if not is_cross:
            if past_k is not None and past_v is not None:
                K = torch.cat([past_k.to(K.dtype), K], dim=-2)
                V = torch.cat([past_v.to(V.dtype), V], dim=-2)
            if past_key_value is not None or kwargs.get("use_cache", False):
                present_key_value = self._write_cache(K, V, past_key_value, layer_idx)
        else:
            present_key_value = past_key_value

        N_k = K.shape[-2]

        if self.kv_repeat > 1:
            K_full = repeat_kv(K, self.kv_repeat)
            V_full = repeat_kv(V, self.kv_repeat)
        else:
            K_full, V_full = K, V

        is_causal = bool(self.is_decoder and not is_cross and N_q > 1)
        attn_weights: torch.Tensor | None = None

        use_sdpa = (
            head_mask is None
            and not output_attentions
            and attention_mask is None
            and encoder_attention_mask is None
        )

        if use_sdpa:
            try:
                context = F.scaled_dot_product_attention(
                    Q,
                    K_full,
                    V_full,
                    attn_mask=None,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                )
            except Exception as e:
                logger.debug("GQA SDPA fast-path failed, falling back: %s", e)
                use_sdpa = False

        if not use_sdpa:
            scale = d**-0.5
            scores = torch.matmul(Q.float(), K_full.float().transpose(-2, -1).contiguous()) * scale

            if is_causal:
                q_idx = torch.arange(N_q, device=device).unsqueeze(1) + past_len
                k_idx = torch.arange(N_k, device=device).unsqueeze(0)
                causal_mask = (k_idx > q_idx).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(causal_mask, -1e4)

            eff_mask = encoder_attention_mask if is_cross else attention_mask
            if eff_mask is None:
                eff_mask = kwargs.get("mask")
            valid_mask = self._prepare_valid_mask(eff_mask, B, N_k, device)
            if valid_mask is not None:
                scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)

            attn = F.softmax(scores, dim=-1).to(dtype)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)

            if head_mask is not None:
                hm = head_mask.to(attn.dtype)
                if hm.dim() == 1:
                    hm = hm.view(1, H, 1, 1)
                elif hm.dim() == 2:
                    if hm.shape[-1] != H:
                        raise ValueError(
                            f"head_mask last dim {hm.shape[-1]} != num_heads {H}"
                        )
                    hm = hm.view(hm.shape[0], H, 1, 1)
                elif hm.dim() == 3:
                    hm = hm.unsqueeze(1)
                attn = attn * hm

            context = torch.matmul(attn, V_full)
            if output_attentions:
                attn_weights = attn

        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_decoder and not is_cross:
            if output_attentions:
                return output, present_key_value, attn_weights
            return output, present_key_value

        if output_attentions:
            return output, attn_weights
        return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.query.weight.device)

    def storage_bytes(self) -> dict[str, int]:
        def _sz(p: Any) -> int:
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out: dict[str, int] = {}
        for name, m in (
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("out_proj", self.out_proj),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["total"] = sum(out.values())
        return out


# =============================================================================
# Injection
# =============================================================================


def inject_gqa(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    rank: int | None = None,
    num_kv_heads: int | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Any | None:
    if _depth > 8:
        logger.warning("inject_gqa: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # ---- Whole-model traversal ----
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
        n_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_gqa(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                rank=rank,
                num_kv_heads=num_kv_heads,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    # ---- Single layer ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None:
        src_attn = target

    if (
        isinstance(src_attn, GroupedQueryAttention)
        or "GroupedQueryAttention" in src_attn.__class__.__name__
    ):
        return None

    if cfg is None:
        cfg = getattr(src_attn, "config", None) or getattr(target, "config", None)

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "dim", None)
        or (getattr(src_attn.query, "in_features", None) if hasattr(src_attn, "query") else None)
        or getattr(cfg, "hidden_size", 768)
    )
    num_heads = (
        getattr(cfg, "num_attention_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "num_heads", None)
        or 12
    )

    try:
        new_attn = GroupedQueryAttention(
            config=cfg,
            pretrained_attn=src_attn if src_attn is not target else None,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            rank=rank,
            num_kv_heads=num_kv_heads,
            **kwargs,
        )
    except Exception as e:
        logger.error("GQA construction failed: %s", e)
        return None

    # ---- Safe device/dtype placement ----
    ref_param = None
    if src_attn is not None and src_attn is not target:
        ref_param = next(src_attn.parameters(), None)
    if ref_param is None:
        ref_param = next(target.parameters(), None)
    if ref_param is not None:
        new_attn.to(device=ref_param.device, dtype=ref_param.dtype)

    # ---- Splice into the parent module ----
    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            new_attn.is_nested_encoder = True
            target.attention.self = new_attn
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn
        spliced = True
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
        spliced = True
    elif hasattr(target, "attn"):
        new_attn.is_nested_encoder = False
        target.attn = new_attn
        spliced = True
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
        spliced = True

    if not spliced:
        logger.error("inject_gqa: no writable attention attribute on %s", type(target).__name__)
        return None

    return new_attn


__all__ = [
    "GroupedQueryAttention",
    "inject_gqa",
    "apply_rotary_pos_emb",
    "LoRALinear",
]