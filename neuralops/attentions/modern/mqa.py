"""
Multi-Query Attention (MQA) — production implementation.

Multi-Query Attention uses H_Q query heads but only 1 shared key/value head.
This reduces KV-cache memory by H_Q × compared to standard MHA.

"""

from __future__ import annotations

from neuralops.core.attention_utils import (
    LoRALinear,
)

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.amp_utils import maybe_half
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half, apply_rotary_pos_emb
from neuralops.core.gqa_utils import repeat_kv

logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract (weight, bias) from an arbitrary linear/Conv1D/LoRA-wrapped module."""
    if module is None:
        return None, None
    if hasattr(module, "base_layer"):
        module = module.base_layer
    w = getattr(module, "weight", None)
    b = getattr(module, "bias", None)
    cls_name = module.__class__.__name__

    if w is not None:
        w = w.data
        # GPT-2 style Conv1D has (in, out) layout → transpose to (out, in)
        if hasattr(module, "nf") or "Conv1D" in cls_name:
            w = w.t().contiguous()
    if b is not None:
        b = b.data

    return w, b


# ===========================================================================
# Multi-Query Attention
# ===========================================================================


@ATTENTION_REGISTRY.register("mqa", supports_causal=True, supports_gqa=True)
class MultiQueryAttention(BaseAttention):
    """
    Multi-Query Attention (MQA) — Shazeer 2019.

    MQA is a special case of GQA with exactly ONE key/value head shared
    across all H_Q query heads. This yields:
      - KV-cache reduction by factor of H_Q (e.g., 12×)
      - Faster inference (fewer memory reads)
      - Slight quality drop that can be recovered via KD

    Architecture:
        q = q_proj(x)                     # (B, N, H*D)
        k = k_proj(x)                     # (B, N, D) — single KV head
        v = v_proj(x)                     # (B, N, D)
        attn = softmax(q @ k^T / √D) @ v  # broadcast k, v across H
        out = out_proj(attn)
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        num_kv_heads: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        src_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---------------- Dimensions ----------------
        h_size = (
            hidden_size
            or kwargs.get("embed_dim")
            or kwargs.get("d_model")
            or kwargs.get("dim")
            or _cfg_get(config, "hidden_size")
            or _cfg_get(config, "embed_dim")
            or _cfg_get(config, "d_model")
            or _cfg_get(config, "n_embd")
            or _cfg_get(config, "dim")
        )
        if h_size is None and src_attn is not None:
            for cand in ("hidden_size", "embed_dim", "all_head_size", "dim"):
                h_size = getattr(src_attn, cand, None)
                if h_size is not None:
                    break
            if h_size is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(src_attn, cand, None)
                    if m is not None:
                        w, _ = _unpack_weight_bias(m)
                        if w is not None and w.dim() == 2:
                            h_size = w.shape[0]
                            break
        self.hidden_size = int(h_size or 768)
        self.d_model = self.hidden_size

        n_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or kwargs.get("n_heads")
            or _cfg_get(config, "num_attention_heads")
            or _cfg_get(config, "num_heads")
            or _cfg_get(config, "n_head")
        )
        if n_heads is None and src_attn is not None:
            for k in ("num_heads", "num_attention_heads", "n_heads"):
                n_heads = getattr(src_attn, k, None)
                if n_heads is not None:
                    break
        self.num_heads = int(n_heads or 12)

        self.head_dim = int(
            head_dim
            or kwargs.get("head_dim")
            or _cfg_get(config, "head_dim")
            or (self.hidden_size // max(1, self.num_heads))
        )
        self.all_head_size = self.num_heads * self.head_dim

        # ---------------- MQA: single KV head ----------------
        # If explicit num_kv_heads given and > 1, this becomes GQA-like.
        # Default: exactly 1 (true MQA).
        if num_kv_heads is not None:
            self.num_kv_heads = max(1, int(num_kv_heads))
        else:
            self.num_kv_heads = 1  # ← MQA

        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # ---------------- Decoder detection ----------------
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            model_type = str(_cfg_get(config, "model_type", "")).lower()
            _dec_pfx = (
                "gpt", "qwen", "llama", "mistral", "gemma", "phi",
                "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
            )
            is_dec = bool(_cfg_get(config, "is_decoder", False))
            is_dec = is_dec or bool(_cfg_get(config, "causal", False))
            is_dec = is_dec or any(model_type.startswith(p) for p in _dec_pfx)
            if src_attn is not None:
                if getattr(src_attn, "is_decoder", False) or hasattr(src_attn, "c_attn"):
                    is_dec = True
            self.is_decoder = bool(is_dec)

        self.is_nested_encoder: bool = False
        self.layer_idx: int = int(kwargs.get("layer_idx", 0))

        # ---------------- LoRA config ----------------
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

        # ---------------- Projections ----------------
        self.q_proj = LoRALinear(
            self.hidden_size, self.all_head_size,
            bias=bias, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.hidden_size, self.kv_head_size,
            bias=bias, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.hidden_size, self.kv_head_size,
            bias=bias, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.hidden_size,
            bias=bias, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # ---------------- Dropout ----------------
        dropout_p = float(
            dropout
            or _cfg_get(config, "dropout")
            or _cfg_get(config, "attention_probs_dropout_prob")
            or _cfg_get(config, "hidden_dropout_prob")
            or 0.0
        )
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_p = dropout_p
        self.rotary_emb = getattr(src_attn, "rotary_emb", None) if src_attn is not None else None

        if src_attn is not None:
            self.copy_weights(src_attn)

    # ------------------------------------------------------------------
    # Aliases for multi-architecture compatibility
    # ------------------------------------------------------------------
    @property
    def query(self) -> LoRALinear:
        return self.q_proj

    @property
    def key(self) -> LoRALinear:
        return self.k_proj

    @property
    def value(self) -> LoRALinear:
        return self.v_proj

    @property
    def q_lin(self) -> LoRALinear:
        return self.q_proj

    @property
    def k_lin(self) -> LoRALinear:
        return self.k_proj

    @property
    def v_lin(self) -> LoRALinear:
        return self.v_proj

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

    # ------------------------------------------------------------------
    # Weight copying (multi-architecture + MQA head pooling)
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        """
        Copy weights from a pretrained attention module.

        For K and V, since MQA has 1 KV head but source may have H_kv,
        we pool heads (mean-pool if H_kv > 1, repeat if H_kv < 1).
        """
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # ----- 1. UniversalAdapter fallback -----
        for proj in ("q", "k", "v", "o"):
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("MQA UniversalAdapter %s failed: %s", proj, e)

        # ----- 2. GPT-2 fused c_attn -----
        if hasattr(src, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            weight, bias = _unpack_weight_bias(src.c_attn)
            if weight is not None:
                split = weight.shape[0] // 3
                extracted["q"] = (
                    weight[:split],
                    bias[:split] if bias is not None else None,
                )
                extracted["k"] = (
                    weight[split : 2 * split],
                    bias[split : 2 * split] if bias is not None else None,
                )
                extracted["v"] = (
                    weight[2 * split :],
                    bias[2 * split :] if bias is not None else None,
                )

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            extracted["o"] = _unpack_weight_bias(src.c_proj)

        # ----- 3. BERT / RoBERTa nested -----
        if hasattr(src, "self") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            s = src.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = _unpack_weight_bias(s.query)
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = _unpack_weight_bias(s.key)
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = _unpack_weight_bias(s.value)

        if (
            hasattr(src, "output")
            and hasattr(src.output, "dense")
            and extracted["o"][0] is None
        ):
            extracted["o"] = _unpack_weight_bias(src.output.dense)

        # ----- 4. Named attribute fallback -----
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

        # ----- 5. Nested encoder detection -----
        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, a)
            for a in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj")
        )
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        # ----- 6. Validate required weights -----
        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"MultiQueryAttention.copy_weights: missing {missing} from "
                f"{src.__class__.__name__}. Refusing random init."
            )

        # ----- 7. Helpers -----
        def _orient(w: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
            if w.dim() != 2:
                return w
            if w.shape == target_shape:
                return w
            if w.shape == (target_shape[1], target_shape[0]):
                return w.t().contiguous()
            return w

        def _pool_kv_heads(w: torch.Tensor, target_heads: int, head_dim: int) -> torch.Tensor:
            """
            Pool source K/V weight to target (num_kv_heads).
            For MQA: target_heads = 1.
            """
            if w.dim() != 2:
                return w
            src_dim = w.shape[0]
            if src_dim == target_heads * head_dim:
                return w

            src_heads = src_dim // head_dim

            if src_heads == target_heads:
                return w

            w_reshaped = w.view(src_heads, head_dim, -1)

            if src_heads > target_heads and src_heads % target_heads == 0:
                group_size = src_heads // target_heads
                w_pooled = w_reshaped.view(
                    target_heads, group_size, head_dim, -1
                ).mean(dim=1)
            elif src_heads > target_heads:
                idx = torch.linspace(0, src_heads - 1, target_heads).long()
                w_pooled = w_reshaped.index_select(0, idx.to(w.device))
            else:
                rep = target_heads // src_heads
                w_pooled = w_reshaped.repeat_interleave(rep, dim=0)

            return w_pooled.reshape(target_heads * head_dim, -1).contiguous()

        # ----- 8. Copy -----
        with torch.no_grad():
            for key, target in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
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

                w = _orient(w, target.weight.shape)

                # MQA: pool K and V down to a single head
                if key in ("k", "v"):
                    w = _pool_kv_heads(w, self.num_kv_heads, self.head_dim)
                    if b is not None:
                        b = _pool_kv_heads(
                            b.unsqueeze(-1), self.num_kv_heads, self.head_dim
                        ).squeeze(-1)

                out_d = min(target.weight.shape[0], w.shape[0])
                in_d = min(target.weight.shape[1], w.shape[1])

                target.weight.data.zero_()
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data.zero_()
                    target.bias.data[:mb].copy_(
                        b[:mb].to(target.bias.device, target.bias.dtype)
                    )
                elif target.bias is not None:
                    target.bias.data.zero_()

                target.freeze_base_weights()

        # ----- 9. Preserve decoder flag / rotary -----
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
        if getattr(src, "rotary_emb", None) is not None:
            self.rotary_emb = src.rotary_emb

    # ------------------------------------------------------------------
    # KV-cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_cache(past_key_value: Any, layer_idx: int) -> tuple[
        torch.Tensor | None, torch.Tensor | None, int
    ]:
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
                    return entry[0], entry[1], entry[0].shape[-2]
                return None, None, 0
            k = past_key_value[0]
            v = past_key_value[1] if len(past_key_value) > 1 else None
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
        k: torch.Tensor, v: torch.Tensor, past_key_value: Any, layer_idx: int
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
    ):
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N_q, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_k_new = kv_input.shape[1]

        # ---- Projections ----
        q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        k_new = (
            self.k_proj(kv_input)
            .view(B, N_k_new, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        v_new = (
            self.v_proj(kv_input)
            .view(B, N_k_new, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE with past offset ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            position_ids = kwargs.get("position_ids")
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, position_ids)
        elif self.rotary_emb is not None and not is_cross:
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                position_ids = (
                    torch.arange(N_q, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
            try:
                cos, sin = self.rotary_emb(k_new, position_ids=position_ids)
            except TypeError:
                try:
                    cos, sin = self.rotary_emb(k_new, position_ids)
                except Exception:
                    cos, sin = None, None
            if cos is not None and sin is not None:
                q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, position_ids)

        # ---- KV-cache ----
        past_k = past_v = None
        past_len = 0
        if not is_cross and past_key_value is not None:
            past_k, past_v, past_len = self._read_cache(past_key_value, self.layer_idx)

        if not is_cross:
            if past_k is not None and past_v is not None:
                k = torch.cat([past_k.to(k_new.dtype), k_new], dim=-2)
                v = torch.cat([past_v.to(v_new.dtype), v_new], dim=-2)
            else:
                k, v = k_new, v_new
        else:
            k, v = k_new, v_new

        N_k = k.shape[-2]

        # ---- GQA expansion (MQA = repeat_kv by H) ----
        if self.kv_repeat > 1:
            k_full = repeat_kv(k, self.kv_repeat)
            v_full = repeat_kv(v, self.kv_repeat)
        else:
            k_full, v_full = k, v

        # ---- Update cache ----
        present_key_value: Any | None = None
        if not is_cross:
            if past_key_value is not None or kwargs.get("use_cache", False):
                present_key_value = self._write_cache(k, v, past_key_value, self.layer_idx)
        else:
            present_key_value = past_key_value

        is_causal = bool(self.is_decoder and not is_cross and N_q > 1)

        # ---- Mask preparation ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None:
            eff_mask = kwargs.get("mask")

        # ---- SDPA fast path ----
        use_sdpa = (
            head_mask is None
            and not output_attentions
            and eff_mask is None
        )
        attn_weights: torch.Tensor | None = None

        if use_sdpa:
            try:
                _qs = maybe_half(q)
                _ks = maybe_half(k_full)
                _vs = maybe_half(v_full)
                context = F.scaled_dot_product_attention(
                    _qs, _ks, _vs,
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=is_causal,
                )
                context = context.to(dtype)
            except Exception as e:
                logger.debug("MQA SDPA fast-path failed (%s); falling back", e)
                use_sdpa = False

        # ---- Manual path ----
        if not use_sdpa:
            scale = 1.0 / math.sqrt(d)
            scores = torch.matmul(q.float(), k_full.float().transpose(-2, -1).contiguous()) * scale

            if is_causal:
                q_idx = torch.arange(N_q, device=device).unsqueeze(1) + past_len
                k_idx = torch.arange(N_k, device=device).unsqueeze(0)
                causal_mask = (k_idx > q_idx).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(causal_mask, _safe_mask_val(scores.dtype))

            if eff_mask is not None:
                if eff_mask.dim() == 2:
                    m = eff_mask[:, None, None, :].to(torch.bool)
                    if m.dtype == torch.bool:
                        scores = scores.masked_fill(~m, _safe_mask_val(scores.dtype))
                elif eff_mask.dim() == 3:
                    m = eff_mask.unsqueeze(1).to(torch.bool)
                    scores = scores.masked_fill(~m, _safe_mask_val(scores.dtype))
                elif eff_mask.dim() == 4:
                    m = eff_mask.to(torch.bool)
                    scores = scores.masked_fill(~m, _safe_mask_val(scores.dtype))

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn = F.softmax(scores, dim=-1).to(dtype)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)

            if head_mask is not None:
                hm = head_mask.to(attn.dtype)
                if hm.dim() == 1:
                    hm = hm.view(1, H, 1, 1)
                elif hm.dim() == 2:
                    hm = hm.view(hm.shape[0], H, 1, 1)
                elif hm.dim() == 3:
                    hm = hm.unsqueeze(1)
                attn = attn * hm

            context = torch.matmul(attn, v_full)
            if output_attentions:
                attn_weights = attn

        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Return format ----
        if output_attentions:
            if self.is_decoder and not is_cross:
                return output, present_key_value, attn_weights
            return output, attn_weights
        return output, present_key_value

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.q_proj.weight.device)

    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            return p.numel() * p.element_size() if isinstance(p, (nn.Parameter, torch.Tensor)) else 0
        out: dict[str, int] = {}
        for name, m in (("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["total"] = sum(out.values())
        return out


def _safe_mask_val(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


# ===========================================================================
# Injection
# ===========================================================================


def inject_mqa(
    model: nn.Module | None = None,
    layer: nn.Module | None = None,
    model_config: Any | None = None,
    comp_config: Any | None = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    num_kv_heads: int | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> MultiQueryAttention | nn.Module | None:
    if _depth > 8:
        logger.warning("inject_mqa: max recursion depth exceeded.")
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
            res = inject_mqa(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
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

    # idempotency
    if isinstance(src_attn, MultiQueryAttention) or "MultiQueryAttention" in src_attn.__class__.__name__:
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
        new_attn = MultiQueryAttention(
            config=cfg,
            pretrained_attn=src_attn if src_attn is not target else None,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            num_kv_heads=num_kv_heads,
            **kwargs,
        )
    except Exception as e:
        logger.error("MQA construction failed: %s", e)
        return None

    # ---- Device/dtype placement ----
    ref_param = None
    if src_attn is not None and src_attn is not target:
        ref_param = next(src_attn.parameters(), None)
    if ref_param is None:
        ref_param = next(target.parameters(), None)
    if ref_param is not None:
        new_attn.to(device=ref_param.device, dtype=ref_param.dtype)

    # ---- Splice ----
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
        logger.error("inject_mqa: no writable attention attribute on %s", type(target).__name__)
        return None

    return new_attn


__all__ = [
    "MultiQueryAttention",
    "LoRALinear",
    "inject_mqa",
    "apply_rotary_pos_emb",
]