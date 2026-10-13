"""
Quality and Capacity-Aware Grouped Query Attention (QCQA).

"""

from __future__ import annotations
from neuralops.core.common import _read_past_kv
from neuralops.core.common import _cfg_get
from neuralops.core.common import _detect_decoder

from neuralops.core.attention_utils import (
    apply_rotary_pos_emb,
    LoRALinear,
)

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from neuralops.core.amp_utils import maybe_half
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half








# ===========================================================================
# QCQA Attention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "qcqa_attention",
    supports_causal=True,
    supports_gqa=True,
    supports_rope=True,
)
class QCQAAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        num_kv_heads: int | None = None,
        router_temp: float = 1.0,
        pretrained_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        rank = kwargs.pop("rank", 64)
        super().__init__(config, rank)
        self.config = config
        comp_config = kwargs.get("comp_config") or (
            getattr(config, "compression_config", None) if config else None
        )

        # ---- hidden_size ----
        h_size = (
            kwargs.get("hidden_size")
            or getattr(config, "hidden_size", None)
            or getattr(config, "d_model", None)
        )
        if h_size is None and pretrained_attn is not None:
            for cand in ("q_proj", "q_lin", "query", "c_attn"):
                mod = getattr(pretrained_attn, cand, None)
                if mod is not None and hasattr(mod, "weight"):
                    w = mod.weight
                    if w.ndim == 2:
                        if mod.__class__.__name__ == "Conv1D" or hasattr(mod, "nf"):
                            h_size = w.shape[0]
                        else:
                            h_size = w.shape[1]
                        break
        if h_size is None:
            raise ValueError("QCQAAttention: could not infer hidden_size.")
        self.dim = int(h_size)

        self.num_heads = int(
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(config, "num_attention_heads", None)
            or getattr(config, "num_heads", None)
            or 12
        )

        cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if num_kv_heads is not None:
            self.num_kv_heads = int(num_kv_heads)
        elif cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        else:
            self.num_kv_heads = int(kwargs.get("num_kv_heads", 4))
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.head_dim = int(
            kwargs.get("head_dim")
            or getattr(config, "head_dim", None)
            or getattr(config, "d_kv", None)
            or (self.dim // max(1, self.num_heads))
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        self.router_temp = float(
            _cfg_get(comp_config, "router_temp", kwargs.get("router_temp", router_temp))
        )
        if self.router_temp <= 0:
            self.router_temp = 1.0

        self.is_decoder = _detect_decoder(config, is_decoder, pretrained_attn)
        self.is_nested_encoder = False

        self.lora_rank = int(
            _cfg_get(
                config,
                "lora_rank",
                _cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
        )
        self.lora_alpha = float(
            _cfg_get(
                config,
                "lora_alpha",
                _cfg_get(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
        )
        self.freeze_base = self.lora_rank > 0

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and pretrained_attn is not None:
            p = next(pretrained_attn.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        # ---- Projections ----
        self.q_proj = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.k_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )

        # Aliases for cross-architecture compatibility
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj
        self.q_lin = self.q_proj
        self.k_lin = self.k_proj
        self.v_lin = self.v_proj
        self.out_lin = self.out_proj
        self.dense = self.out_proj
        self.o_proj = self.out_proj
        self.c_proj = self.out_proj

        # ---- FIX-1 + FIX-4: Static routing (cache-stable, GQA-mimicking init) ----
        # Shape: (num_heads, num_kv_heads) — learnable per-head mixing weights
        self.router_static = nn.Parameter(
            torch.zeros(self.num_heads, self.num_kv_heads)
        )
        self._init_router_gqa()

        # ---- Dynamic routing for encoder-only (optional) ----
        # Shape: (num_kv_heads, head_dim) — computes per-token routing from Q
        self.router_dynamic = nn.Linear(self.head_dim, self.num_kv_heads, bias=False)
        nn.init.normal_(self.router_dynamic.weight, std=0.02)

        dropout_p = _cfg_get(
            config, "dropout", _cfg_get(config, "attention_probs_dropout_prob", 0.1)
        )
        self.dropout = nn.Dropout(dropout_p)

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn")
        if source_attn is not None:
            self.copy_weights(source_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    # ------------------------------------------------------------------
    def _init_router_gqa(self) -> None:
        """FIX-4: Initialize routing to mimic standard GQA at step 0.

        Q head i attends mainly to KV head i // kv_repeat.
        With logit +2 for matching head and -2 for others,
        softmax gives ~0.88 weight to the matching KV head.
        """
        with torch.no_grad():
            self.router_static.fill_(-2.0)
            for i in range(self.num_heads):
                kv_idx = min(i // self.kv_repeat, self.num_kv_heads - 1)
                self.router_static.data[i, kv_idx] = 2.0

    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")

        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or cls_name in (
                "BertSelfAttention", "RobertaSelfAttention",
                "XLMRobertaSelfAttention", "ElectraSelfAttention",
            )
            or (
                cls_name.endswith("SelfAttention")
                and "distil" not in cls_name.lower()
                and not has_out_lin
            )
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None),
            "v": (None, None), "o": (None, None),
        }

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or hasattr(c_attn_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size : 2 * split_size, :],
                                   c_b[split_size : 2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size :, :],
                                   c_b[2 * split_size :] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or hasattr(c_proj_mod, "nf")
                        or hasattr(base_o, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # BERT/RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # Flat (DistilBERT, LLaMA)
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        missing = [
            k for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"QCQAAttention.copy_weights: missing mandatory projections "
                f"{missing_after} from {src.__class__.__name__}."
            )

        target_map = {"q": self.q_proj, "k": self.k_proj, "v": self.v_proj, "o": self.out_proj}

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                        continue
                    else:
                        raise RuntimeError(f"QCQAAttention: missing weight for {proj_key}.")

                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif (
                    w.shape[0] < target_mod.weight.shape[0]
                    and target_mod.weight.shape[0] % w.shape[0] == 0
                    and w.shape[1] == target_mod.weight.shape[1]
                ):
                    repeats = target_mod.weight.shape[0] // w.shape[0]
                    target_mod.weight.copy_(w.repeat(repeats, 1))
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    target_mod.weight.zero_()
                    target_mod.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])

                if b is not None and target_mod.bias is not None:
                    if b.shape == target_mod.bias.shape:
                        target_mod.bias.copy_(b)
                    elif (
                        b.shape[0] < target_mod.bias.shape[0]
                        and target_mod.bias.shape[0] % b.shape[0] == 0
                    ):
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

        if self.freeze_base:
            for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
                mod.freeze_base_weights()

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int,
        N_kv: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return (B, N_kv) bool mask (True = keep)."""
        if attention_mask is None:
            return None

        def _to_bool(t: torch.Tensor) -> torch.Tensor:
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if attention_mask.dim() == 2:
            return _to_bool(attention_mask)
        if attention_mask.dim() == 3:
            v = _to_bool(attention_mask)
            if v.shape[1] > 1:
                v = v.any(dim=1)
            else:
                v = v.squeeze(1)
            return v
        if attention_mask.dim() == 4:
            v = _to_bool(attention_mask)
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    _read_past_kv = staticmethod(_read_past_kv)

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
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:

        B, L, _ = hidden_states.shape
        device = hidden_states.device
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        # ---- Projections ----
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k_new = self.k_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, self.head_dim)
        v_new = self.v_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, self.head_dim)

        # ---- RoPE ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len_rope = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    pk = past_key_value[0]
                    past_len_rope = pk.shape[-2] if pk.dim() == 4 else pk.shape[1]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                    pk = past_key_value.key_cache[idx]
                    past_len_rope = pk.shape[-2] if pk.dim() == 4 else pk.shape[1]

            cos_slice = cos[..., past_len_rope : past_len_rope + N_kv_new, :]
            sin_slice = sin[..., past_len_rope : past_len_rope + N_kv_new, :]

            q_t = q.permute(0, 2, 1, 3).contiguous()
            k_t = k_new.permute(0, 2, 1, 3).contiguous()

            cos_q = cos_slice[..., :L, :]
            sin_q = sin_slice[..., :L, :]
            cos_k = cos_slice[..., :N_kv_new, :]
            sin_k = sin_slice[..., :N_kv_new, :]

            q_rot, _ = apply_rotary_pos_emb(q_t, q_t, cos_q, sin_q)
            _, k_rot = apply_rotary_pos_emb(k_t, k_t, cos_k, sin_k)
            q = q_rot.permute(0, 2, 1, 3).contiguous()
            k_new = k_rot.permute(0, 2, 1, 3).contiguous()

        # ---- KV cache ----
        past_k, past_v, _ = self._read_past_kv(past_key_value, layer_idx) if not is_cross else (None, None, 0)

        if past_k is not None and past_v is not None and not is_cross:
            K_cat = torch.cat([past_k.to(k_new.dtype), k_new], dim=1)
            V_cat = torch.cat([past_v.to(v_new.dtype), v_new], dim=1)
        else:
            K_cat = k_new
            V_cat = v_new

        # ---- FIX-2: Return proper present_key_value ----
        present_kv: Any | None = None
        if not is_cross:
            use_cache = kwargs.get("use_cache", False) or past_key_value is not None
            if use_cache:
                if hasattr(past_key_value, "update"):
                    try:
                        past_key_value.update(k_new, v_new, layer_idx)
                        present_kv = past_key_value
                    except Exception:
                        present_kv = (K_cat, V_cat)
                else:
                    present_kv = (K_cat, V_cat)

        N_kv = K_cat.shape[1]
        past_len = N_kv - N_kv_new if (self.is_decoder and not is_cross) else 0

        # ============================================================
        # FIX-1: Routing
        #   Decoder → static (cache-stable)
        #   Encoder → dynamic (input-dependent, averaged)
        # ============================================================
        if self.is_decoder and not is_cross:
            # Static routing (H_q, H_kv) — independent of input, stable across time
            w_head = F.softmax(self.router_static.float(), dim=-1).to(q.dtype)  # (H_q, H_kv)
            w_head = w_head.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, H_q, H_kv)
        else:
            # Encoder: dynamic per-token routing, averaged over sequence
            route_logits = self.router_dynamic(q) / max(self.router_temp, 1e-4)
            route_weights = F.softmax(route_logits.float(), dim=-1).to(q.dtype)
            w_head = route_weights.mean(dim=1)  # (B, H_q, H_kv)
            w_head = w_head / w_head.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Mix KV heads on-the-fly (raw K_cat/V_cat stay in cache)
        K_perm = K_cat.permute(0, 2, 1, 3)  # (B, H_kv, N_kv, d)
        V_perm = V_cat.permute(0, 2, 1, 3)
        K_mixed = torch.einsum("bqk,bkjd->bqjd", w_head, K_perm)  # (B, H_q, N_kv, d)
        V_mixed = torch.einsum("bqk,bkjd->bqjd", w_head, V_perm)

        q_heads = q.permute(0, 2, 1, 3).contiguous()  # (B, H_q, L, d)
        scale = 1.0 / math.sqrt(self.head_dim)

        # ---- Causal mask ----
        causal_mask = None
        if self.is_decoder and not is_cross:
            if past_len > 0:
                abs_q = (past_len + torch.arange(L, device=device)).unsqueeze(1)
                abs_k = torch.arange(N_kv, device=device).unsqueeze(0)
                causal = (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)  # (1, 1, L, N_kv)
            else:
                causal = (
                    torch.tril(torch.ones(L, N_kv, device=device, dtype=torch.bool))
                    .unsqueeze(0).unsqueeze(0)
                )
            causal_mask = causal

        # ---- Padding mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)  # (B, N_kv) or None

        # ============================================================
        # FIX-3: SDPA with bool mask (FlashAttention-compatible)
        # ============================================================
        use_sdpa = (not output_attentions) and (head_mask is None)
        context: torch.Tensor | None = None
        attn_probs_out: torch.Tensor | None = None

        if use_sdpa:
            try:
                # Build a single bool mask (or None)
                bool_mask = None
                if causal_mask is not None and token_valid is not None:
                    bool_mask = (causal_mask & token_valid[:, None, None, :]).expand(B, 1, L, N_kv)
                elif causal_mask is not None:
                    bool_mask = causal_mask  # (1, 1, L, N_kv)
                elif token_valid is not None:
                    bool_mask = token_valid[:, None, None, :].expand(B, 1, L, N_kv)

                q_sdp = maybe_half(q_heads)
                k_sdp = maybe_half(K_mixed)
                v_sdp = maybe_half(V_mixed)

                out_sdp = F.scaled_dot_product_attention(
                    q_sdp, k_sdp, v_sdp,
                    attn_mask=bool_mask,
                    is_causal=False,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                context = (
                    out_sdp.to(q.dtype)
                    .permute(0, 2, 1, 3).contiguous()
                    .view(B, L, self.all_head_size)
                )
                context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)
            except Exception as e:
                logger.debug("QCQA SDPA failed: %s; falling back to manual.", e)
                context = None

        # ---- Manual path (fallback or when output_attentions/head_mask) ----
        if context is None:
            scores = torch.matmul(
                q_heads.float(), K_mixed.float().transpose(-2, -1).contiguous()
            ) * scale

            if causal_mask is not None:
                scores = scores.masked_fill(~causal_mask, -1e4)
            if token_valid is not None:
                scores = scores.masked_fill(~token_valid[:, None, None, :], -1e4)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn_probs = F.softmax(scores, dim=-1).to(q.dtype)
            attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
            attn_probs = self.dropout(attn_probs)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask[:, :, :1].unsqueeze(-1)
                attn_probs = attn_probs * head_mask.to(attn_probs.dtype)

            context = torch.matmul(attn_probs, V_mixed)
            context = context.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
            context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)
            attn_probs_out = attn_probs if output_attentions else None

        # ---- Output projection ----
        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- FIX-2: Return in the correct format ----
        if self.is_decoder and not is_cross:
            if output_attentions:
                return out, present_kv, attn_probs_out
            return out, present_kv
        if output_attentions:
            return out, attn_probs_out
        return out, None


# ===========================================================================
# Injector (FIX-5: registered in ATTENTION_REGISTRY)
# ===========================================================================
def inject_qcqa(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    target_layer_idx: int | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_qcqa: max recursion depth exceeded, aborting.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
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
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_qcqa(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # ---- Single layer ----
    try:
        def _already_injected(m):
            if m is None:
                return False
            if isinstance(m, QCQAAttention):
                return True
            if "QCQA" in m.__class__.__name__:
                return True
            inner = getattr(m, "inner", None)
            if isinstance(inner, QCQAAttention):
                return True
            wrapped = getattr(m, "_wrapped_attention", None)
            if isinstance(wrapped, QCQAAttention):
                return True
            return False

        if _already_injected(old_attn):
            return None

        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            return None
        if _already_injected(src):
            return None

        new_attn = QCQAAttention(
            config=cfg,
            comp_config=comp_config,
            pretrained_attn=src,
            **kwargs,
        )

        # Place on correct device/dtype
        try:
            ref = next(src.parameters(), None)
            if ref is not None:
                new_attn.to(device=ref.device, dtype=ref.dtype)
        except Exception:
            pass

        # Splice into parent
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
            spliced = True
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            spliced = True
        elif hasattr(target, "attn"):
            target.attn = new_attn
            spliced = True
        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
            spliced = True

        if not spliced:
            logger.error("inject_qcqa: no writable attention attribute on %s", type(target).__name__)
            return None

        return new_attn
    except Exception as e:
        logger.exception("inject_qcqa failed: %s", e)
        return None


inject_qcqa_attention = inject_qcqa


__all__ = [
    "QCQAAttention",
    "LoRALinear",
    "inject_qcqa",
    "inject_qcqa_attention",
]