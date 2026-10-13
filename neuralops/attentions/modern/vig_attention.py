"""
ViG Attention: Vision GNN-inspired 2D Grid Linear Attention with Gated Fusion.

"""

from __future__ import annotations
from neuralops.core.common import _read_past_kv
from neuralops.core.common import _cfg_get
from neuralops.core.common import _detect_decoder

from neuralops.core.attention_utils import (
    safe_mask_value as _safe_mask_value,
    apply_rotary_pos_emb,
    LoRALinear,
)

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half
from neuralops.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)








# ═══════════════════════════════════════════════════════════════════
# ViG Attention
# ═══════════════════════════════════════════════════════════════════


@ATTENTION_REGISTRY.register(
    "vig_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class ViGAttention(BaseAttention):
    r"""Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion.

    Two parallel branches:
      1. Graph branch: local window + first-token anchor (sparse attention).
      2. Linear branch: softplus kernel attention (O(N) memory), L2-normalized.

    Gated fusion combines both per head, per token.
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: int | None = None,
        pretrained_attn: nn.Module | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        local_window: int = 16,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        src_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn")
        self.config = config

        # ---- Dimensions ----
        if config is not None and not isinstance(config, int):
            self.adapter = ModelAdapter(config)
            self.num_heads: int = self.adapter.num_heads
            self.num_kv_heads: int = self.adapter.num_key_value_heads
            self.dim: int = self.adapter.hidden_size
            self.head_dim: int = self.adapter.head_dim
            self.all_head_size: int = self.num_heads * self.head_dim
            self.kv_head_size: int = self.num_kv_heads * self.head_dim
            self.kv_repeat: int = self.adapter.kv_repeat
        else:
            self.adapter = None
            self.dim = _cfg_get(
                config, "hidden_size",
                kwargs.get("hidden_size", kwargs.get("d_model", 768)),
            )
            self.num_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_heads", 12))
            self.num_kv_heads = _cfg_get(config, "num_key_value_heads", self.num_heads)
            self.head_dim = self.dim // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        if "hidden_size" in kwargs:
            self.dim = kwargs["hidden_size"]
            self.head_dim = self.dim // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
        if "num_heads" in kwargs:
            self.num_heads = kwargs["num_heads"]
            self.head_dim = self.dim // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # ---- Decoder detection ----
        self.is_decoder = _detect_decoder(config, is_decoder, src_attn)
        self.is_nested_encoder: bool = False

        self.lora_rank = int(kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank)))
        self.lora_alpha = float(kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha)))

        # ---- Local window ----
        self.local_window: int = int(local_window)

        # ---- Projections ----
        target_dtype = kwargs.get("dtype")
        if target_dtype is None and src_attn is not None:
            p = next(src_attn.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        self.q_proj = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.k_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )

        # ---- [FIX-10] gate_net stays fp32 for stability ----
        self.gate_net = nn.Linear(self.dim, 2 * self.num_heads).float()
        # ---- [FIX-12] zero-init bias so softmax gives 0.5/0.5 at start ----
        nn.init.zeros_(self.gate_net.bias)
        nn.init.normal_(self.gate_net.weight, std=0.02)

        dropout_p = _cfg_get(
            config, "dropout",
            _cfg_get(config, "hidden_dropout_prob",
                     _cfg_get(config, "attention_probs_dropout_prob", 0.1)),
        )
        self.dropout = nn.Dropout(float(dropout_p))
        self.rotary_emb = getattr(src_attn, "rotary_emb", None)

        # ---- [FIX-15] cache for graph_adj ----
        self._graph_adj_cache: dict = {}

        if src_attn is not None:
            self.copy_weights(src_attn)

        # ---- [FIX-10] only cast projections, NOT the whole module ----
        if target_dtype is not None:
            self.q_proj.to(target_dtype)
            self.k_proj.to(target_dtype)
            self.v_proj.to(target_dtype)
            self.out_proj.to(target_dtype)

    # ---- Aliases (state-dict compatibility) ----
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None),
            "v": (None, None), "o": (None, None),
        }

        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed for %s: %s", proj, e)

        # GPT-2 fused c_attn
        if hasattr(old_attn, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            weight = old_attn.c_attn.weight.data
            bias = getattr(old_attn.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if (
                "Conv1D" in old_attn.c_attn.__class__.__name__
                or hasattr(old_attn.c_attn, "nf")
                or (weight.dim() == 2 and (
                    weight.shape[0] < weight.shape[1]
                    or weight.shape[1] == 3 * self.dim
                ))
            ):
                weight = weight.t().contiguous()
            split_size = weight.shape[0] // 3
            extracted["q"] = (weight[:split_size, :],
                              bias_data[:split_size] if bias_data is not None else None)
            extracted["k"] = (weight[split_size:2 * split_size, :],
                              bias_data[split_size:2 * split_size] if bias_data is not None else None)
            extracted["v"] = (weight[2 * split_size:, :],
                              bias_data[2 * split_size:] if bias_data is not None else None)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            w_o = old_attn.c_proj.weight.data
            if (
                "Conv1D" in old_attn.c_proj.__class__.__name__
                or hasattr(old_attn.c_proj, "nf")
                or (w_o.dim() == 2 and (
                    w_o.shape[0] < w_o.shape[1]
                    or (w_o.shape[0] != self.dim and w_o.shape[1] == self.dim)
                ))
            ):
                w_o = w_o.t().contiguous()
            b_o = getattr(old_attn.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # BERT/RoBERTa nested
        if hasattr(old_attn, "self") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            s = old_attn.self
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if (
            hasattr(old_attn, "output")
            and hasattr(old_attn.output, "dense")
            and extracted["o"][0] is None
        ):
            dense = old_attn.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # Flat (DistilBERT, LLaMA, Qwen)
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "c_proj"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(old_attn, name):
                        layer = getattr(old_attn, name)
                        base = getattr(layer, "base_layer", layer)
                        if hasattr(base, "weight"):
                            extracted[proj] = (base.weight.data, getattr(base, "bias", None))
                            break

        cls_name = old_attn.__class__.__name__
        has_out_proj = any(
            hasattr(old_attn, attr)
            for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "out")
        )
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("ViGAttention.copy_weights: missing %s; keeping random init.", missing)

        # GQA head pooling
        def _pool_gqa_heads(w: torch.Tensor, target_heads: int, head_dim: int) -> torch.Tensor:
            if w.dim() != 2:
                return w
            in_feat = w.shape[1]
            src_heads = w.shape[0] // head_dim
            if src_heads == target_heads:
                return w
            w_reshaped = w.view(src_heads, head_dim, in_feat)
            if src_heads > target_heads and src_heads % target_heads == 0:
                group = src_heads // target_heads
                return w_reshaped.view(target_heads, group, head_dim, in_feat).mean(dim=1).reshape(
                    target_heads * head_dim, in_feat)
            if src_heads > target_heads:
                idx = torch.linspace(0, src_heads - 1, target_heads).long()
                return w_reshaped.index_select(0, idx).reshape(target_heads * head_dim, in_feat)
            rep = target_heads // src_heads
            return w_reshaped.repeat_interleave(rep, dim=0).reshape(target_heads * head_dim, in_feat)

        def _orient(w: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
            if w.dim() != 2:
                return w
            if w.shape == target_shape:
                return w
            if w.shape == (target_shape[1], target_shape[0]):
                return w.t().contiguous()
            return w

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
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if key in ("k", "v"):
                    w = _pool_gqa_heads(w, self.num_kv_heads, self.head_dim)
                    if b is not None:
                        b = _pool_gqa_heads(b.unsqueeze(-1), self.num_kv_heads, self.head_dim).squeeze(-1)

                w = _orient(w, target.weight.shape)
                out_d = min(target.weight.shape[0], w.shape[0])
                in_d = min(target.weight.shape[1], w.shape[1])
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

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int,
        N_kv: int,
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
            return _to_bool(attention_mask)
        if attention_mask.dim() == 3:
            v = _to_bool(attention_mask)
            if v.shape[1] > 1:
                v = v.any(dim=1)
            else:
                v = v.squeeze(1)
            return v
        if attention_mask.dim() == 4:
            return _to_bool(attention_mask).any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    _read_past_kv = staticmethod(_read_past_kv)

    def _build_graph_adj(
        self,
        L: int,
        N_kv: int,
        past_len: int,
        device: torch.device,
        is_decoder: bool,
    ) -> torch.Tensor:
        """Build (L, N_kv) boolean graph adjacency.

        In decoder mode:
          - STRICTLY causal: k_abs <= q_abs (never attend to future).
          - Local window: |k_abs - q_abs| <= local_window.
          - First-token anchor: only when k_abs == 0.

        In encoder mode:
          - Local window + first-token + last-token anchors.
          - Bidirectional.
        """
        # ---- [FIX-15] cache key ----
        cache_key = (L, N_kv, past_len, is_decoder, str(device))
        if cache_key in self._graph_adj_cache:
            return self._graph_adj_cache[cache_key]

        q_abs = torch.arange(L, device=device) + past_len   # (L,)
        k_abs = torch.arange(N_kv, device=device)           # (N_kv,)
        diff = k_abs.unsqueeze(0) - q_abs.unsqueeze(1)      # (L, N_kv): k - q

        # Local window
        local = diff.abs() <= self.local_window

        # First-token anchor (k=0)
        anchor_first = (k_abs == 0).unsqueeze(0).expand(L, -1)

        if is_decoder:
            # ---- [FIX-9] STRICT CAUSALITY: k_abs <= q_abs ----
            causal = diff <= 0
            graph_adj = (local | anchor_first) & causal
        else:
            # Encoder: bidirectional local + first + last
            anchor_last = (k_abs == N_kv - 1).unsqueeze(0).expand(L, -1)
            graph_adj = local | anchor_first | anchor_last

        # ---- [FIX-15] cache (limit size) ----
        if len(self._graph_adj_cache) > 8:
            self._graph_adj_cache.clear()
        self._graph_adj_cache[cache_key] = graph_adj

        return graph_adj

    # ------------------------------------------------------------------
    # [FIX-3 + FIX-13 + FIX-17] Stabilized causal linear attention
    # ------------------------------------------------------------------
    @staticmethod
    def _linear_attention_decoder_cumsum(
        q_lin: torch.Tensor,   # (B, H, L, Dh)
        k_lin: torch.Tensor,   # (B, H, N_kv, Dh)
        v_f: torch.Tensor,     # (B, H, N_kv, Dh)
        past_len: int,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """Fully vectorized causal cumsum without memory allocation loops."""
        L = q_lin.shape[2]

        # ضرب خارجی و جمع تجمعی بدون حلقه پایتون و بدون ساخت متغیرهای ۵-بعدی صفر
        kv_outer = k_lin.unsqueeze(-1) * v_f.unsqueeze(-2)  # (B, H, N_kv, Dh, Dh)
        kv_cumsum = kv_outer.cumsum(dim=2)
        k_cumsum = k_lin.cumsum(dim=2)

        # اسلایس مستقیم برای طول کوئری فعلی
        kv_at_q = kv_cumsum[:, :, past_len : past_len + L, :, :]
        k_at_q = k_cumsum[:, :, past_len : past_len + L, :]

        num = torch.einsum("bhnf,bhnfd->bhnd", q_lin, kv_at_q)
        denom = torch.einsum("bhnf,bhnf->bhn", q_lin, k_at_q).unsqueeze(-1).clamp(min=1e-4)

        return num / denom

    @staticmethod
    def _linear_attention_encoder(
        q_lin: torch.Tensor,
        k_lin: torch.Tensor,
        v_f: torch.Tensor,
    ) -> torch.Tensor:
        """Bidirectional linear attention via associativity: Q(K^T V)."""
        kv = torch.matmul(k_lin.transpose(-1, -2).contiguous(), v_f)   # (B, H, Dh, Dh)
        num = torch.matmul(q_lin, kv)                                   # (B, H, L, Dh)
        k_sum = k_lin.sum(dim=-2, keepdim=True)                         # (B, H, 1, Dh)
        # ---- [FIX-17] safer clamp ----
        denom = torch.matmul(q_lin, k_sum.transpose(-1, -2).contiguous()).clamp(min=1e-4)
        return num / denom

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
    ) -> tuple[Any, ...]:
        B, N, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        Hh = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim
        layer_idx = int(kwargs.get("layer_idx", 0))

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        # ---- Projections ----
        q = self.q_proj(hidden_states).view(B, N, Hh, Dh).transpose(1, 2).contiguous()
        k_new = self.k_proj(kv_input).view(B, N_kv_new, H_kv, Dh).transpose(1, 2).contiguous()
        v_new = self.v_proj(kv_input).view(B, N_kv_new, H_kv, Dh).transpose(1, 2).contiguous()

        # ---- Read past KV ----
        past_k, past_v = self._read_past_kv(past_key_value, layer_idx) if not is_cross else (None, None)
        past_len = past_k.shape[-2] if past_k is not None else 0

        # ---- RoPE ----
        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)
        elif getattr(self, "rotary_emb", None) is not None and not is_cross:
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                position_ids = torch.arange(
                    past_len, past_len + N_kv_new, device=device
                ).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = self.rotary_emb(k_new, position_ids=position_ids)
                q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)
            except Exception as e:
                logger.debug("rotary_emb failed: %s", e)

        # ---- Concat KV cache ----
        present_kv: Any | None = None
        if is_cross:
            K_cat, V_cat = k_new, v_new
            present_kv = past_key_value
        else:
            if past_k is not None and past_v is not None:
                K_cat = torch.cat([past_k.to(k_new.dtype), k_new], dim=-2)
                V_cat = torch.cat([past_v.to(v_new.dtype), v_new], dim=-2)
            else:
                K_cat, V_cat = k_new, v_new

            use_cache = kwargs.get("use_cache", False) or past_key_value is not None
            if use_cache:
                if past_key_value is not None and hasattr(past_key_value, "update"):
                    try:
                        past_key_value.update(k_new, v_new, layer_idx)
                        present_kv = past_key_value
                    except Exception:
                        present_kv = (K_cat, V_cat)
                else:
                    present_kv = (K_cat, V_cat)

        N_kv = K_cat.shape[-2]

        # ---- GQA expand ----
        if self.kv_repeat > 1:
            K_full = repeat_kv(K_cat, self.kv_repeat)
            V_full = repeat_kv(V_cat, self.kv_repeat)
        else:
            K_full, V_full = K_cat, V_cat

        # ---- Graph adjacency ----
        graph_adj = self._build_graph_adj(N, N_kv, past_len, device, self.is_decoder)

        # ---- Padding mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)

        # ---- [FIX-11] safe mask value ----
        mask_val = _safe_mask_value(dtype)

        # ---- Fast Graph Branch (Vectorized Float Masking) ----
        scale = 1.0 / math.sqrt(Dh)
        scores_graph = torch.matmul(q.float(), K_full.float().transpose(-2, -1)) * scale

        # ترکیب ماسک همسایگی گراف و ماسک پدینگ توکن‌ها
        combined_mask = graph_adj.unsqueeze(0).unsqueeze(1)
        if token_valid is not None:
            combined_mask = combined_mask & token_valid[:, None, None, :]

        scores_graph = scores_graph.masked_fill(~combined_mask, -1e4)
        attn_graph = F.softmax(scores_graph, dim=-1).to(dtype)
        attn_graph_out = torch.matmul(attn_graph, V_full)

        # ═══════════════════════════════════════════════════════════════
        # [FIX-17] STABILIZED LINEAR BRANCH
        # ═══════════════════════════════════════════════════════════════
        # 1. L2-normalize Q and K before activation to prevent explosion.
        # 2. Use softplus (bounded, strictly positive) instead of elu+1.
        # 3. Small epsilon ensures no absolute zeros.
        q_norm = F.normalize(q.float(), p=2, dim=-1)
        k_norm = F.normalize(K_full.float(), p=2, dim=-1)

        q_lin = F.softplus(q_norm) + 1e-6
        k_lin = F.softplus(k_norm) + 1e-6

        if token_valid is not None:
            vmask = token_valid[:, None, :, None].to(k_lin.dtype)
            k_lin = k_lin * vmask
            v_f = V_full.float() * vmask
        else:
            v_f = V_full.float()

        if self.is_decoder and not is_cross:
            out_global = self._linear_attention_decoder_cumsum(q_lin, k_lin, v_f, past_len).to(dtype)
        else:
            out_global = self._linear_attention_encoder(q_lin, k_lin, v_f).to(dtype)

        # ---- [FIX-10 + FIX-12] Gated fusion in fp32 ----
        gates = F.softmax(
            self.gate_net(hidden_states.float()).view(B, N, Hh, 2), dim=-1
        ).to(dtype)
        gate_graph = gates[..., 0].transpose(1, 2).contiguous().unsqueeze(-1)    # (B, H, N, 1)
        gate_global = gates[..., 1].transpose(1, 2).contiguous().unsqueeze(-1)   # (B, H, N, 1)

        out = gate_graph * attn_graph_out + gate_global * out_global
        out = self.dropout(out)

        # ---- Head mask ----
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            out = out * head_mask.to(dtype)

        out = out.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Return format ----
        if self.is_decoder and not is_cross:
            if output_attentions:
                return output.to(dtype), present_kv, None
            return output.to(dtype), present_kv

        if output_attentions:
            return output.to(dtype), None
        return output.to(dtype), None


# ═══════════════════════════════════════════════════════════════════
# Injector
# ═══════════════════════════════════════════════════════════════════


def inject_vig(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_vig: max recursion depth exceeded.")
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
            res = inject_vig(
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
            if isinstance(m, ViGAttention):
                return True
            if "ViGAttention" in m.__class__.__name__:
                return True
            inner = getattr(m, "inner", None)
            if isinstance(inner, ViGAttention):
                return True
            wrapped = getattr(m, "_wrapped_attention", None)
            if isinstance(wrapped, ViGAttention):
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

        if model_config is None:
            model_config = getattr(target, "config", None) or getattr(src, "config", None)

        if model_config is None:
            import types
            hidden_size = (
                getattr(src, "hidden_size", None)
                or getattr(src, "embed_dim", None)
                or (src.q_proj.in_features if hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features") else None)
                or (src.q_lin.in_features if hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features") else None)
                or (src.query.in_features if hasattr(src, "query") and hasattr(src.query, "in_features") else None)
                or 768
            )
            num_heads = (
                getattr(src, "num_heads", None)
                or getattr(src, "num_attention_heads", None)
                or 12
            )
            model_config = types.SimpleNamespace(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=getattr(src, "num_key_value_heads", num_heads),
                is_decoder=getattr(src, "is_decoder", False),
            )

        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
        rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

        new_attn = ViGAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=src,
            **kwargs,
        )

        try:
            ref = next(src.parameters(), None)
            if ref is not None:
                new_attn.to(device=ref.device, dtype=ref.dtype)
        except Exception:
            pass

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
            logger.error("inject_vig: no writable attention attribute on %s", type(target).__name__)
            return None

        return new_attn
    except Exception as e:
        logger.exception("inject_vig failed at layer %d: %s", layer_idx, e)
        return None


__all__ = [
    "ViGAttention",
    "LoRALinear",
    "inject_vig",
    "apply_rotary_pos_emb",
]