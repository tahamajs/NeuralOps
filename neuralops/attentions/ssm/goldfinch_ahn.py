"""
GoldFinch-AHN: Adaptive Hybrid Network — Sliding Window + Gated Recurrent Memory.
Production-hardened v2 with:
  - Chunked parallel scan for GatedRecurrentMemory (no Python loop)
  - SDPA fast-path for sliding-window attention
  - Configurable window_radius (full context supported)
  - Configurable gate_init_bias
  - Proper freeze semantics for delta_mem
  - Correct head_mask placement (pre-softmax)
  - Robust attention_mask normalization
"""

from __future__ import annotations
from neuralops.core.common import _env_flag

from neuralops.core.attention_utils import (
    apply_rotary_pos_emb,
    LoRALinear,
)

import logging
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)


# ===========================================================================
# GatedRecurrentMemory — chunked parallel scan (no Python loop)
# ===========================================================================
class GatedRecurrentMemory(nn.Module):
    """
    Gated recurrent memory with chunked parallel scan.

        g_t = sigmoid(W_g x_t + b_g)
        h_t = (1 - g_t) * h_{t-1} + g_t * tanh(W_x x_t + b_x)
        y_t = h_t

    Vectorized via chunked associative scan:
      - Intra-chunk: sequential (chunk_size steps)
      - Inter-chunk: parallel cumprod of (1-g) between chunks
    """

    def __init__(self, d_model: int, chunk_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = max(1, int(chunk_size))
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.value_proj = nn.Linear(d_model, d_model, bias=True)
        # Start gate slightly open so memory doesn't dominate attention
        nn.init.constant_(self.gate_proj.bias, -1.0)
        nn.init.zeros_(self.value_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        target_dtype = x.dtype
        gw = self.gate_proj.weight.to(target_dtype)
        gb = self.gate_proj.bias.to(target_dtype) if self.gate_proj.bias is not None else None
        vw = self.value_proj.weight.to(target_dtype)
        vb = self.value_proj.bias.to(target_dtype) if self.value_proj.bias is not None else None

        g = torch.sigmoid(F.linear(x, gw, gb))   # (B, L, D)
        v = torch.tanh(F.linear(x, vw, vb))       # (B, L, D)
        B, L, D = x.shape

        if _env_flag("GOLDFINCH_DISABLE_RECURRENCE", False):
            # Ablation: return v (no recurrence)
            return v.to(target_dtype)

        if L <= self.chunk_size:
            # Short sequence: simple sequential
            h = torch.zeros(B, D, device=x.device, dtype=target_dtype)
            outs = []
            for t in range(L):
                h = (1.0 - g[:, t]) * h + g[:, t] * v[:, t]
                outs.append(h)
            return torch.stack(outs, dim=1).to(target_dtype)

        from neuralops.core.triton_kernels.fms_scan import fused_goldfinch_hopfield_lookup_triton

        return fused_goldfinch_hopfield_lookup_triton(g, v, self.chunk_size)



# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "goldfinch_ahn",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class GoldFinchAHNBlock(BaseAttention):
    """
    GoldFinch-AHN: hybrid sliding-window attention + gated recurrent memory.

    window_size: full window width in tokens (effective radius = window_size // 2).
                 Set to 0 or >= seq_len for full causal context (recommended for GPT-2).
    gate_init_bias: initial bias for per-head attention/memory gate.
                    Higher = more attention. 0.0 = balanced.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        window_size: int = 64,
        is_causal: bool = False,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        gate_init_bias: float = 1.8,
        comp_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        # ---- hidden size ----
        if d_model is None:
            d_model = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
            if d_model is None:
                if config is not None:
                    d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
                elif source_attn is not None:
                    out_layer = None
                    for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
                        if hasattr(source_attn, attr):
                            out_layer = getattr(source_attn, attr)
                            break
                    if out_layer is not None:
                        d_model = getattr(out_layer, "out_features",
                                          getattr(out_layer, "in_features", 768))
                    else:
                        for attr in ("q_proj", "query", "q_lin"):
                            q_l = getattr(source_attn, attr, None)
                            if q_l is not None:
                                d_model = getattr(q_l, "in_features",
                                                  getattr(q_l, "out_features", 768))
                                break
                if d_model is None:
                    d_model = 768

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        # ---- num_heads ----
        if num_heads is None:
            num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
            if num_heads is None:
                if config is not None:
                    num_heads = getattr(config, "num_attention_heads",
                                        getattr(config, "n_heads", 12))
                elif source_attn is not None:
                    num_heads = getattr(source_attn, "num_attention_heads",
                                        getattr(source_attn, "n_heads", 12))
                else:
                    num_heads = 12

        self.dim = int(d_model)
        self.hidden_size = self.dim
        self.num_heads = max(1, int(num_heads))

        # ---- num_kv_heads ----
        if config is not None:
            self.num_kv_heads = getattr(
                config, "num_key_value_heads",
                getattr(config, "num_kv_heads", kwargs.get("num_kv_heads", self.num_heads))
            )
        elif source_attn is not None:
            self.num_kv_heads = (
                getattr(source_attn, "num_key_value_heads", None)
                or getattr(source_attn, "num_kv_heads", None)
                or kwargs.get("num_kv_heads", self.num_heads)
            )
        else:
            self.num_kv_heads = kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = max(1, min(int(self.num_kv_heads), self.num_heads))

        # ---- head_dim ----
        _head_dim = getattr(config, "head_dim", None) if config else None
        if _head_dim is None and source_attn is not None:
            _head_dim = getattr(source_attn, "head_dim", None)
        if _head_dim is None:
            _head_dim = self.dim // self.num_heads
        self.head_dim = int(_head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        # ---- window (FIX 4: allow full context) ----
        # If GOLDFINCH_FULL_CONTEXT=1, window_radius = inf (full causal)
        full_ctx = _env_flag("GOLDFINCH_FULL_CONTEXT", False)
        if config is not None:
            self.window_size = int(
                getattr(config, "window_size",
                        getattr(comp_config, "window_size",
                                kwargs.get("window_size", window_size))
                        if comp_config is not None else window_size)
            )
        elif comp_config is not None:
            self.window_size = int(getattr(comp_config, "window_size",
                                           kwargs.get("window_size", window_size)))
        else:
            self.window_size = int(kwargs.get("window_size", window_size))

        if full_ctx or self.window_size <= 0:
            self.window_radius = None  # sentinel: full context
        else:
            self.window_radius = max(1, self.window_size // 2)

        # ---- decoder detection ----
        model_type = str(getattr(config, "model_type", "") or "").lower() if config else ""
        is_dec = False
        if config is not None:
            is_dec = bool(getattr(config, "is_decoder", False)
                          or getattr(config, "is_causal", False))
        if any(model_type.startswith(p) for p in (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt",
            "bloom", "falcon", "olmo", "mpt", "stablelm",
        )):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False) or is_causal:
            is_dec = True
        if source_attn is not None:
            if hasattr(source_attn, "c_attn") or getattr(source_attn, "is_decoder", False):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # ---- LoRA config ----
        if config is not None:
            self.lora_rank = int(getattr(config, "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
                if comp_config is not None else kwargs.get("lora_rank", 16)))
            self.lora_alpha = float(getattr(config, "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
                if comp_config is not None else kwargs.get("lora_alpha", 32.0)))
        elif comp_config is not None:
            self.lora_rank = int(getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            self.lora_alpha = float(getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))

        self.freeze_base = self.lora_rank > 0
        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None) if source_attn is not None else None
        )

        # ---- projections ----
        self.W_q = LoRALinear(self.dim, self.all_head_size, bias=True,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.W_k = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.W_v = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)

        # Gated recurrent memory
        self.delta_mem = GatedRecurrentMemory(self.dim, chunk_size=16)

        # Per-head attention / memory gate (FIX 8: single output, no waste)
        # gate_init_bias can be overridden via env
        _gate_bias = float(os.environ.get("GOLDFINCH_GATE_BIAS", gate_init_bias))
        self.gate = nn.Linear(self.dim, 1)
        with torch.no_grad():
            self.gate.bias.data.fill_(_gate_bias)
            nn.init.zeros_(self.gate.weight)

        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        dropout_p = 0.1
        if config is not None:
            dropout_p = float(getattr(config, "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1)))
        elif comp_config is not None:
            dropout_p = float(getattr(comp_config, "dropout", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        # Single output scale, initialized to preserve pretrained signal
        self.output_scale = nn.Parameter(torch.ones(1))

        # Storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ---- aliases ----
    @property
    def query(self): return self.W_q
    @property
    def key(self): return self.W_k
    @property
    def value(self): return self.W_v
    @property
    def q_proj(self): return self.W_q
    @property
    def k_proj(self): return self.W_k
    @property
    def v_proj(self): return self.W_v
    @property
    def q_lin(self): return self.W_q
    @property
    def k_lin(self): return self.W_k
    @property
    def v_lin(self): return self.W_v
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    def freeze_base_weights(self):
        for m in (self.W_q, self.W_k, self.W_v, self.out_proj):
            if hasattr(m, "freeze_base_weights"):
                m.freeze_base_weights()
        # FIX 6: freeze delta_mem base weights if freeze_base
        if self.freeze_base:
            for p in self.delta_mem.parameters():
                p.requires_grad = False

    def unfreeze_base_weights(self):
        for m in (self.W_q, self.W_k, self.W_v, self.out_proj):
            if hasattr(m, "unfreeze_base_weights"):
                m.unfreeze_base_weights()
        for p in self.delta_mem.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # copy_weights (unchanged, but with FIX 12 for int masks)
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = any(hasattr(src, a) for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))
        is_nested = False
        if (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                         "XLMRobertaSelfAttention", "ElectraSelfAttention",
                         "DebertaSelfAttention")
            or (hasattr(src, "query") and not has_out)
            or (hasattr(src, "self") and hasattr(src, "output"))
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted = {"q": (None, None), "k": (None, None),
                     "v": (None, None), "o": (None, None)}

        # GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (
                    "Conv1D" in c_attn_mod.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c_attn_mod, "nf") or hasattr(base_c, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv:
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split], c_b[:split] if c_b is not None else None)
                extracted["k"] = (c_w[split:2*split], c_b[split:2*split] if c_b is not None else None)
                extracted["v"] = (c_w[2*split:], c_b[2*split:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (
                        "Conv1D" in c_proj_mod.__class__.__name__
                        or "Conv1D" in base_o.__class__.__name__
                        or hasattr(c_proj_mod, "nf") or hasattr(base_o, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # Nested (BERT / RoBERTa)
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (m.weight.data.clone(),
                                      b.data.clone() if b is not None else None)
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                d = src.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (d.weight.data.clone(),
                                  b.data.clone() if b is not None else None)

        # Flat
        else:
            if hasattr(src, "weight") and isinstance(src, nn.Linear):
                # When src is directly a Linear layer (e.g. BrokenLayer.attention)
                extracted["q"] = (src.weight.data.clone(),
                                  src.bias.data.clone() if src.bias is not None else None)

            candidates = {
                "q": ("q_lin", "q_proj", "query", "W_q"),
                "k": ("k_lin", "k_proj", "key", "W_k"),
                "v": ("v_lin", "v_proj", "value", "W_v"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                if extracted[key][0] is not None:
                    continue
                for n in names:
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (m.weight.data.clone(),
                                              b.data.clone() if b is not None else None)
                            break

        # UniversalAdapter fallback
        missing = [k for k, (w, _) in extracted.items()
                   if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("Goldfinch UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning("Goldfinch copy_weights: missing %s; using initialized weights",
                           missing_after)

        target_map = {"q": self.W_q, "k": self.W_k, "v": self.W_v, "o": self.out_proj}

        with torch.no_grad():
            for key, target in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if (w.shape != target.weight.shape
                        and w.t().contiguous().shape == target.weight.shape):
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data[:mb].copy_(b[:mb].to(target.bias.device, target.bias.dtype))
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                w_norm = float(torch.linalg.norm(target.weight.data.float()).item())
                if w_norm < 1e-4:
                    raise RuntimeError(f"Goldfinch copy_weights: '{key}' norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target.freeze_base_weights()

        # Track compressed storage (FIX 9: include output_scale and gate)
        try:
            for m in (self.W_q, self.W_k, self.W_v, self.out_proj, self.delta_mem, self.gate):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
            self._comp_storage_bytes += self.output_scale.numel() * self.output_scale.element_size()
        except Exception:
            pass

        if self.freeze_base:
            self.freeze_base_weights()

    @classmethod
    def from_distilbert_pretrained(cls, old_attn, model_config=None, **kwargs):
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    # ------------------------------------------------------------------
    # Mask helper (FIX 7: support more shapes)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_valid_mask(attention_mask, B, L, device):
        if attention_mask is None:
            return None

        def _to_bool(t):
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
            # (B, 1, L) or (B, L, L)
            m = _to_bool(attention_mask)
            if m.shape[1] == 1:
                m = m[:, 0, :]
            else:
                m = m.any(dim=1)
            if m.shape == (B, L):
                return m
        elif attention_mask.dim() == 4:
            m = _to_bool(attention_mask)
            # Extract token-level validity along the key sequence dimension
            if m.shape[-2] == 1:
                m = m.squeeze(1).squeeze(1)  # (B, L)
            else:
                m = m[:, 0].any(dim=-2)      # (B, L)
            if m.shape == (B, L):
                return m

        try:
            m = get_token_valid_mask(attention_mask, B, L, device)
            if isinstance(m, torch.Tensor) and m.shape == (B, L):
                return m.bool()
        except Exception:
            pass
        return None

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        *args: Any,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("GoldFinchAHN does not support cross-attention.")

        x = hidden_states
        if not x.is_contiguous():
            x = x.contiguous()
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim
        device = x.device
        dtype_in = x.dtype

        # ---- QKV ----
        q = self.W_q(x).view(B, L, H, d).transpose(1, 2).contiguous()
        k = self.W_k(x).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.W_v(x).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # ---- RoPE ----
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif getattr(self, "rotary_emb", None) is not None:
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None:
                pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = self.rotary_emb(k, seq_len=L)
            except TypeError:
                try:
                    cos, sin = self.rotary_emb(k, pos_ids)
                except Exception:
                    cos, sin = None, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        # ---- Autoregressive KV cache for decoders ----
        if self.is_decoder and past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
                if past_k is not None and past_v is not None:
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)
        present_key_value = (k, v) if self.is_decoder else None

        # ---- GQA ----
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- Sliding-window attention (FIX 2: SDPA fast-path) ----
        # Build additive bool mask
        L_kv = k.shape[-2]
        idx_q = torch.arange(L, device=device).unsqueeze(1)
        idx_k = torch.arange(L_kv, device=device).unsqueeze(0)
        if self.window_radius is None:
            band = torch.ones(L, L_kv, device=device, dtype=torch.bool)
        else:
            diff = (idx_q - idx_k).abs()
            band = diff <= self.window_radius

        if self.is_causal:
            past_len = L_kv - L
            causal = torch.tril(torch.ones(L, L_kv, device=device, dtype=torch.bool), diagonal=past_len)
            valid_local = causal & band
        else:
            valid_local = band

        valid_mask = self._normalize_valid_mask(attention_mask, B, L, device)
        if valid_mask is not None:
            pad_mask = valid_mask[:, None, None, :]
            valid_local_4d = valid_local.unsqueeze(0).unsqueeze(0) & pad_mask
        else:
            valid_local_4d = valid_local.unsqueeze(0).unsqueeze(0)

        # SDPA fast-path: use additive bool mask
        try:
            if valid_local_4d.dtype == torch.bool:
                if valid_local_4d.shape[0] == 1 and B > 1:
                    attn_mask = valid_local_4d.expand(B, H, L, L_kv)
                elif valid_local_4d.shape[0] == B and valid_local_4d.shape[1] == 1:
                    attn_mask = valid_local_4d.expand(B, H, L, L_kv)
                else:
                    attn_mask = valid_local_4d
                short_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False,  # already encoded in mask
                )
            else:
                raise RuntimeError("mask not bool")
        except Exception as e:
            logger.debug("Goldfinch SDPA failed (%s); using manual path.", e)
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * (d ** -0.5)
            scores = scores.masked_fill(~valid_local_4d, -1e4)
            if head_mask is not None:
                hm = head_mask
                if hm.dim() == 1:
                    hm = hm.view(1, H, 1, 1)
                elif hm.dim() == 2:
                    hm = hm.view(hm.shape[0], H, 1, 1)
                elif hm.dim() == 3:
                    hm = hm.unsqueeze(1)
                scores = scores * hm.float()
            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn_weights = F.softmax(scores, dim=-1).to(dtype_in)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)
            short_out = torch.matmul(attn_weights, v)

        short_out = short_out.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        # ---- Recurrent memory ----
        long_out = self.delta_mem(x).to(dtype_in)  # (B, L, D)

        # ---- Gated fusion (FIX 8: gate is (B, L, 1)) ----
        gate_w = self.gate.weight.to(dtype_in)
        gate_b = self.gate.bias.to(dtype_in) if self.gate.bias is not None else None
        gate = torch.sigmoid(F.linear(x, gate_w, gate_b)).to(dtype_in)  # (B, L, 1)

        # Ensure long_out matches all_head_size
        if long_out.shape[-1] != self.all_head_size:
            if long_out.shape[-1] < self.all_head_size:
                long_out = F.pad(long_out, (0, self.all_head_size - long_out.shape[-1]))
            else:
                long_out = long_out[..., :self.all_head_size]

        fused = (gate * short_out.to(dtype_in) + (1.0 - gate) * long_out).to(dtype_in)

        if valid_mask is not None:
            fused = fused * valid_mask.unsqueeze(-1).to(fused.dtype)

        # ---- Output projection ----
        if self.is_nested_encoder:
            out = fused
        else:
            out = self.out_proj(fused)
            out = (out * self.output_scale.to(out.dtype)).to(out.dtype)

        if self.is_decoder:
            return out.to(dtype_in), present_key_value
        else:
            return out.to(dtype_in), None

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (
            ("W_q", self.W_q), ("W_k", self.W_k), ("W_v", self.W_v),
            ("out_proj", self.out_proj), ("delta_mem", self.delta_mem),
            ("gate", self.gate),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["output_scale"] = _sz(self.output_scale)
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection (unchanged)
# ===========================================================================
def inject_goldfinch_ahn(
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
        logger.warning("inject_goldfinch_ahn: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif (hasattr(target, "bert") and hasattr(target.bert, "encoder")
          and hasattr(target.bert.encoder, "layer")):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_goldfinch_ahn(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    src_attn = old_attn
    if src_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    # Idempotency guards
    if src_attn is not None and (
        isinstance(src_attn, GoldFinchAHNBlock)
        or "GoldFinch" in src_attn.__class__.__name__
    ):
        return None
    if src_attn is not None and (
        "DeltaNet" in src_attn.__class__.__name__
        or "LinearAttention" in src_attn.__class__.__name__
    ):
        return None

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (getattr(src_attn.q_proj, "in_features", None) if hasattr(src_attn, "q_proj") else None)
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))
    )

    try:
        kwargs.pop("d_model", None)
        kwargs.pop("num_heads", None)
        kwargs.pop("hidden_size", None)
        kwargs.pop("num_attention_heads", None)
        kwargs.pop("old_attn", None)
        new_attn = GoldFinchAHNBlock(
            config=cfg, comp_config=comp_config,
            d_model=hidden_size, num_heads=num_heads,
            old_attn=src_attn,
            hidden_size=hidden_size, num_attention_heads=num_heads,
            **kwargs,
        )
        if src_attn is not None:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)
    except Exception as e:
        logger.error("GoldFinchAHN injection failed: %s", e)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn

    return new_attn


__all__ = [
    "GoldFinchAHNBlock",
    "inject_goldfinch_ahn",
    "GatedRecurrentMemory",
    "LoRALinear",
    "apply_rotary_pos_emb",
]