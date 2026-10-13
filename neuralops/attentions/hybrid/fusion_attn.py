"""
FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention Module.

Hybrid sparse attention with:
  - Short-sequence fast path (full attention via SDPA / manual)
  - Long-sequence block-gated sparse attention (MoBA-style)
  - Per-head dynamic gating for block selection
  - RoPE support with partial-dim slicing
  - GQA with repeat_kv
  - Nested encoder bypass
  - LoRA adapters on frozen base projections
  - Strict causal masking across both paths
  - Padding masking across both paths
  - KV-cache support
"""

from __future__ import annotations
from neuralops.core.common import repeat_kv

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

from neuralops.core.amp_utils import maybe_half
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)




# ===========================================================================
# FUSIONATTN
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "fusion_attn", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
class FUSIONATTN(BaseAttention):
    """
    Hybrid sparse attention with block gating.

    Two execution paths:
      * Short sequence (N <= 128) or cross-attention  → dense attention
      * Long sequence                                  → block-gated sparse
    """

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        block_size: int = 64,
        top_k_blocks: int = 4,
        rank: int | None = 64,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        if config is not None:
            hidden_size = int(
                getattr(
                    config,
                    "hidden_size",
                    getattr(config, "d_model", getattr(config, "n_embd", hidden_size)),
                )
            )
            num_heads = int(
                getattr(
                    config,
                    "num_attention_heads",
                    getattr(config, "num_heads", getattr(config, "n_heads", num_heads)),
                )
            )
        elif source_attn is not None:
            out_layer = None
            for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
                if hasattr(source_attn, attr):
                    out_layer = getattr(source_attn, attr)
                    break
            if out_layer is not None:
                hidden_size = int(
                    getattr(
                        out_layer,
                        "out_features",
                        getattr(out_layer, "in_features", hidden_size),
                    )
                )
            num_heads = int(
                getattr(
                    source_attn,
                    "n_heads",
                    getattr(
                        source_attn,
                        "num_attention_heads",
                        getattr(source_attn, "num_heads", num_heads),
                    ),
                )
            )

        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        self.hidden_size = int(hidden_size)
        self.num_heads = max(1, int(num_heads))

        if config is not None:
            self.num_kv_heads = int(
                getattr(
                    config,
                    "num_key_value_heads",
                    getattr(config, "num_kv_heads", self.num_heads),
                )
            )
        else:
            self.num_kv_heads = self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))

        cfg_hd = getattr(config, "head_dim", None) if config else None
        self.head_dim = (
            int(cfg_hd) if cfg_hd is not None else max(1, self.hidden_size // self.num_heads)
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        self.block_size = int(block_size)
        self.top_k_blocks = int(top_k_blocks)

        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (
                layer_idx
                if layer_idx is not None
                else kwargs.get("layer_id", kwargs.get("layer_idx", 0))
            )
        )
        self.layer_idx = int(resolved_layer_id)

        self.lora_rank = int(
            getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
            if config
            else kwargs.get("lora_rank", 16)
        )
        self.lora_alpha = float(
            getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
            if config
            else kwargs.get("lora_alpha", 32.0)
        )
        self.freeze_base = self.lora_rank > 0

        model_type = str(getattr(config, "model_type", "") if config else "").lower()
        is_dec = False
        if config is not None:
            is_dec = bool(
                getattr(config, "is_decoder", False) or getattr(config, "is_causal", False)
            )
        _dec_prefixes = (
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
        if any(model_type.startswith(p) for p in _dec_prefixes):
            is_dec = True
        if config is not None and hasattr(config, "architectures"):
            arch = getattr(config, "architectures", []) or []
            if any("GPT" in str(a) or "CausalLM" in str(a) for a in arch):
                is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if source_attn is not None:
            if hasattr(source_attn, "c_attn") or getattr(source_attn, "is_decoder", False):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
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

        # KD-FIX: lora_ prefix so KD optimizer tracks these params
        self.lora_attn_gate = nn.Sequential(
            nn.Linear(self.head_dim * 2, 4 * self.head_dim),
            nn.GELU(),
            nn.Linear(4 * self.head_dim, 1),
        )
        self.lora_gate_head_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.lora_gate_temp = nn.Parameter(torch.ones(1))

        dropout_p = (
            float(
                getattr(
                    config,
                    "dropout",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                )
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None) if source_attn is not None else None
        )
        self.comp_config = comp_config

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ---- aliases ----
    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_proj

    @property
    def v_lin(self):
        return self.v_proj

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def W_q(self):
        return self.q_proj

    @property
    def W_k(self):
        return self.k_proj

    @property
    def W_v(self):
        return self.v_proj

    @property
    def c_proj(self):
        return self.out_proj

    # ------------------------------------------------------------------
    # copy_weights — unchanged from original (working correctly)
    # ------------------------------------------------------------------
    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        if pretrained_attn is None:
            return

        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = any(hasattr(pretrained_attn, a) for a in ("out_lin", "c_proj", "o_proj"))
        is_nested = False
        if (
            (
                hasattr(pretrained_attn, "self")
                and hasattr(pretrained_attn.self, "query")
                and hasattr(pretrained_attn, "output")
                and hasattr(pretrained_attn.output, "dense")
            )
            or (
                cls_name
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                )
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(pretrained_attn, "rotary_emb"):
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c_attn_mod = pretrained_attn.c_attn
            if (
                hasattr(c_attn_mod, "merge")
                and callable(c_attn_mod.merge)
                and not getattr(c_attn_mod, "merged", False)
            ):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    logger.debug("c_attn merge skipped: %s", e)

            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                cls_c = c_attn_mod.__class__.__name__
                base_cls_c = base_c.__class__.__name__
                is_conv = (
                    "Conv1D" in cls_c
                    or "Conv1D" in base_cls_c
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
                c_proj_mod = pretrained_attn.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    cls_o = c_proj_mod.__class__.__name__
                    base_cls_o = base_o.__class__.__name__
                    is_conv_o = (
                        "Conv1D" in cls_o
                        or "Conv1D" in base_cls_o
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        elif hasattr(pretrained_attn, "self") and hasattr(pretrained_attn.self, "query"):
            s = pretrained_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                mod = getattr(s, sub, None)
                if mod is not None and hasattr(mod, "weight"):
                    b = getattr(mod, "bias", None)
                    extracted[key] = (
                        mod.weight.data.clone(),
                        b.data.clone() if b is not None else None,
                    )
            if hasattr(pretrained_attn, "output") and hasattr(pretrained_attn.output, "dense"):
                out_mod = pretrained_attn.output.dense
                b = getattr(out_mod, "bias", None)
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    b.data.clone() if b is not None else None,
                )

        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query", "W_q"),
                "k": ("k_lin", "k_proj", "key", "W_k"),
                "v": ("v_lin", "v_proj", "value", "W_v"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(pretrained_attn, n):
                        mod = getattr(pretrained_attn, n)
                        if mod is not None and hasattr(mod, "weight"):
                            b = getattr(mod, "bias", None)
                            extracted[key] = (
                                mod.weight.data.clone(),
                                b.data.clone() if b is not None else None,
                            )
                            break

        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(pretrained_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            logger.warning("FUSIONATTN copy_weights: missing %s from %s", missing_after, cls_name)

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if (
                    w.shape != target_mod.weight.shape
                    and target_mod.weight.shape == w.t().contiguous().shape
                ):
                    w = w.t().contiguous()

                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target_mod.weight.device, target_mod.weight.dtype)
                )

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(target_mod.bias.device, target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning("FUSIONATTN: '%s' near-zero norm (%.2e)", proj_key, w_norm)

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # KV-cache helpers
    # ------------------------------------------------------------------
    _read_past_kv = staticmethod(_read_past_kv)

    @staticmethod
    def _normalize_valid_mask(mask, B: int, N: int, device: torch.device) -> torch.Tensor | None:
        if mask is None:
            return None

        def _to_bool(t):
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if mask.dim() == 2:
            m = _to_bool(mask)
            if m.shape[-1] == N:
                return m
        elif mask.dim() == 4:
            m = _to_bool(mask)
            m = m.any(dim=-2).squeeze(1)
            if m.shape[-1] == N:
                return m
        return None

    # ------------------------------------------------------------------
    # Dense path
    # ------------------------------------------------------------------
    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_mask: torch.Tensor | None,
        head_mask: torch.Tensor | None,
        is_causal: bool,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, H, N, D = q.shape
        N_kv = k.shape[2]
        scale = D**-0.5

        use_sdpa = (
            not output_attentions
            and head_mask is None
            and (valid_mask is None or valid_mask.all())
            and not self.training
        )

        if use_sdpa:
            try:
                _qs = maybe_half(q)
                _ks = maybe_half(k)
                _vs = maybe_half(v)
                _out = F.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    is_causal=is_causal and N > 1 and N_kv == N,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                _out = _out.to(q.dtype).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
                _out = torch.nan_to_num(_out, nan=0.0, posinf=1e4, neginf=-1e4)
                return _out, None
            except Exception as e:
                logger.debug("SDPA failed, falling back to manual: %s", e)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        if is_causal and N > 1:
            past_len = max(0, N_kv - N)
            causal = (
                torch.triu(
                    torch.full((N, N_kv), float("-inf"), device=q.device, dtype=torch.float32),
                    diagonal=1 + past_len,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(v.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        if head_mask is not None:
            hm = head_mask.to(probs.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            probs = probs * hm

        context = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        return context, (probs if output_attentions else None)

    # ------------------------------------------------------------------
    # Block-gated sparse attention (CLEANED)
    # ------------------------------------------------------------------
    def _block_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_mask: torch.Tensor | None,
        head_mask: torch.Tensor | None,
        is_causal: bool,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, H, N, D = q.shape
        N_kv = k.shape[2]
        bs = self.block_size
        device = q.device

        # --- Pad to multiple of block_size ---
        pad_len = (bs - (N % bs)) % bs
        if pad_len > 0:
            q_pad = F.pad(q, (0, 0, 0, pad_len))
            k_pad = F.pad(k, (0, 0, 0, pad_len))
            v_pad = F.pad(v, (0, 0, 0, pad_len))
            valid_pad = (
                F.pad(valid_mask, (0, pad_len), value=False) if valid_mask is not None else None
            )
        else:
            q_pad, k_pad, v_pad, valid_pad = q, k, v, valid_mask

        N_pad = q_pad.size(2)
        num_blocks = N_pad // bs
        top_k = min(self.top_k_blocks, num_blocks)

        # --- Block centroids ---
        q_blocks = q_pad.reshape(B, H, num_blocks, bs, D)
        k_blocks = k_pad.reshape(B, H, num_blocks, bs, D)
        v_blocks = v_pad.reshape(B, H, num_blocks, bs, D)

        q_cent = q_blocks.mean(dim=3)  # (B, H, nb, D)
        k_cent = k_blocks.mean(dim=3)  # (B, H, nb, D)

        # --- Gate features ---
        q_feat = torch.cat([q_cent, torch.zeros_like(q_cent)], dim=-1)
        k_feat = torch.cat([torch.zeros_like(k_cent), k_cent], dim=-1)

        q_gate = self.lora_attn_gate(q_feat).squeeze(-1)  # (B, H, nb)
        k_gate = self.lora_attn_gate(k_feat).squeeze(-1)  # (B, H, nb)

        # --- Gate scores (B, H, nb_q, nb_k) ---
        gate_scores = torch.matmul(q_gate.unsqueeze(-1), k_gate.unsqueeze(-2))
        gate_scores = (gate_scores + self.lora_gate_head_bias.view(1, H, 1, 1)) / (
            self.lora_gate_temp.abs() + 1e-5
        )

        # --- Causal mask at block level ---
        if is_causal:
            q_blk = torch.arange(num_blocks, device=device).unsqueeze(1)
            k_blk = torch.arange(num_blocks, device=device).unsqueeze(0)
            past_blocks = 0  # N == N_kv in this path
            causal_blocks = k_blk <= (q_blk + past_blocks)
            gate_scores = gate_scores.masked_fill(
                ~causal_blocks.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # --- Top-k block selection ---
        _, top_idx = torch.topk(gate_scores, top_k, dim=-1)  # (B, H, nb_q, top_k)

        # --- Gather K / V / valid ---
        top_idx_exp = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, bs, D)
        k_exp = k_blocks.unsqueeze(2).expand(-1, -1, num_blocks, -1, -1, -1)
        v_exp = v_blocks.unsqueeze(2).expand(-1, -1, num_blocks, -1, -1, -1)

        k_sel = torch.gather(k_exp, 3, top_idx_exp).reshape(B, H, num_blocks, top_k * bs, D)
        v_sel = torch.gather(v_exp, 3, top_idx_exp).reshape(B, H, num_blocks, top_k * bs, D)

        if valid_pad is not None:
            valid_blocks = valid_pad.reshape(B, 1, num_blocks, bs)
            valid_exp = valid_blocks.unsqueeze(2).expand(-1, H, num_blocks, -1, -1)
            top_idx_m = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, bs)
            valid_sel = torch.gather(valid_exp, 3, top_idx_m).reshape(
                B, H, num_blocks, top_k * bs
            )
        else:
            valid_sel = None

        # --- Attention ---
        scale = D**-0.5
        scores = (
            torch.matmul(q_blocks.float(), k_sel.float().transpose(-2, -1).contiguous()) * scale
        )

        if is_causal:
            q_pos = (
                torch.arange(num_blocks, device=device).view(num_blocks, 1) * bs
                + torch.arange(bs, device=device).view(1, bs)
            )
            q_pos = q_pos.view(1, 1, num_blocks, bs, 1)

            k_base = top_idx.unsqueeze(-1) * bs
            k_off = torch.arange(bs, device=device).view(1, 1, 1, 1, bs)
            k_pos = (k_base + k_off).view(B, H, num_blocks, 1, top_k * bs)

            causal_tok = k_pos <= q_pos
            scores = scores.masked_fill(~causal_tok, float("-inf"))

        if valid_sel is not None:
            scores = scores.masked_fill(~valid_sel.unsqueeze(-2), float("-inf"))

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(v_sel.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        if head_mask is not None:
            hm = head_mask.to(probs.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, H, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], H, 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            probs = probs * hm

        out_blocks = torch.matmul(probs, v_sel)  # (B, H, nb, bs, D)
        out_blocks = torch.nan_to_num(out_blocks, nan=0.0, posinf=1e4, neginf=-1e4)

        context = out_blocks.reshape(B, H, N_pad, D)[:, :, :N, :]
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if output_attentions:
            return context, probs
        return context, None

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
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, _ = hidden_states.shape
        device = hidden_states.device
        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len_new = kv_input.shape[1]

        q = (
            self.q_proj(hidden_states)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k_new = (
            self.k_proj(kv_input)
            .view(B, kv_len_new, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v_new = (
            self.v_proj(kv_input)
            .view(B, kv_len_new, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        if not is_cross_attention:
            past_k_probe, _ = self._read_past_kv(past_key_value, self.layer_idx)
            past_len = 0 if past_k_probe is None else past_k_probe.shape[-2]

            if position_embeddings is not None and isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
                if past_len > 0:
                    cos = cos[..., past_len : past_len + N, :]
                    sin = sin[..., past_len : past_len + N, :]
                q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, position_ids)
            elif getattr(self, "rotary_emb", None) is not None:
                try:
                    cos, sin = self.rotary_emb(hidden_states, seq_len=past_len + N)
                except TypeError:
                    try:
                        cos, sin = self.rotary_emb(hidden_states, position_ids)
                    except Exception:
                        cos, sin = None, None
                if cos is not None and sin is not None:
                    if past_len > 0:
                        cos = cos[..., past_len : past_len + N, :]
                        sin = sin[..., past_len : past_len + N, :]
                    q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, position_ids)

        past_k, past_v = self._read_past_kv(past_key_value, self.layer_idx)
        if past_k is not None and not is_cross_attention:
            k_cat = torch.cat([past_k, k_new], dim=-2)
            v_cat = torch.cat([past_v, v_new], dim=-2)
        else:
            k_cat, v_cat = k_new, v_new

        if (
            past_key_value is not None
            and hasattr(past_key_value, "update")
            and not is_cross_attention
        ):
            past_key_value.update(k_new, v_new, self.layer_idx)

        if self.num_kv_heads < self.num_heads:
            k_cat = repeat_kv(k_cat, self.kv_repeat)
            v_cat = repeat_kv(v_cat, self.kv_repeat)

        N_kv = k_cat.shape[2]

        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
        valid_mask = self._normalize_valid_mask(eff_mask, B, N_kv, device)

        is_causal = self.is_decoder and not is_cross_attention and N > 1

        if N <= 128 or is_cross_attention:
            context, attn_weights = self._dense_attention(
                q,
                k_cat,
                v_cat,
                valid_mask,
                head_mask,
                is_causal=is_causal and N_kv == N,
                output_attentions=bool(output_attentions),
            )
        else:
            context, attn_weights = self._block_sparse_attention(
                q,
                k_cat,
                v_cat,
                valid_mask,
                head_mask,
                is_causal=is_causal and N_kv == N,
                output_attentions=bool(output_attentions),
            )

        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        if self.is_decoder and not is_cross_attention:
            if past_key_value is not None and hasattr(past_key_value, "update"):
                new_past_kv = past_key_value
            else:
                new_past_kv = (k_cat, v_cat)
            return out, new_past_kv

        return out, (attn_weights if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.weight.device)


# ===========================================================================
# Injection
# ===========================================================================
def inject_fusion_attn(
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
        logger.warning("inject_fusion_attn: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
        if ok and isinstance(obj, (torch.nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_fusion_attn(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, torch.nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    if src_attn is not None and (
        isinstance(src_attn, FUSIONATTN) or "FUSIONATTN" in src_attn.__class__.__name__
    ):
        return None

    try:
        new_attn = FUSIONATTN(
            config=cfg,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )
        p = next(src_attn.parameters(), None) if src_attn is not None else None
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
        else:
            logger.error(
                "inject_fusion_attn: no writable attention attribute on %s",
                target.__class__.__name__,
            )
            return None

        return new_attn
    except Exception as e:
        logger.exception("inject_fusion_attn failed at layer %d: %s", layer_idx, e)
        return None


__all__ = ["FUSIONATTN", "inject_fusion_attn"]