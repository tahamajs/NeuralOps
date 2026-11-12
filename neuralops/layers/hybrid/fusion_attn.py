#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FusionAttn: Multi-Paradigm Hybrid Sparse Attention

Multi-paradigm attention fusing local sliding-window attention with global low-rank sparse pathways.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.amp_utils import maybe_half
from neuralops.core.attention_utils import LoRALinear, apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.common import _read_past_kv, repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("fusion_attn", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
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
        elif source_attn is not None:
            nkv = getattr(source_attn, "num_key_value_heads", None) or getattr(source_attn, "num_kv_heads", None)
            if nkv is None:
                hd_probe = getattr(source_attn, "head_dim", None) or max(1, self.hidden_size // self.num_heads)
                for attr in ("k_proj", "key", "k_lin", "W_k"):
                    mod = getattr(source_attn, attr, None)
                    if mod is not None and hasattr(mod, "weight"):
                        base_mod = getattr(mod, "base_layer", mod)
                        w = base_mod.weight
                        is_conv = "Conv1D" in type(base_mod).__name__ or hasattr(base_mod, "nf")
                        out_dim = w.shape[1] if (is_conv and w.ndim == 2) else w.shape[0]
                        nkv = out_dim // max(1, int(hd_probe))
                        break
            self.num_kv_heads = int(nkv or self.num_heads)
        else:
            self.num_kv_heads = self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))

        cfg_hd = getattr(config, "head_dim", None) if config else None
        if cfg_hd is None and source_attn is not None:
            cfg_hd = getattr(source_attn, "head_dim", None)
        self.head_dim = int(cfg_hd) if cfg_hd is not None else max(1, self.hidden_size // self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        self.block_size = int(block_size)
        self.top_k_blocks = int(top_k_blocks)

        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (layer_idx if layer_idx is not None else kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
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
            is_dec = bool(getattr(config, "is_decoder", False) or getattr(config, "is_causal", False))
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
            cls_lower = source_attn.__class__.__name__.lower()
            if (
                hasattr(source_attn, "c_attn")
                or getattr(source_attn, "is_decoder", False)
                or getattr(source_attn, "is_causal", False)
                or any(p in cls_lower for p in ("qwen", "llama", "mistral", "gpt", "causal"))
            ):
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

        self.lora_attn_gate = nn.Sequential(
            nn.Linear(self.head_dim * 2, 4 * self.head_dim),
            nn.GELU(),
            nn.Linear(4 * self.head_dim, 1),
        )
        nn.init.normal_(self.lora_attn_gate[2].weight, mean=0.0, std=0.02)
        if self.lora_attn_gate[2].bias is not None:
            nn.init.zeros_(self.lora_attn_gate[2].bias)
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

        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None
        self.comp_config = comp_config

        if source_attn is not None:
            self.copy_weights(source_attn)

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
            if hasattr(c_attn_mod, "merge") and callable(c_attn_mod.merge) and not getattr(c_attn_mod, "merged", False):
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
                is_conv = "Conv1D" in cls_c or "Conv1D" in base_cls_c or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
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
                        "Conv1D" in cls_o or "Conv1D" in base_cls_o or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        def _extract_mod_wb(mod: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            if mod is None:
                return None, None
            if hasattr(mod, "merge") and callable(mod.merge) and not getattr(mod, "merged", False):
                try:
                    mod.merge()
                except Exception:
                    # allowed-swallow: best-effort-telemetry
                    pass
            base_m = getattr(mod, "base_layer", mod)
            w_m = getattr(base_m, "weight", getattr(mod, "weight", None))
            if w_m is None or not isinstance(w_m, torch.Tensor) or w_m.ndim != 2:
                return None, None
            w_m = w_m.data.clone()
            cls_m = type(mod).__name__
            base_cls_m = type(base_m).__name__
            if ("Conv1D" in cls_m) or ("Conv1D" in base_cls_m) or hasattr(base_m, "nf") or hasattr(mod, "nf"):
                w_m = w_m.t().contiguous()
            b_mod = getattr(base_m, "bias", getattr(mod, "bias", None))
            b_m = b_mod.data.clone() if isinstance(b_mod, torch.Tensor) else None
            return w_m, b_m

        if hasattr(pretrained_attn, "self") and hasattr(pretrained_attn.self, "query"):
            s = pretrained_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                w_m, b_m = _extract_mod_wb(getattr(s, sub, None))
                if w_m is not None:
                    extracted[key] = (w_m, b_m)
            if hasattr(pretrained_attn, "output") and hasattr(pretrained_attn.output, "dense"):
                w_m, b_m = _extract_mod_wb(pretrained_attn.output.dense)
                if w_m is not None:
                    extracted["o"] = (w_m, b_m)

        elif extracted["q"][0] is None:
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
                    if hasattr(pretrained_attn, n):
                        w_m, b_m = _extract_mod_wb(getattr(pretrained_attn, n))
                        if w_m is not None:
                            extracted[key] = (w_m, b_m)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
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

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
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
                if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().contiguous().shape:
                    w = w.t().contiguous()
                elif w.shape[1] != target_mod.weight.size(1) and w.shape[0] == target_mod.weight.size(1):
                    w = w.t().contiguous()

                tgt_out = target_mod.weight.size(0)
                if proj_key in ("k", "v") and w.size(0) != tgt_out:
                    hd = self.head_dim
                    src_out = w.size(0)
                    if src_out > tgt_out and src_out % tgt_out == 0 and tgt_out % hd == 0:
                        rep = src_out // tgt_out
                        w = w.view(self.num_kv_heads, rep, hd, -1).mean(dim=1).reshape(tgt_out, -1)
                        if b is not None and b.numel() == src_out:
                            b = b.view(self.num_kv_heads, rep, hd).mean(dim=1).reshape(tgt_out)
                    elif src_out < tgt_out and tgt_out % src_out == 0 and src_out % hd == 0:
                        rep = tgt_out // src_out
                        src_kv = src_out // hd
                        w = w.view(src_kv, hd, -1).repeat_interleave(rep, dim=0).reshape(tgt_out, -1)
                        if b is not None and b.numel() == src_out:
                            b = b.view(src_kv, hd).repeat_interleave(rep, dim=0).reshape(tgt_out)

                out_d = min(tgt_out, w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                target_mod.weight.data.zero_()
                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target_mod.weight.device, target_mod.weight.dtype)
                )

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data.zero_()
                    target_mod.bias.data[:min_b].copy_(b[:min_b].to(target_mod.bias.device, target_mod.bias.dtype))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning("FUSIONATTN: '%s' near-zero norm (%.2e)", proj_key, w_norm)

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    _read_past_kv = staticmethod(_read_past_kv)

    @staticmethod
    def _normalize_valid_mask(mask, B: int, N: int, device: torch.device) -> torch.Tensor | None:
        if mask is None:
            return None

        def _to_bool(t):
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e3
            return t > 0

        if mask.dim() == 2:
            m = _to_bool(mask)
            if m.shape[-1] == N:
                return m.to(device=device)
        elif mask.dim() == 3:
            m = _to_bool(mask)
            m = m.any(dim=-2)
            if m.shape[-1] == N:
                return m.to(device=device)
        elif mask.dim() == 4:
            m = _to_bool(mask)
            m = m.any(dim=1).any(dim=-2)
            if m.shape[-1] == N:
                return m.to(device=device)
        return None

    def _compute_gate_logits(
        self,
        q_cent: torch.Tensor,
        k_cent: torch.Tensor,
    ) -> torch.Tensor:
        """Compute differentiable content-aware + learned block gate logits [B, H, n_q, n_k]."""
        B, H, _, D = q_cent.shape
        raw_affinity = torch.matmul(q_cent.float(), k_cent.float().transpose(-2, -1)) * (1.0 / math.sqrt(int(D)))
        gate_dtype = self.lora_attn_gate[0].weight.dtype
        q_f = q_cent.to(gate_dtype)
        k_f = k_cent.to(gate_dtype)
        q_feat = torch.cat([q_f, torch.zeros_like(q_f)], dim=-1)
        k_feat = torch.cat([torch.zeros_like(k_f), k_f], dim=-1)
        q_gate = self.lora_attn_gate(q_feat).squeeze(-1).float()
        k_gate = self.lora_attn_gate(k_feat).squeeze(-1).float()
        learned_gate = torch.matmul(q_gate.unsqueeze(-1), k_gate.unsqueeze(-2))
        temp = self.lora_gate_temp.abs().clamp(min=0.1, max=10.0).float()
        head_bias = self.lora_gate_head_bias.float().view(1, H, 1, 1)
        return (raw_affinity + learned_gate + head_bias) / temp

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
        scale = 1.0 / math.sqrt(int(D))

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        gate_dt = self.lora_attn_gate[0].weight.dtype
        q_g = self.lora_attn_gate(torch.cat([q.to(gate_dt), torch.zeros_like(q.to(gate_dt))], dim=-1)).squeeze(-1).float()
        k_g = self.lora_attn_gate(torch.cat([torch.zeros_like(k.to(gate_dt)), k.to(gate_dt)], dim=-1)).squeeze(-1).float()
        temp = self.lora_gate_temp.abs().clamp(min=0.1, max=10.0).float()
        head_bias = self.lora_gate_head_bias.float().view(1, H, 1, 1)
        gate_bias = 0.05 * torch.tanh((q_g.unsqueeze(-1) * k_g.unsqueeze(-2) + head_bias) / temp)
        scores = scores + gate_bias

        invalid = torch.zeros((B, 1, N, N_kv), device=q.device, dtype=torch.bool)

        if is_causal and N > 1:
            past_len = max(0, N_kv - N)
            q_idx = torch.arange(N, device=q.device).unsqueeze(1)
            k_idx = torch.arange(N_kv, device=q.device).unsqueeze(0)
            invalid = invalid | (k_idx > (q_idx + past_len)).unsqueeze(0).unsqueeze(0)

        if valid_mask is not None:
            invalid = invalid | (~valid_mask[:, None, None, :])

        all_masked = invalid.all(dim=-1, keepdim=True)
        safe_invalid = invalid.clone()
        if all_masked.any():
            safe_invalid[..., :1] = safe_invalid[..., :1] & (~all_masked)

        neg_inf = -65500.0 if scores.dtype == torch.float16 else -1e4
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4).masked_fill(safe_invalid, neg_inf).clamp(min=-65500.0, max=65500.0)
        probs = F.softmax(scores, dim=-1).masked_fill(all_masked, 0.0)
        probs = self.dropout(probs)

        if head_mask is not None:
            hm = head_mask.float()
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            probs = probs * hm

        context = torch.matmul(probs, v.float()).to(q.dtype)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        return context, (probs.to(q.dtype) if output_attentions else None)

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
        past_len = max(0, N_kv - N)
        bs = self.block_size
        device = q.device

        pad_q = (bs - (N % bs)) % bs
        pad_k = (bs - (N_kv % bs)) % bs
        q_pad = F.pad(q, (0, 0, 0, pad_q)) if pad_q > 0 else q
        k_pad = F.pad(k, (0, 0, 0, pad_k)) if pad_k > 0 else k
        v_pad = F.pad(v, (0, 0, 0, pad_k)) if pad_k > 0 else v

        if valid_mask is not None:
            valid_pad = F.pad(valid_mask, (0, pad_k), value=False) if pad_k > 0 else valid_mask
        elif pad_k > 0:
            valid_pad = torch.cat(
                [
                    torch.ones((B, N_kv), device=device, dtype=torch.bool),
                    torch.zeros((B, pad_k), device=device, dtype=torch.bool),
                ],
                dim=-1,
            )
        else:
            valid_pad = None

        N_q_pad = q_pad.size(2)
        N_k_pad = k_pad.size(2)
        num_q_blocks = N_q_pad // bs
        num_k_blocks = N_k_pad // bs

        q_blocks = q_pad.reshape(B, H, num_q_blocks, bs, D)
        k_blocks = k_pad.reshape(B, H, num_k_blocks, bs, D)
        v_blocks = v_pad.reshape(B, H, num_k_blocks, bs, D)

        q_cent = q_blocks.float().mean(dim=3)
        if valid_pad is not None:
            valid_k_blk = valid_pad.reshape(B, 1, num_k_blocks, bs).float()
            k_cnt = valid_k_blk.sum(dim=-1, keepdim=True).clamp(min=1.0)
            k_cent = (k_blocks.float() * valid_k_blk.unsqueeze(-1)).sum(dim=3) / k_cnt
            v_cent = (v_blocks.float() * valid_k_blk.unsqueeze(-1)).sum(dim=3) / k_cnt
            block_valid = valid_k_blk.sum(dim=-1) > 0
        else:
            k_cent = k_blocks.float().mean(dim=3)
            v_cent = v_blocks.float().mean(dim=3)
            block_valid = torch.ones((B, 1, num_k_blocks), device=device, dtype=torch.bool)

        gate_scores = self._compute_gate_logits(q_cent, k_cent)

        q_blk = torch.arange(num_q_blocks, device=device).unsqueeze(1)
        k_blk = torch.arange(num_k_blocks, device=device).unsqueeze(0)
        if is_causal:
            q_blk_max_pos = torch.clamp((q_blk + 1) * bs - 1, max=N - 1) + past_len
            k_blk_min_pos = k_blk * bs
            causal_blocks = k_blk_min_pos <= q_blk_max_pos
        else:
            causal_blocks = torch.ones((num_q_blocks, num_k_blocks), device=device, dtype=torch.bool)

        allowed_blocks = causal_blocks.unsqueeze(0).unsqueeze(0) & block_valid.unsqueeze(2)
        neg_inf_gate = -65500.0 if gate_scores.dtype == torch.float16 else -1e4
        gate_masked = gate_scores.masked_fill(~allowed_blocks, neg_inf_gate).clamp(min=-65500.0, max=65500.0)

        top_k = min(self.top_k_blocks, num_k_blocks)
        _, top_idx = torch.topk(gate_masked, top_k, dim=-1)

        # Guarantee sink (0), local diagonal block(s), and immediate neighbor blocks are never dropped.
        sel_mask = torch.zeros((B, H, num_q_blocks, num_k_blocks), device=device, dtype=torch.bool)
        sel_mask.scatter_(-1, top_idx, True)
        sel_mask[:, :, :, 0] = True
        q_start_kblk = torch.clamp((q_blk.squeeze(1) * bs + past_len) // bs, max=num_k_blocks - 1)
        q_end_kblk = torch.clamp((torch.clamp((q_blk.squeeze(1) + 1) * bs - 1, max=N - 1) + past_len) // bs, max=num_k_blocks - 1)
        q_idx_1d = torch.arange(num_q_blocks, device=device)
        sel_mask[:, :, q_idx_1d, q_start_kblk] = True
        sel_mask[:, :, q_idx_1d, q_end_kblk] = True
        prev_kblk = torch.clamp(q_start_kblk - 1, min=0)
        sel_mask[:, :, q_idx_1d, prev_kblk] = True
        if not is_causal:
            next_kblk = torch.clamp(q_end_kblk + 1, max=num_k_blocks - 1)
            sel_mask[:, :, q_idx_1d, next_kblk] = True
        sel_mask = sel_mask & allowed_blocks

        max_sel = max(1, min(num_k_blocks, int(sel_mask.sum(dim=-1).max().item())))
        sel_priority = torch.where(
            sel_mask,
            100.0 + gate_scores.clamp(-50.0, 50.0),
            gate_scores.clamp(-50.0, 50.0) - 100.0,
        ).masked_fill(~allowed_blocks, neg_inf_gate).clamp(min=-65500.0, max=65500.0)
        _, sel_idx = torch.topk(sel_priority, k=max_sel, dim=-1)
        sel_blk_valid = torch.gather(allowed_blocks.expand(B, H, num_q_blocks, num_k_blocks), -1, sel_idx)
        sel_gate_vals = torch.gather(gate_scores, -1, sel_idx)

        sel_idx_exp = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, bs, D)
        k_exp = k_blocks.unsqueeze(2).expand(-1, -1, num_q_blocks, -1, -1, -1)
        v_exp = v_blocks.unsqueeze(2).expand(-1, -1, num_q_blocks, -1, -1, -1)

        k_sel = torch.gather(k_exp, 3, sel_idx_exp).reshape(B, H, num_q_blocks, max_sel * bs, D)
        v_sel = torch.gather(v_exp, 3, sel_idx_exp).reshape(B, H, num_q_blocks, max_sel * bs, D)

        if valid_pad is not None:
            valid_blocks = valid_pad.reshape(B, 1, num_k_blocks, bs)
            valid_exp = valid_blocks.unsqueeze(2).expand(-1, H, num_q_blocks, -1, -1)
            sel_idx_m = sel_idx.unsqueeze(-1).expand(-1, -1, -1, -1, bs)
            valid_sel = torch.gather(valid_exp, 3, sel_idx_m).reshape(B, H, num_q_blocks, max_sel * bs)
        else:
            valid_sel = None

        scale = 1.0 / math.sqrt(int(D))
        scores_tok = torch.matmul(q_blocks.float(), k_sel.float().transpose(-2, -1).contiguous()) * scale
        gate_bias_tok = (0.05 * torch.tanh(sel_gate_vals)).unsqueeze(-1).expand(-1, -1, -1, -1, bs)
        scores_tok = scores_tok + gate_bias_tok.reshape(B, H, num_q_blocks, 1, max_sel * bs)

        invalid_tok = (~sel_blk_valid).unsqueeze(-1).expand(-1, -1, -1, -1, bs).reshape(
            B, H, num_q_blocks, 1, max_sel * bs
        )
        if is_causal:
            q_pos = (
                torch.arange(num_q_blocks, device=device).view(num_q_blocks, 1) * bs
                + torch.arange(bs, device=device).view(1, bs)
                + past_len
            ).view(1, 1, num_q_blocks, bs, 1)
            k_base = sel_idx.unsqueeze(-1) * bs
            k_off = torch.arange(bs, device=device).view(1, 1, 1, 1, bs)
            k_pos = (k_base + k_off).view(B, H, num_q_blocks, 1, max_sel * bs)
            invalid_tok = invalid_tok | (k_pos > q_pos)

        if valid_sel is not None:
            invalid_tok = invalid_tok | (~valid_sel.unsqueeze(-2))

        # Compressed centroid pathway for unselected allowed causal blocks so distant context is preserved.
        unsel_mask = allowed_blocks & (~sel_mask)
        has_unsel = bool(unsel_mask.any().item())
        if has_unsel:
            scores_cent = torch.matmul(q_blocks.float(), k_cent.unsqueeze(2).transpose(-2, -1)) * scale
            scores_cent = scores_cent + 0.05 * torch.tanh(gate_scores.unsqueeze(3))
            invalid_cent = ~unsel_mask.unsqueeze(3).expand(-1, H, -1, bs, -1)
            scores_all = torch.cat([scores_tok, scores_cent], dim=-1)
            invalid_all = torch.cat([invalid_tok.expand(-1, -1, -1, bs, -1), invalid_cent], dim=-1)
        else:
            scores_all = scores_tok
            invalid_all = invalid_tok.expand(-1, -1, -1, bs, -1)

        all_masked = invalid_all.all(dim=-1, keepdim=True)
        safe_invalid = invalid_all.clone()
        if all_masked.any():
            safe_invalid[..., :1] = safe_invalid[..., :1] & (~all_masked)

        neg_inf_all = -65500.0 if scores_all.dtype == torch.float16 else -1e4
        scores_all = torch.nan_to_num(scores_all, nan=0.0, posinf=1e4, neginf=-1e4).masked_fill(safe_invalid, neg_inf_all).clamp(min=-65500.0, max=65500.0)
        probs = F.softmax(scores_all, dim=-1).masked_fill(all_masked, 0.0)
        probs = self.dropout(probs)

        if head_mask is not None:
            hm = head_mask.float()
            if hm.dim() == 1:
                hm = hm.view(1, H, 1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], H, 1, 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1).unsqueeze(-1)
            probs = probs * hm

        tok_len = max_sel * bs
        out_blocks = torch.matmul(probs[..., :tok_len], v_sel.float())
        if has_unsel:
            out_blocks = out_blocks + torch.matmul(probs[..., tok_len:], v_cent.unsqueeze(2))
        out_blocks = torch.nan_to_num(out_blocks, nan=0.0, posinf=1e4, neginf=-1e4).to(q.dtype)

        context = out_blocks.reshape(B, H, N_q_pad, D)[:, :, :N, :]
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if output_attentions:
            return context, probs.to(q.dtype)
        return context, None

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

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k_new = self.k_proj(kv_input).view(B, kv_len_new, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v_new = self.v_proj(kv_input).view(B, kv_len_new, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if not is_cross_attention:
            past_k_probe, _, _ = self._read_past_kv(past_key_value, self.layer_idx)
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

        past_k, past_v, _ = self._read_past_kv(past_key_value, self.layer_idx)
        if past_k is not None and not is_cross_attention:
            k_cat = torch.cat([past_k, k_new], dim=-2)
            v_cat = torch.cat([past_v, v_new], dim=-2)
        else:
            k_cat, v_cat = k_new, v_new

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross_attention:
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
                is_causal=is_causal,
                output_attentions=bool(output_attentions),
            )
        else:
            context, attn_weights = self._block_sparse_attention(
                q,
                k_cat,
                v_cat,
                valid_mask,
                head_mask,
                is_causal=is_causal,
                output_attentions=bool(output_attentions),
            )

        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        use_cache = bool(kwargs.get("use_cache", False))
        if self.is_decoder and not is_cross_attention and (use_cache or past_key_value is not None):
            if past_key_value is not None and hasattr(past_key_value, "update"):
                new_past_kv = past_key_value
            else:
                new_past_kv = (k_cat, v_cat)
            return out, new_past_kv

        return out, (attn_weights if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.weight.device)


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

    if src_attn is not None and (isinstance(src_attn, FUSIONATTN) or "FUSIONATTN" in src_attn.__class__.__name__):
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
