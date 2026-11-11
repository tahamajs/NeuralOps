#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
MoE-DeltaKV-Paged Hybrid Attention

Hybrid attention combining mixture-of-block attention with paged delta KV-cache management.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import LoRALinear, apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv

logger = logging.getLogger(__name__)


def ste_quantize(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    Symmetric uniform quantization with straight-through estimator.

    MDP-FIX-02: returns the quantized tensor; caller is responsible for
    checking reconstruction error and deciding whether to keep it.
    """
    if x.numel() == 0 or bits <= 0:
        return x
    qmax = float((1 << (int(bits) - 1)) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale
    return x + (x_q - x).detach()


def low_rank_delta_compress(
    delta: torch.Tensor,
    rank: int,
    bits: int,
) -> tuple[torch.Tensor, float]:
    """
    Low-rank + STE-quantize a temporal delta tensor.

    Supports both 4D `(B, H, T, d)` and 5D `(B, H, P, T, d)` batched per-page inputs.
    Returns (compressed, relative_reconstruction_error).
    """
    if delta.numel() == 0:
        return delta, 0.0

    d = delta.shape[-1]
    original_norm = delta.float().norm().item() + 1e-12

    if delta.dim() == 5:
        B, H, P, T, _ = delta.shape
        flat = delta.reshape(B * H * P, T, d)
        r = min(rank, T, d)
    else:
        B, H, T, _ = delta.shape
        flat = delta.reshape(-1, d)
        r = min(rank, flat.shape[0], flat.shape[1])

    if r < 1:
        compressed = ste_quantize(delta, bits)
        err = (delta - compressed).float().norm().item() / original_norm
        return compressed, err

    try:
        U, S, Vh = torch.linalg.svd(flat.float(), full_matrices=False)
        U_r = U[..., :, :r]
        S_r = S[..., :r]
        Vh_r = Vh[..., :r, :]
        approx = ((U_r * S_r.unsqueeze(-2)) @ Vh_r).to(delta.dtype)
        approx = approx.reshape(delta.shape)
    except Exception as e:
        logger.debug("SVD failed (%s); falling back to direct quantization.", e)
        approx = delta

    compressed = ste_quantize(approx, bits)
    err = (delta - compressed).float().norm().item() / original_norm
    return compressed, err


def simulate_paged_delta_kv(
    K: torch.Tensor,
    V: torch.Tensor,
    rank: int,
    bits: int,
    max_delta_error: float = 0.35,
    page_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Delta-KV temporal compression with reconstruction-error guard and STE gradient pass-through.

    MDP-FIX-02: if the reconstruction error exceeds `max_delta_error`, the
    raw K/V is returned instead of the compressed version.
    When `page_size` is provided (> 1), compression runs independently per page so
    future pages never leak into earlier pages in causal mode.
    Runs SVD/quantization under no_grad and uses STE (K + (K_hat - K).detach())
    so singular-value degeneracies in torch.linalg.svd backward never produce NaNs.

    Returns (K_hat, V_hat, err_K, err_V).
    """
    B, H, N, d = K.shape
    if N <= 1:
        return K, V, 0.0, 0.0

    with torch.no_grad():
        if page_size is not None and page_size > 1:
            num_full_pages = N // page_size
            full_len = num_full_pages * page_size
            K_hat_val = K.clone()
            V_hat_val = V.clone()
            err_K, err_V = 0.0, 0.0

            if num_full_pages > 0:
                Kp = K[:, :, :full_len, :].view(B, H, num_full_pages, page_size, d)
                Vp = V[:, :, :full_len, :].view(B, H, num_full_pages, page_size, d)
                dKp = Kp[:, :, :, 1:, :] - Kp[:, :, :, :-1, :]
                dVp = Vp[:, :, :, 1:, :] - Vp[:, :, :, :-1, :]
                dKp_hat, err_K = low_rank_delta_compress(dKp, rank, bits)
                dVp_hat, err_V = low_rank_delta_compress(dVp, rank, bits)
                if err_K <= max_delta_error and err_V <= max_delta_error:
                    Kp_rec = torch.empty_like(Kp)
                    Vp_rec = torch.empty_like(Vp)
                    Kp_rec[:, :, :, 0, :] = Kp[:, :, :, 0, :]
                    Vp_rec[:, :, :, 0, :] = Vp[:, :, :, 0, :]
                    Kp_rec[:, :, :, 1:, :] = Kp[:, :, :, 0:1, :] + torch.cumsum(dKp_hat, dim=3)
                    Vp_rec[:, :, :, 1:, :] = Vp[:, :, :, 0:1, :] + torch.cumsum(dVp_hat, dim=3)
                    K_hat_val[:, :, :full_len, :] = Kp_rec.reshape(B, H, full_len, d)
                    V_hat_val[:, :, :full_len, :] = Vp_rec.reshape(B, H, full_len, d)

            tail_len = N - full_len
            if tail_len > 1:
                Kt = K[:, :, full_len:, :]
                Vt = V[:, :, full_len:, :]
                dKt = Kt[:, :, 1:, :] - Kt[:, :, :-1, :]
                dVt = Vt[:, :, 1:, :] - Vt[:, :, :-1, :]
                dKt_hat, err_Kt = low_rank_delta_compress(dKt, rank, bits)
                dVt_hat, err_Vt = low_rank_delta_compress(dVt, rank, bits)
                if err_Kt <= max_delta_error and err_Vt <= max_delta_error:
                    K_hat_val[:, :, full_len + 1 :, :] = Kt[:, :, 0:1, :] + torch.cumsum(dKt_hat, dim=2)
                    V_hat_val[:, :, full_len + 1 :, :] = Vt[:, :, 0:1, :] + torch.cumsum(dVt_hat, dim=2)
                err_K = max(err_K, err_Kt)
                err_V = max(err_V, err_Vt)
        else:
            delta_K = K[:, :, 1:, :] - K[:, :, :-1, :]
            delta_V = V[:, :, 1:, :] - V[:, :, :-1, :]

            delta_K_hat, err_K = low_rank_delta_compress(delta_K, rank, bits)
            delta_V_hat, err_V = low_rank_delta_compress(delta_V, rank, bits)

            if err_K > max_delta_error or err_V > max_delta_error:
                return K, V, err_K, err_V

            K_hat_val = torch.empty_like(K)
            V_hat_val = torch.empty_like(V)
            K_hat_val[:, :, 0, :] = K[:, :, 0, :]
            V_hat_val[:, :, 0, :] = V[:, :, 0, :]
            K_hat_val[:, :, 1:, :] = K[:, :, 0:1, :] + torch.cumsum(delta_K_hat, dim=2)
            V_hat_val[:, :, 1:, :] = V[:, :, 0:1, :] + torch.cumsum(delta_V_hat, dim=2)

    K_hat = K + (K_hat_val - K).detach()
    V_hat = V + (V_hat_val - V).detach()
    return K_hat, V_hat, err_K, err_V


@ATTENTION_REGISTRY.register("moe_deltakv_paged", supports_causal=True, supports_gqa=True)
class MoEDeltaKVPagedAttention(BaseAttention):
    """
    MoE-DeltaKV-Paged Hybrid Attention.

    Two execution paths:
      * Full dense attention when `num_pages <= 1` (page_size >= seq_len)
        OR when `enable_moe=False`.
      * Paged MoE attention otherwise.
    """

    def __init__(
        self,
        config: Any | None = None,
        page_size: int = 16,
        k_experts: int = 4,
        svd_rank: int = 32,
        q_bits: int = 4,
        max_delta_error: float = 0.35,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        enable_moe: bool = True,
        enable_deltakv: bool = True,
        is_decoder: bool | None = None,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)

        if config is None:

            class _Cfg:
                pass

            config = _Cfg()
            config.hidden_size = kwargs.get("hidden_size", 768)
            config.num_attention_heads = kwargs.get("num_attention_heads", 12)
            config.head_dim = kwargs.get("head_dim", 64)
            config.model_type = kwargs.get("model_type", "")
            config.is_decoder = kwargs.get("is_decoder", False)

        self.hidden_size = int(
            getattr(
                config,
                "hidden_size",
                getattr(config, "d_model", kwargs.get("hidden_size", 768)),
            )
        )
        self.num_heads = int(
            getattr(
                config,
                "num_attention_heads",
                getattr(config, "num_heads", kwargs.get("num_attention_heads", 12)),
            )
        )
        self.head_dim = int(
            getattr(
                config,
                "head_dim",
                kwargs.get("head_dim", self.hidden_size // max(1, self.num_heads)),
            )
        )

        cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim

        self.page_size = max(4, int(page_size))
        self.k_experts = max(1, int(k_experts))
        self.svd_rank = max(4, int(svd_rank))
        self.q_bits = int(q_bits)
        if self.q_bits not in (2, 4, 8):
            self.q_bits = 4
        self.max_delta_error = float(max_delta_error)

        self.enable_moe = bool(enable_moe)
        self.enable_deltakv = bool(enable_deltakv)
        self.lora_rank = max(0, int(lora_rank))
        self.lora_alpha = float(lora_alpha)

        if is_decoder is not None:
            self.is_causal = bool(is_decoder)
        else:
            _mt = str(getattr(config, "model_type", "") or "").lower()
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
            self.is_causal = bool(
                getattr(config, "is_causal", False)
                or getattr(config, "is_decoder", False)
                or kwargs.get("is_causal", False)
                or kwargs.get("is_decoder", False)
                or any(_mt.startswith(p) for p in _dec_pfx)
            )
        self.is_decoder = self.is_causal
        self.is_nested_encoder = False

        self.query = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.key = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.value = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        self.W_rg = nn.Linear(self.head_dim, self.head_dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.W_rg.weight)

        self.register_buffer("aux_loss", torch.zeros(1), persistent=False)

        self.register_buffer("delta_kv_error_k", torch.zeros(1), persistent=False)
        self.register_buffer("delta_kv_error_v", torch.zeros(1), persistent=False)

        dropout_p = float(
            getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                ),
            )
        )
        self.dropout = nn.Dropout(dropout_p)

        self._orig_storage_bytes: int = 0
        self._comp_storage_bytes: int = 0
        self._source_id: int | None = None

        src = pretrained_attn or kwargs.get("old_attn")
        if src is not None:
            self.copy_weights(src)

    @property
    def q_proj(self):
        return self.query

    @property
    def k_proj(self):
        return self.key

    @property
    def v_proj(self):
        return self.value

    @property
    def q_lin(self):
        return self.query

    @property
    def k_lin(self):
        return self.key

    @property
    def v_lin(self):
        return self.value

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

    def storage_bytes(self) -> dict[str, int]:
        return {"orig": self._orig_storage_bytes, "comp": self._comp_storage_bytes}

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        from neuralops.core.adapters import UniversalAdapter

        if self._source_id is not None and id(src) == self._source_id:
            return

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
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass

        if hasattr(src, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            try:
                mod_c = src.c_attn
                base_c = getattr(mod_c, "base_layer", mod_c)
                weight = getattr(base_c, "weight", getattr(mod_c, "weight", None))
                if weight is not None:
                    weight = weight.data.clone()
                    bias = getattr(base_c, "bias", getattr(mod_c, "bias", None))
                    bias_data = bias.data.clone() if bias is not None else None
                    cls_c = mod_c.__class__.__name__
                    base_cls = base_c.__class__.__name__
                    if (
                        hasattr(mod_c, "nf")
                        or hasattr(base_c, "nf")
                        or "Conv1D" in cls_c
                        or "Conv1D" in base_cls
                        or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                    ):
                        weight = weight.t().contiguous()
                    split = weight.shape[0] // 3
                    extracted["q"] = (
                        weight[:split],
                        bias_data[:split] if bias_data is not None else None,
                    )
                    extracted["k"] = (
                        weight[split : 2 * split],
                        bias_data[split : 2 * split] if bias_data is not None else None,
                    )
                    extracted["v"] = (
                        weight[2 * split :],
                        bias_data[2 * split :] if bias_data is not None else None,
                    )
            except Exception as e:
                logger.debug("GPT-2 c_attn extraction failed: %s", e)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            try:
                mod_o = src.c_proj
                base_o = getattr(mod_o, "base_layer", mod_o)
                w_o = getattr(base_o, "weight", getattr(mod_o, "weight", None))
                if w_o is not None:
                    w_o = w_o.data.clone()
                    cls_o = mod_o.__class__.__name__
                    base_cls_o = base_o.__class__.__name__
                    if (
                        hasattr(mod_o, "nf")
                        or hasattr(base_o, "nf")
                        or "Conv1D" in cls_o
                        or "Conv1D" in base_cls_o
                        or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                    ):
                        w_o = w_o.t().contiguous()
                    b_o = getattr(base_o, "bias", getattr(mod_o, "bias", None))
                    extracted["o"] = (
                        w_o,
                        b_o.data.clone() if b_o is not None else None,
                    )
            except Exception as e:
                logger.debug("GPT-2 c_proj extraction failed: %s", e)

        if hasattr(src, "self") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            try:
                s = src.self
                if hasattr(s, "query") and extracted["q"][0] is None:
                    extracted["q"] = (
                        s.query.weight.data.clone(),
                        s.query.bias.data.clone() if s.query.bias is not None else None,
                    )
                if hasattr(s, "key") and extracted["k"][0] is None:
                    extracted["k"] = (
                        s.key.weight.data.clone(),
                        s.key.bias.data.clone() if s.key.bias is not None else None,
                    )
                if hasattr(s, "value") and extracted["v"][0] is None:
                    extracted["v"] = (
                        s.value.weight.data.clone(),
                        s.value.bias.data.clone() if s.value.bias is not None else None,
                    )
            except Exception as e:
                logger.debug("Nested self extraction failed: %s", e)

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            try:
                dense = src.output.dense
                extracted["o"] = (
                    dense.weight.data.clone(),
                    dense.bias.data.clone() if dense.bias is not None else None,
                )
            except Exception as e:
                logger.debug("output.dense extraction failed: %s", e)

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
                        try:
                            layer = getattr(src, name)
                            if hasattr(layer, "weight"):
                                extracted[proj] = (
                                    layer.weight.data.clone(),
                                    (layer.bias.data.clone() if getattr(layer, "bias", None) is not None else None),
                                )
                                break
                        except Exception:
                            continue

        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, a)
            for a in (
                "dense",
                "out_proj",
                "c_proj",
                "out_lin",
                "o_proj",
                "wo",
                "o",
                "proj",
            )
        )
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            or cls_name
            in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
            )
        ):
            is_nested = True
        elif cls_name in ("DistilBertAttention", "DistilBertSABlock"):
            is_nested = True
        elif (
            cls_name.endswith("SelfAttention")
            and not has_out_proj
            or cls_name.endswith("Attention")
            and not has_out_proj
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("MoEDeltaKVPaged: missing %s; aborting copy.", missing)
            return

        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

        with torch.no_grad():
            for proj_key, target in (
                ("q", self.query),
                ("k", self.key),
                ("v", self.value),
            ):
                w, b = extracted[proj_key]
                if w is None:
                    continue
                in_dim = self.hidden_size
                out_dim = target.weight.shape[0]

                if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                    w = w.t().contiguous()

                if w.shape[0] < out_dim and out_dim % w.shape[0] == 0:
                    rep = out_dim // w.shape[0]
                    w = w.repeat(rep, 1)
                    if b is not None:
                        b = b.repeat(rep)
                elif w.shape[0] > out_dim and w.shape[0] % out_dim == 0:
                    rep = w.shape[0] // out_dim
                    w = w.view(out_dim, rep, -1).mean(dim=1)
                    if b is not None:
                        b = b.view(out_dim, rep).mean(dim=1)

                target.weight.data.zero_()
                out_rows = min(out_dim, w.shape[0])
                in_cols = min(in_dim, w.shape[1])
                target.weight.data[:out_rows, :in_cols].copy_(
                    w[:out_rows, :in_cols].to(target.weight.device, target.weight.dtype)
                )
                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data.zero_()
                    target.bias.data[:mb].copy_(b[:mb].to(target.bias.device, target.bias.dtype))

            w_o, b_o = extracted["o"]
            if w_o is not None and not self.is_nested_encoder:
                in_dim = self.all_head_size
                out_dim = self.hidden_size
                if w_o.dim() == 2 and w_o.shape[1] != in_dim and w_o.shape[0] == in_dim:
                    w_o = w_o.t().contiguous()
                self.out_proj.weight.data.zero_()
                out_rows = min(out_dim, w_o.shape[0])
                in_cols = min(in_dim, w_o.shape[1])
                self.out_proj.weight.data[:out_rows, :in_cols].copy_(
                    w_o[:out_rows, :in_cols].to(self.out_proj.weight.device, self.out_proj.weight.dtype)
                )
                if b_o is not None and self.out_proj.bias is not None:
                    mb = min(self.out_proj.bias.numel(), b_o.numel())
                    self.out_proj.bias.data.zero_()
                    self.out_proj.bias.data[:mb].copy_(b_o[:mb].to(self.out_proj.bias.device, self.out_proj.bias.dtype))
            elif self.is_nested_encoder:
                pass

            for m in (self.query, self.key, self.value, self.out_proj):
                m.freeze_base_weights()

        try:
            for m in (self.query, self.key, self.value, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
            self.is_causal = self.is_decoder

        self._source_id = id(src)

    def _paged_moe_attention(
        self,
        q: torch.Tensor,
        k_hat: torch.Tensor,
        v_hat: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, H, N, d = q.shape
        device = q.device
        page_size = self.page_size
        num_pages = (N + page_size - 1) // page_size
        pad_len = num_pages * page_size - N

        with torch.amp.autocast(device_type=device.type, enabled=False):
            q_f = q.float()
            k_f = k_hat.float()
            v_f = v_hat.float()

            if pad_len > 0:
                q_pad = F.pad(q_f, (0, 0, 0, pad_len))
                k_pad = F.pad(k_f, (0, 0, 0, pad_len))
                v_pad = F.pad(v_f, (0, 0, 0, pad_len))
                valid_pad = F.pad(valid_mask, (0, pad_len), value=False)
            else:
                q_pad, k_pad, v_pad, valid_pad = q_f, k_f, v_f, valid_mask

            padded_N = num_pages * page_size

            q_blocks = q_pad.view(B, H, num_pages, page_size, d)
            k_blocks = k_pad.view(B, H, num_pages, page_size, d)
            v_blocks = v_pad.view(B, H, num_pages, page_size, d)

            valid_blocks = valid_pad.view(B, 1, num_pages, page_size).float()

            valid_sum = valid_blocks.sum(dim=-1, keepdim=True).clamp_min(1.0)
            q_bar = (q_blocks * valid_blocks.unsqueeze(-1)).sum(dim=3) / valid_sum
            k_bar = (k_blocks * valid_blocks.unsqueeze(-1)).sum(dim=3) / valid_sum

            q_bar_rg = F.linear(
                q_bar,
                self.W_rg.weight.float(),
                self.W_rg.bias.float() if getattr(self.W_rg, "bias", None) is not None else None,
            )
            router_logits = torch.matmul(q_bar_rg, k_bar.transpose(-1, -2).contiguous()) / math.sqrt(d)

            page_valid = valid_pad.view(B, 1, num_pages, page_size).any(dim=-1)
            router_invalid = ~page_valid.unsqueeze(2).expand(B, 1, num_pages, num_pages)
            if self.is_causal:
                causal_pages = torch.tril(torch.ones(num_pages, num_pages, device=device, dtype=torch.bool))
                router_invalid = router_invalid | (~causal_pages.unsqueeze(0).unsqueeze(0))

            router_all_masked = router_invalid.all(dim=-1, keepdim=True)
            router_safe_mask = router_invalid.clone()
            if router_all_masked.any():
                router_safe_mask[..., 0:1] = router_safe_mask[..., 0:1] & (~router_all_masked)

            f32_min = torch.finfo(torch.float32).min
            router_logits = router_logits.masked_fill(router_safe_mask, f32_min)

            router_logits_for_topk = router_logits.clone()
            router_logits_for_topk[:, :, :, 0] = router_logits_for_topk[:, :, :, 0] + 1e4

            k_experts = min(self.k_experts, num_pages)
            _, topk_indices = torch.topk(router_logits_for_topk, k_experts, dim=-1)
            topk_logits = torch.gather(router_logits, -1, topk_indices)
            g_j = F.softmax(topk_logits, dim=-1)

            router_probs = F.softmax(router_logits, dim=-1)
            expert_mask = F.one_hot(topk_indices, num_classes=num_pages).float()
            density = expert_mask.sum(dim=-2).mean(dim=(0, 1, 2))
            prob_mean = router_probs.mean(dim=(0, 1, 2))
            with torch.no_grad():
                self.aux_loss.copy_(num_pages * torch.sum(density * prob_mean).detach())

            expanded_idx = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, page_size, d)
            k_blocks_exp = k_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1, -1)
            v_blocks_exp = v_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1, -1)
            k_gathered = torch.gather(k_blocks_exp, dim=3, index=expanded_idx)
            v_gathered = torch.gather(v_blocks_exp, dim=3, index=expanded_idx)

            valid_blocks_exp = valid_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1)
            exp_valid_idx = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, page_size)
            valid_blocks_exp_h = valid_blocks_exp.expand(-1, H, -1, -1, -1)
            k_valid = torch.gather(valid_blocks_exp_h, dim=3, index=exp_valid_idx)
            attn_invalid = ~k_valid.bool().unsqueeze(-2).expand(-1, -1, -1, -1, page_size, -1)

            if self.is_causal:
                q_pos = (
                    torch.arange(num_pages, device=device).view(1, 1, num_pages, 1, 1, 1) * page_size
                    + torch.arange(page_size, device=device).view(1, 1, 1, 1, page_size, 1)
                )
                k_pos = (
                    topk_indices.unsqueeze(-1).unsqueeze(-1) * page_size
                    + torch.arange(page_size, device=device).view(1, 1, 1, 1, 1, page_size)
                )
                attn_invalid = attn_invalid | (k_pos > q_pos)

            attn_all_masked = attn_invalid.all(dim=-1, keepdim=True)
            attn_safe_mask = attn_invalid.clone()
            if attn_all_masked.any():
                attn_safe_mask[..., 0:1] = attn_safe_mask[..., 0:1] & (~attn_all_masked)

            q_exp = q_blocks.unsqueeze(3)
            attn_scores = torch.matmul(q_exp, k_gathered.transpose(-1, -2).contiguous()) / math.sqrt(d)
            attn_scores = attn_scores.masked_fill(attn_safe_mask, f32_min)

            scores_max = attn_scores.amax(dim=-1, keepdim=True).detach()
            exp_scores = torch.exp((attn_scores - scores_max).clamp(max=0.0)).masked_fill(attn_safe_mask, 0.0)
            attn_den = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-4)
            attn_probs = (exp_scores / attn_den).masked_fill(attn_all_masked, 0.0)
            attn_probs_drop = self.dropout(attn_probs)

            context_per_expert = torch.matmul(attn_probs_drop, v_gathered)
            g_j_exp = g_j.unsqueeze(-1).unsqueeze(-1)
            context = (context_per_expert * g_j_exp).sum(dim=3)
            context = context.reshape(B, H, padded_N, d)

        return context, attn_probs

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        H_kv = self.num_kv_heads
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        k = self.key(kv_input).view(B, kv_len, H_kv, d).transpose(1, 2).contiguous()
        v = self.value(kv_input).view(B, kv_len, H_kv, d).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.enable_deltakv and kv_len > 1:
            k_hat_kv, v_hat_kv, err_k, err_v = simulate_paged_delta_kv(
                k,
                v,
                rank=self.svd_rank,
                bits=self.q_bits,
                max_delta_error=self.max_delta_error,
                page_size=self.page_size if (self.is_causal and not is_cross) else None,
            )
            with torch.no_grad():
                self.delta_kv_error_k.copy_(torch.tensor(err_k, device=device))
                self.delta_kv_error_v.copy_(torch.tensor(err_v, device=device))
        else:
            k_hat_kv, v_hat_kv = k, v

        if H_kv != H:
            k_hat = repeat_kv(k_hat_kv, self.kv_repeat)
            v_hat = repeat_kv(v_hat_kv, self.kv_repeat)
        else:
            k_hat, v_hat = k_hat_kv, v_hat_kv

        valid_mask: torch.Tensor | None = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.dtype == torch.bool:
                    valid_mask = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    valid_mask = attention_mask > -1e4
                else:
                    valid_mask = attention_mask > 0
            elif attention_mask.dim() == 3:
                m = attention_mask
                valid_mask = (m > -1e4).squeeze(1) if m.dtype.is_floating_point else m.squeeze(1)
            elif attention_mask.dim() == 4:
                m = attention_mask
                v_bool = m if m.dtype == torch.bool else (m > -1e4 if m.dtype.is_floating_point else m > 0)
                valid_mask = v_bool.any(dim=-2).squeeze(1)
            if valid_mask is not None and valid_mask.shape[-1] != kv_len:
                valid_mask = None

        if valid_mask is None:
            valid_mask = torch.ones(B, kv_len, device=device, dtype=torch.bool)

        num_pages = (kv_len + self.page_size - 1) // self.page_size
        use_paged = self.enable_moe and num_pages > 1 and not is_cross

        if use_paged:
            context, attn_probs = self._paged_moe_attention(q, k_hat, v_hat, valid_mask)
        else:
            scale = 1.0 / math.sqrt(d)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                scores = torch.matmul(q.float(), k_hat.float().transpose(-2, -1).contiguous()) * scale

                invalid_mask = ~valid_mask[:, None, None, :].bool().expand(B, 1, N, kv_len)
                if self.is_causal and not is_cross and N > 1:
                    q_idx = torch.arange(N, device=device).unsqueeze(1)
                    k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
                    offset = max(0, kv_len - N)
                    causal_invalid = k_idx > (q_idx + offset)
                    invalid_mask = invalid_mask | causal_invalid.unsqueeze(0).unsqueeze(0)

                all_masked = invalid_mask.all(dim=-1, keepdim=True)
                safe_mask = invalid_mask.clone()
                if all_masked.any():
                    safe_mask[..., 0:1] = safe_mask[..., 0:1] & (~all_masked)

                f32_min = torch.finfo(torch.float32).min
                scores = scores.masked_fill(safe_mask, f32_min)
                scores_max = scores.amax(dim=-1, keepdim=True).detach()
                exp_scores = torch.exp((scores - scores_max).clamp(max=0.0)).masked_fill(safe_mask, 0.0)
                den = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-4)
                attn_probs = (exp_scores / den).masked_fill(all_masked, 0.0)
                attn_probs_drop = self.dropout(attn_probs)
                context = torch.matmul(attn_probs_drop, v_hat.float())

            with torch.no_grad():
                self.aux_loss.zero_()

        if head_mask is not None:
            hm = head_mask.float()
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            context = context * hm

        context = context[:, :, :N, :] if context.shape[2] != N else context
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        if dtype == torch.float16:
            fmax = torch.finfo(dtype).max * 0.5
            context = context.clamp(min=-fmax, max=fmax)
        context = context.to(dtype)

        if self.is_nested_encoder:
            out = context
        else:
            proj_dtype = self.out_proj.weight.dtype if hasattr(self.out_proj, "weight") else context.dtype
            out = self.out_proj(context.to(proj_dtype))

        attn_ret = attn_probs if output_attentions else None
        return out, attn_ret

    def step(self, x: torch.Tensor, state: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Autoregressive single-step forward for generation.

        Args:
            x: (B, 1, D) input tensor.
            state: Optional dictionary containing past 'k_cache' and 'v_cache'.

        Returns:
            (out, new_state) where out is (B, 1, D) and new_state is updated dict.
        """
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim
        H_kv = self.num_kv_heads

        q = self.query(x).view(B, N, H, d).transpose(1, 2)
        k_t = self.key(x).view(B, N, H_kv, d).transpose(1, 2)
        v_t = self.value(x).view(B, N, H_kv, d).transpose(1, 2)

        if state is None:
            state = {}

        if "k_cache" in state and "v_cache" in state:
            k = torch.cat([state["k_cache"], k_t], dim=2)
            v = torch.cat([state["v_cache"], v_t], dim=2)
        else:
            k = k_t
            v = v_t

        new_state = {
            "k_cache": k,
            "v_cache": v,
        }

        if H_kv != H:
            k_exp = repeat_kv(k, self.kv_repeat)
            v_exp = repeat_kv(v, self.kv_repeat)
        else:
            k_exp = k
            v_exp = v

        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q.float(), k_exp.float().transpose(-2, -1)) * scale
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(x.dtype)
        context = torch.matmul(probs, v_exp)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, new_state


def inject_moe_deltakv_paged(
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
        logger.warning("inject_moe_deltakv_paged: recursion depth exceeded.")
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
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_moe_deltakv_paged(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    if src_attn is None:
        return None

    if isinstance(src_attn, MoEDeltaKVPagedAttention) or "MoEDeltaKV" in src_attn.__class__.__name__:
        return None

    if cfg is None and hasattr(src_attn, "config"):
        cfg = src_attn.config

    page_size = int(kwargs.pop("page_size", getattr(comp_config, "page_size", 16) if comp_config else 16))
    k_experts = int(kwargs.pop("k_experts", getattr(comp_config, "k_experts", 4) if comp_config else 4))
    svd_rank = int(kwargs.pop("svd_rank", getattr(comp_config, "svd_rank", 32) if comp_config else 32))
    q_bits = int(kwargs.pop("q_bits", getattr(comp_config, "q_bits", 4) if comp_config else 4))
    max_delta_error = float(
        kwargs.pop(
            "max_delta_error",
            getattr(comp_config, "max_delta_error", 0.35) if comp_config else 0.35,
        )
    )
    lora_rank = int(kwargs.pop("lora_rank", getattr(comp_config, "lora_rank", 16) if comp_config else 16))
    enable_moe = bool(kwargs.pop("enable_moe", True))
    enable_deltakv = bool(kwargs.pop("enable_deltakv", True))

    try:
        new_attn = MoEDeltaKVPagedAttention(
            config=cfg,
            page_size=page_size,
            k_experts=k_experts,
            svd_rank=svd_rank,
            q_bits=q_bits,
            max_delta_error=max_delta_error,
            lora_rank=lora_rank,
            enable_moe=enable_moe,
            enable_deltakv=enable_deltakv,
            pretrained_attn=src_attn if src_attn is not target else None,
        )
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)
    except Exception as e:
        logger.error("MoEDeltaKVPaged construction failed: %s", e)
        return None

    spliced = False
    if hasattr(target, "attention") and hasattr(target.attention, "self"):
        new_attn.is_nested_encoder = True
        target.attention.self = new_attn
        spliced = True
    elif hasattr(target, "attention"):
        new_attn.is_nested_encoder = False
        target.attention = new_attn
        spliced = True
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
        spliced = True
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
        spliced = True
    elif hasattr(target, "attn"):
        new_attn.is_nested_encoder = False
        target.attn = new_attn
        spliced = True

    if not spliced:
        logger.error(
            "inject_moe_deltakv_paged: no writable attribute on %s",
            type(target).__name__,
        )
        return None

    return new_attn


__all__ = [
    "MoEDeltaKVPagedAttention",
    "inject_moe_deltakv_paged",
    "simulate_paged_delta_kv",
    "ste_quantize",
    "low_rank_delta_compress",
    "apply_rotary_pos_emb",
    "LoRALinear",
]
