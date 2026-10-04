"""
Multi-Head Latent Attention (MLA) – DeepSeek-V2).

Paper: https://arxiv.org/abs/2405.04434

Reference implementation for DeepSeek-V2, adapted for multi-architecture
deployment across DistilBERT, RoBERTa, GPT-2, Qwen/LLaMA.

Note on encoder-only models:
  MLA was designed for decoder inference with KV-cache compression. On
  encoder-only models (BERT/RoBERTa/DistilBERT) it acts as a low-rank
  bottleneck on K/V. If the reconstruction error is too high the model
  loses accuracy. This version validates reconstruction error and falls
  back to the original attention when needed.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.triton_kernels import triton_dispatch

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    # MLA-FIX-09
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **fk))
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
        # MLA-FIX-11
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# MLA Attention
# ===========================================================================
@ATTENTION_REGISTRY.register("mla", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("mla_attention", supports_causal=True, supports_gqa=True)
class MLAAttention(BaseAttention):
    """
    Multi-Head Latent Attention (MLA) — DeepSeek-V2 style.

    Idea: compress K and V into a low-rank latent `c_kv` of size `rank`,
    then up-project back to per-head K and V.

        c_kv = kv_down(x)             # (B, N, rank)
        k    = k_up(c_kv)              # (B, N, num_kv_heads * head_dim)
        v    = v_up(c_kv)              # (B, N, num_kv_heads * head_dim)

    Query is projected with a LoRA-augmented linear. K/V down/up projections
    are initialized via joint SVD of [W_k; W_v] and frozen.
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "low_rank"

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 128,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        max_error: float = 0.35,
        fallback_to_original: bool = True,
        **kwargs: Any,
    ) -> None:
        # MLA-FIX-05: config None-safe
        if config is None:

            class _Cfg:
                pass

            config = _Cfg()
            config.hidden_size = kwargs.get("hidden_size", 768)
            config.num_attention_heads = kwargs.get("num_attention_heads", 12)
            config.head_dim = kwargs.get("head_dim", 64)
            config.model_type = kwargs.get("model_type", "")
            config.num_key_value_heads = kwargs.get(
                "num_key_value_heads", kwargs.get("num_attention_heads", 12)
            )
            config.is_decoder = kwargs.get("is_decoder", False)

        try:
            super().__init__(config, rank if rank is not None else 128)
        except TypeError:
            super().__init__(config)

        self.hidden_size: int = int(
            getattr(
                config,
                "hidden_size",
                getattr(config, "d_model", kwargs.get("hidden_size", 768)),
            )
        )
        self.num_heads: int = int(
            getattr(
                config,
                "num_attention_heads",
                getattr(
                    config,
                    "n_heads",
                    getattr(config, "n_head", getattr(config, "num_heads", 12)),
                ),
            )
        )
        self.head_dim: int = int(
            getattr(config, "head_dim", self.hidden_size // max(1, self.num_heads))
        )
        self.q_dim: int = int(getattr(config, "q_proj_out", self.num_heads * self.head_dim))
        self.all_head_size: int = self.q_dim

        cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_dim: int = self.num_kv_heads * self.head_dim
        self.kv_head_size: int = self.kv_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.rank: int = int(getattr(config, "mla_rank", rank if rank is not None else 128))
        self.lora_rank: int = int(lora_rank)
        self.lora_alpha: float = float(lora_alpha)
        self.max_error: float = float(max_error)
        self.fallback_to_original: bool = bool(fallback_to_original)

        # ---- decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
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
            self.is_decoder = (
                bool(getattr(config, "is_decoder", False))
                or bool(getattr(config, "causal", False))
                or any(_mt.startswith(p) for p in _dec_pfx)
            )

        self.is_nested_encoder: bool = False

        # ---- projections ----
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.q_dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.kv_down = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.k_up = nn.Linear(self.rank, self.kv_dim, bias=False)
        self.v_up = nn.Linear(self.rank, self.kv_dim, bias=False)
        self.out_proj = LoRALinear(
            self.q_dim,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        dropout_p = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        )
        self.dropout = nn.Dropout(dropout_p)

        # ---- storage accounting (MLA-FIX-08) ----
        self._orig_storage_bytes: int = 0
        self._comp_storage_bytes: int = 0
        self._reconstruction_error_kv: float | None = None

        # ---- source attention (for injector) ----
        self._source_attn_id: int | None = None
        self.triton_kwargs = {}

        pretrained_attn = kwargs.get("pretrained_attn")
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    # Aliases (MLA-FIX-02, MLA-FIX-10)
    # ------------------------------------------------------------------
    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_up

    @property
    def value(self):
        return self.v_up

    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_up

    @property
    def v_lin(self):
        return self.v_up

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        # MLA-FIX-06: idempotency guard
        if self._source_attn_id is not None and id(src) == self._source_attn_id:
            logger.debug("MLAAttention.copy_weights: source already applied, skipping")
            return

        # --- extract weights ---
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
                logger.debug("UniversalAdapter fallback failed for %s: %s", proj, e)

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            c_attn = src.c_attn
            base_c = getattr(c_attn, "base_layer", c_attn)
            weight = getattr(base_c, "weight", getattr(c_attn, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                is_conv = (
                    c_attn.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size)
                )
                if is_conv:
                    weight = weight.t().contiguous()
                split = weight.shape[0] // 3
                bias = getattr(base_c, "bias", getattr(c_attn, "bias", None))
                bd = bias.data if bias is not None else None
                extracted["q"] = (
                    weight[:split],
                    bd[:split] if bd is not None else None,
                )
                extracted["k"] = (
                    weight[split : 2 * split],
                    bd[split : 2 * split] if bd is not None else None,
                )
                extracted["v"] = (
                    weight[2 * split :],
                    bd[2 * split :] if bd is not None else None,
                )

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj = src.c_proj
            base_o = getattr(c_proj, "base_layer", c_proj)
            w_o = getattr(base_o, "weight", getattr(c_proj, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                if (
                    w_o.dim() == 2
                    and w_o.shape[0] != self.hidden_size
                    and w_o.shape[1] == self.hidden_size
                ):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(c_proj, "bias", None))
                extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # BERT / RoBERTa nested
        if hasattr(src, "self") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            s = src.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # Flat modules
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
                            extracted[proj] = (
                                layer.weight.data,
                                getattr(layer, "bias", None),
                            )
                            break

        # --- nested encoder detection (MLA-FIX-07) ---
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
            or cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention")
            or cls_name.endswith("SelfAttention")
            and not has_out_proj
            or cls_name.endswith("Attention")
            and not has_out_proj
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        # --- required weights check ---
        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("MLAAttention.copy_weights: missing %s; aborting.", missing)
            return

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # --- storage accounting ---
        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        with torch.no_grad():
            # ---------- 1. Q ----------
            w_q, b_q = extracted["q"]
            w_q_oriented = _orient(w_q, self.hidden_size)
            out_q = min(self.q_dim, w_q_oriented.shape[0])
            in_q = min(self.hidden_size, w_q_oriented.shape[1])
            self.q_proj.weight.data.zero_()
            self.q_proj.weight.data[:out_q, :in_q].copy_(
                w_q_oriented[:out_q, :in_q].to(self.q_proj.weight.device, self.q_proj.weight.dtype)
            )
            if b_q is not None and self.q_proj.bias is not None:
                mb = min(self.q_proj.bias.numel(), b_q.numel())
                self.q_proj.bias.data.zero_()
                self.q_proj.bias.data[:mb].copy_(
                    b_q[:mb].to(self.q_proj.bias.device, self.q_proj.bias.dtype)
                )
            self.q_proj.freeze_base_weights()

            # ---------- 2. Output ----------
            w_o, b_o = extracted["o"]
            if w_o is not None:
                w_o_oriented = _orient(w_o, self.q_dim)
                out_o = min(self.hidden_size, w_o_oriented.shape[0])
                in_o = min(self.q_dim, w_o_oriented.shape[1])
                self.out_proj.weight.data.zero_()
                self.out_proj.weight.data[:out_o, :in_o].copy_(
                    w_o_oriented[:out_o, :in_o].to(
                        self.out_proj.weight.device, self.out_proj.weight.dtype
                    )
                )
                if b_o is not None and self.out_proj.bias is not None:
                    mb = min(self.out_proj.bias.numel(), b_o.numel())
                    self.out_proj.bias.data.zero_()
                    self.out_proj.bias.data[:mb].copy_(
                        b_o[:mb].to(self.out_proj.bias.device, self.out_proj.bias.dtype)
                    )
            elif self.is_nested_encoder:
                # Identity init for nested encoders
                min_d = min(self.out_proj.weight.shape[0], self.out_proj.weight.shape[1])
                self.out_proj.weight.data.zero_()
                self.out_proj.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                if self.out_proj.bias is not None:
                    self.out_proj.bias.data.zero_()
            self.out_proj.freeze_base_weights()

            # ---------- 3. K/V low-rank SVD ----------
            w_k, _ = extracted["k"]
            w_v, _ = extracted["v"]
            w_k_f = _orient(w_k, self.hidden_size).float()
            w_v_f = _orient(w_v, self.hidden_size).float()

            # GQA dimension matching
            def _match_gqa(w, target_dim, hidden_size):
                if w.shape[0] < target_dim and target_dim % w.shape[0] == 0:
                    rep = target_dim // w.shape[0]
                    return w.repeat(rep, 1)
                if w.shape[0] > target_dim and w.shape[0] % target_dim == 0:
                    rep = w.shape[0] // target_dim
                    return w.view(target_dim, rep, -1).mean(dim=1)
                return w

            w_k_f = _match_gqa(w_k_f, self.kv_dim, self.hidden_size)
            w_v_f = _match_gqa(w_v_f, self.kv_dim, self.hidden_size)

            # Joint SVD of [K; V]
            w_kv = torch.cat(
                [
                    w_k_f[: self.kv_dim, : self.hidden_size],
                    w_v_f[: self.kv_dim, : self.hidden_size],
                ],
                dim=0,
            )  # (2 * kv_dim, hidden_size)

            u_kv, s_kv, vh_kv = torch.linalg.svd(w_kv, full_matrices=False)

            r = min(
                self.rank,
                s_kv.shape[0],
                self.kv_down.weight.shape[0],
                self.k_up.weight.shape[1],
                self.v_up.weight.shape[1],
            )

            # MLA-FIX-01: NO norm_scale. The SVD low-rank approximation
            # naturally has the correct norm: ||U_r Σ_r V_r^T||_F = ||Σ_r||_F.
            self.kv_down.weight.data.zero_()
            self.k_up.weight.data.zero_()
            self.v_up.weight.data.zero_()

            # kv_down: (rank, hidden_size) = vh_kv[:r, :]
            self.kv_down.weight.data[:r, :].copy_(
                vh_kv[:r, :].to(self.kv_down.weight.device, self.kv_down.weight.dtype)
            )

            # up: (2 * kv_dim, r) = U[:, :r] * Σ[:r]
            up = (u_kv[:, :r] * s_kv[:r].unsqueeze(0)).to(
                self.k_up.weight.device, self.k_up.weight.dtype
            )

            d_k = min(self.kv_dim, self.k_up.weight.shape[0])
            d_v = min(self.kv_dim, self.v_up.weight.shape[0])
            # K rows come first, V rows next
            self.k_up.weight.data[:d_k, :r].copy_(up[:d_k, :r])
            self.v_up.weight.data[:d_v, :r].copy_(up[self.kv_dim : self.kv_dim + d_v, :r])

            self.kv_down.weight.requires_grad = False
            self.k_up.weight.requires_grad = False
            self.v_up.weight.requires_grad = False

            # MLA-FIX-04: reconstruction error validation
            W_recon = torch.cat(
                [
                    self.k_up.weight.data.float() @ self.kv_down.weight.data.float(),
                    self.v_up.weight.data.float() @ self.kv_down.weight.data.float(),
                ],
                dim=0,
            )
            num = torch.norm(w_kv - W_recon, p="fro").item()
            den = torch.norm(w_kv, p="fro").item() + 1e-12
            self._reconstruction_error_kv = num / den

            if self._reconstruction_error_kv > self.max_error:
                logger.warning(
                    "MLAAttention: K/V reconstruction error %.4f > max_error %.4f. "
                    "Module marked for fallback.",
                    self._reconstruction_error_kv,
                    self.max_error,
                )
            else:
                logger.info(
                    "MLAAttention: K/V reconstruction error %.4f (max allowed %.4f).",
                    self._reconstruction_error_kv,
                    self.max_error,
                )

        # ---- storage accounting (compressed) ----
        try:
            for m in (self.q_proj, self.kv_down, self.k_up, self.v_up, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        # ---- is_decoder / nested override ----
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
        elif (
            cls_name in ("GPT2Attention", "GPT2Block")
            or hasattr(src, "c_attn")
            or "gpt" in cls_name.lower()
        ):
            self.is_decoder = True

        # Record source id for idempotency
        self._source_attn_id = id(src)

    # ------------------------------------------------------------------
    def is_compatible(self) -> bool:
        """Return True if reconstruction error is within tolerance."""
        if self._reconstruction_error_kv is None:
            return True
        return self._reconstruction_error_kv <= self.max_error

    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            return p.numel() * p.element_size() if isinstance(p, torch.Tensor) else 0

        return {
            "orig": self._orig_storage_bytes,
            "comp": self._comp_storage_bytes,
            "kv_reconstruction_error": self._reconstruction_error_kv,
        }

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
        B, N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # ---- projections ----
        q = (
            self.q_proj(hidden_states)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k_raw = triton_dispatch(self.triton_kernel, kv_input, self.k_up.weight, self.kv_down.weight)
        v_raw = triton_dispatch(self.triton_kernel, kv_input, self.v_up.weight, self.kv_down.weight)
        k = k_raw.view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v_raw.view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # ---- RoPE ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- GQA expansion ----
        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = 1.0 / math.sqrt(self.head_dim)
        eff_mask = encoder_attention_mask if is_cross else attention_mask

        # ---- SDPA fast path ----
        if eff_mask is None and head_mask is None and not output_attentions:
            try:
                _qs = q.half() if q.dtype == torch.float32 else q
                _ks = k.half() if k.dtype == torch.float32 else k
                _vs = v.half() if v.dtype == torch.float32 else v
                _is_causal = bool(self.is_decoder and not is_cross and N > 1)
                out = F.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                out = out.to(dtype).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                if self.is_nested_encoder:
                    return out, None
                return self.out_proj(out), None
            except Exception:
                pass

        # ---- Manual path ----
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        if self.is_decoder and not is_cross and N > 1:
            diag = 1 + max(0, kv_len - N)
            causal = (
                torch.triu(
                    torch.full((N, kv_len), float("-inf"), device=device, dtype=torch.float32),
                    diagonal=diag,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal

        if eff_mask is not None:
            if eff_mask.dim() == 2:
                scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                scores = scores + eff_mask.float()

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

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        out = context if self.is_nested_encoder else self.out_proj(context)
        return out, (attn if output_attentions else None)


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("mla_inject")
@ATTENTION_REGISTRY.register("inject_mla")
def inject_mla(
    model: nn.Module | None = None,
    layer: nn.Module | None = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 1,
    old_attn: Any = None,
    _depth: int = 0,
    **kwargs: Any,
) -> MLAAttention | nn.Module | None:
    if _depth > 8:
        logger.warning("inject_mla: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None)

    # ---- Whole-model traversal ----
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_mla(
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

    # ---- Single-layer ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attn", "attention", "self_attn", "self"):
            candidate = getattr(target, attr, None)
            if candidate is None:
                continue
            if isinstance(candidate, MLAAttention) or "MLA" in candidate.__class__.__name__:
                return None  # idempotency
            if hasattr(candidate, "self") and hasattr(candidate.self, "query"):
                src_attn = candidate.self
            else:
                src_attn = candidate
            break

    if src_attn is None:
        logger.warning("inject_mla: no source attention on %s", type(target).__name__)
        return None

    rank = getattr(comp_config, "mla_rank", 128) if comp_config is not None else 128
    max_err = float(
        kwargs.get(
            "max_error",
            getattr(comp_config, "mla_max_error", 0.35) if comp_config else 0.35,
        )
    )
    fallback = bool(
        kwargs.get(
            "fallback_to_original",
            getattr(comp_config, "fallback_to_original", True) if comp_config else True,
        )
    )

    try:
        new_attn = MLAAttention(
            cfg,
            rank=rank,
            max_error=max_err,
            fallback_to_original=fallback,
        )
        new_attn.copy_weights(src_attn)
    except Exception as e:
        logger.error("inject_mla construction/copy failed: %s", e, exc_info=True)
        return None

    # MLA-FIX-03: fallback if reconstruction error is too high
    if fallback and not new_attn.is_compatible():
        logger.warning(
            "inject_mla: MLA rejected on %s (err=%.4f > %.4f); keeping original.",
            type(target).__name__,
            new_attn._reconstruction_error_kv or 0.0,
            new_attn.max_error,
        )
        return None

    p = next(src_attn.parameters(), None)
    if p is not None:
        new_attn.to(device=p.device, dtype=p.dtype)

    # ---- Splice ----
    replaced = False
    if hasattr(target, "attn"):
        target.attn = new_attn
        new_attn.is_nested_encoder = False
        replaced = True
    elif hasattr(target, "attention"):
        if hasattr(target.attention, "self") and hasattr(target.attention.self, "query"):
            new_attn.is_nested_encoder = True
            target.attention.self = new_attn
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn
        replaced = True
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
        replaced = True
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
        replaced = True

    if not replaced:
        logger.error("inject_mla: no writable attribute on %s", type(target).__name__)
        return None

    return new_attn


inject_mla_attention = inject_mla

__all__ = [
    "MLAAttention",
    "LoRALinear",
    "inject_mla",
    "inject_mla_attention",
    "apply_rotary_pos_emb",
]
