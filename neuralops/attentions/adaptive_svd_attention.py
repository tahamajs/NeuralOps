"""
Adaptive SVD Attention — Production-Hardened (v2).

Low-rank factorized attention with dynamic cumulative energy-thresholded SVD
and trainable residual adapters:

    W ≈ U_r V_r + W_lora_B · W_lora_A
    where  U_r = U_{:, :r} · √Σ_r,  V_r = √Σ_r · V^T_{:r, :}
    and    r = argmin_{r} [ Σ_{i=1}^r σ_i² / Σ_j σ_j² ≥ τ ]
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import apply_attention_mask
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.utils.compat import repeat_kv

logger = logging.getLogger(__name__)


# ===========================================================================
# Module-level RoPE helper (ASVD-FIX-05)
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    rot = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rot = torch.cat([rot, x[..., -1:]], dim=-1)
    return rot


# ===========================================================================
# Factorized Linear
# ===========================================================================
class AdaptiveSVDLinear(nn.Module):
    """
    Frozen SVD base + trainable LoRA residual.

    Forward:
        y = x V^T U^T + x lora_A^T lora_B^T + b

    Storage:
        U (out, r), V (r, in), lora_A (residual_rank, in), lora_B (out, residual_rank)
    """

    # ASVD-FIX-03: not attention
    is_custom_attention: bool = False
    is_svd_linear: bool = True
    is_compressed_svd: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int,
        residual_rank: int = 16,
        bias: bool = True,
        trainable_svd: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        max_rank = min(self.in_features, self.out_features)
        self.svd_rank = int(min(svd_rank, max_rank))
        self.residual_rank = int(min(residual_rank, max_rank))

        fk = {"dtype": dtype, "device": device}
        self.U = nn.Parameter(
            torch.empty(self.out_features, self.svd_rank, **fk),
            requires_grad=trainable_svd,
        )
        self.V = nn.Parameter(
            torch.empty(self.svd_rank, self.in_features, **fk),
            requires_grad=trainable_svd,
        )

        if self.residual_rank > 0:
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.residual_rank, **fk))
            self.lora_A = nn.Parameter(torch.empty(self.residual_rank, self.in_features, **fk))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **fk))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for p in (self.U, self.V):
                if p.dtype in (torch.float16, torch.bfloat16):
                    tmp = torch.empty_like(p, dtype=torch.float32, device="cpu")
                    nn.init.orthogonal_(tmp)
                    p.copy_(tmp.to(dtype=p.dtype, device=p.device))
                else:
                    nn.init.orthogonal_(p)
            if self.lora_A is not None and self.lora_B is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def set_svd_factors(self, U: torch.Tensor, V: torch.Tensor) -> None:
        with torch.no_grad():
            r = min(U.size(1), V.size(0), self.svd_rank)
            out_d = min(self.U.size(0), U.size(0))
            in_d = min(self.V.size(1), V.size(1))
            self.U.zero_()
            self.V.zero_()
            self.U[:out_d, :r].copy_(U[:out_d, :r].to(self.U.dtype))
            self.V[:r, :in_d].copy_(V[:r, :in_d].to(self.V.dtype))
            if self.lora_B is not None:
                nn.init.zeros_(self.lora_B)

    # ---- aliases for framework compatibility ----
    @property
    def A(self) -> nn.Parameter | None:
        return self.lora_A

    @property
    def B(self) -> nn.Parameter | None:
        return self.lora_B

    @property
    def weight(self) -> torch.Tensor:
        W = torch.matmul(self.U, self.V)
        if self.lora_A is not None and self.lora_B is not None:
            W = W + torch.matmul(self.lora_B, self.lora_A)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ASVD-FIX-09: dtype-safe
        orig_dtype = x.dtype
        V = self.V.to(x.dtype) if self.V.dtype != x.dtype else self.V
        U = self.U.to(x.dtype) if self.U.dtype != x.dtype else self.U

        out = F.linear(F.linear(x, V), U)

        if self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            b = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), b)

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(orig_dtype)


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "adaptive_svd_attn",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "adaptive_svd", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "adaptive_svd_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class AdaptiveSVDAttention(BaseAttention):
    """Multi-head attention with adaptive SVD + residual factorized projections."""

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: Any,
        svd_rank: int | None = None,
        residual_rank: int = 16,
        energy_threshold: float = 0.999,  # ASVD-FIX-02
        pretrained_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)

        # ASVD-FIX-12: always store config
        self.config = config

        # ===========================================================
        # ASVD-FIX-04: single source of truth for dims
        # ===========================================================
        def _first_attr(obj, names, default=None):
            if obj is None:
                return default
            for n in names:
                v = getattr(obj, n, None)
                if v is not None and v != 0:
                    return v
            return default

        # hidden_size
        hidden_size = (
            kwargs.get("hidden_size")
            or _first_attr(config, ("hidden_size", "d_model", "n_embd", "dim"))
            or _first_attr(pretrained_attn, ("hidden_size", "embed_dim", "all_head_size"))
        )
        if hidden_size is None:
            for attr in ("q_proj", "query", "q_lin", "c_attn"):
                mod = getattr(pretrained_attn, attr, None) if pretrained_attn else None
                if mod is not None and hasattr(mod, "in_features"):
                    hidden_size = mod.in_features
                    break
        self.hidden_size = int(hidden_size or 768)

        # num_heads
        num_heads = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or _first_attr(config, ("num_attention_heads", "num_heads", "n_heads", "n_head"))
            or _first_attr(pretrained_attn, ("num_heads", "num_attention_heads", "n_heads"))
        )
        self.num_heads = max(1, int(num_heads or 12))

        # num_kv_heads
        num_kv_heads = (
            kwargs.get("num_kv_heads")
            or kwargs.get("num_key_value_heads")
            or _first_attr(config, ("num_key_value_heads", "num_kv_heads"))
            or _first_attr(pretrained_attn, ("num_key_value_heads", "num_kv_heads"))
            or self.num_heads
        )
        self.num_kv_heads = max(1, min(int(num_kv_heads), self.num_heads))

        # head_dim
        head_dim = kwargs.get("head_dim") or _first_attr(config, ("head_dim", "d_kv"))
        if not head_dim and pretrained_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                mod = getattr(pretrained_attn, attr, None)
                if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
                    if mod.weight.ndim == 2:
                        head_dim = mod.weight.shape[0] // max(1, self.num_heads)
                        break
        self.head_dim = int(head_dim or (self.hidden_size // self.num_heads))

        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        # ===========================================================
        # ASVD-FIX-08: is_causal computed once
        # ===========================================================
        _mt = str(getattr(config, "model_type", "") or "").lower() if config else ""
        _decoder_prefixes = (
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
            "codegen",
        )
        is_causal = (
            (is_decoder is True)
            or bool(kwargs.get("is_decoder", False))
            or bool(kwargs.get("is_causal", False))
            or bool(getattr(config, "is_decoder", False))
            if config
            else False
        )
        if not is_causal and config is not None:
            is_causal = (
                bool(getattr(config, "is_causal", False))
                or bool(getattr(config, "causal", False))
                or any(_mt.startswith(p) for p in _decoder_prefixes)
            )
        if not is_causal and pretrained_attn is not None:
            is_causal = (
                hasattr(pretrained_attn, "c_attn")
                or bool(getattr(pretrained_attn, "is_decoder", False))
                or bool(getattr(pretrained_attn, "is_causal", False))
            )
        self.is_decoder = bool(is_causal)
        self.is_causal = bool(is_causal)

        # ===========================================================
        # SVD configuration
        # ===========================================================
        cfg_svd_rank = getattr(config, "svd_rank", None) if config else None
        cfg_rank = getattr(config, "rank", None) if config else None
        resolved_svd_rank = svd_rank if svd_rank is not None else (cfg_svd_rank or cfg_rank or 64)
        # Clamp to legal range
        self.svd_rank = max(1, min(int(resolved_svd_rank), self.hidden_size, self.all_head_size))

        self.residual_rank = max(0, int(residual_rank))

        # ASVD-FIX-02: consistent threshold
        cfg_et = getattr(config, "energy_threshold", None) if config else None
        self.energy_threshold = float(cfg_et if cfg_et is not None else energy_threshold)
        # Clamp to valid range so rank genuinely varies with svd_rank
        self.energy_threshold = max(0.5, min(self.energy_threshold, 0.9999))

        dropout_p = 0.1
        if config is not None:
            dropout_p = getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                ),
            )
        self.dropout = nn.Dropout(float(dropout_p))

        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None) if pretrained_attn else None

        # Relative attention bias (T5)
        self.has_relative_attention_bias = False
        self.relative_attention_bias: nn.Embedding | None = None
        self.relative_attention_num_buckets = int(
            getattr(config, "relative_attention_num_buckets", 32) if config else 32
        )
        self.relative_attention_max_distance = int(
            getattr(config, "relative_attention_max_distance", 128) if config else 128
        )

        # ===========================================================
        # Projections
        # ===========================================================
        self.query = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
        self.key = AdaptiveSVDLinear(
            self.hidden_size,
            self.kv_head_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
        self.value = AdaptiveSVDLinear(
            self.hidden_size,
            self.kv_head_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
        self.out_proj = AdaptiveSVDLinear(
            self.all_head_size,
            self.hidden_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )

        # Aliases
        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value
        self.q_lin = self.query
        self.k_lin = self.key
        self.v_lin = self.value
        self.out_lin = self.out_proj
        self.dense = self.out_proj
        self.o_proj = self.out_proj
        self.c_proj = self.out_proj

        self.is_nested_encoder: bool = False

        # ASVD-FIX-11: storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    @property
    def q_lin_alias(self):
        return self.query

    @property
    def k_lin_alias(self):
        return self.key

    @property
    def v_lin_alias(self):
        return self.value

    # ------------------------------------------------------------------
    def _relative_position_bucket(
        self, relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        if self.relative_attention_bias is None:
            return None
        ctx = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        mem = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        rel = mem - ctx
        rel_bucket = self._relative_position_bucket(
            rel,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(rel_bucket)
        return values.permute([2, 0, 1]).unsqueeze(0)

    # ------------------------------------------------------------------
    def _compute_energy_rank(self, S, max_rank, min_rank=4):
        """
        ASVD-FIX-02: with `energy_threshold` at 0.999, this returns a rank
        close to `max_rank`; with a lower threshold, more aggressive truncation.
        The actual return is always clamped to `[min_rank, max_rank]`.
        """
        s_sq = S**2
        total = s_sq.sum()
        if total <= 1e-12:
            return max(1, min(min_rank, max_rank))
        cum = torch.cumsum(s_sq, dim=0) / total
        idx = torch.where(cum >= self.energy_threshold)[0]
        r = int(idx[0].item()) + 1 if len(idx) > 0 else max_rank
        return max(1, min(max(r, min_rank), max_rank))

    def _init_svd_part(self, layer: AdaptiveSVDLinear, W_data, bias_data):
        with torch.no_grad():
            W = W_data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            max_r = min(layer.svd_rank, U.size(1), Vt.size(0))
            r = self._compute_energy_rank(S, max_rank=max_r)

            sqrt_s = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_s.unsqueeze(0)).to(layer.U.dtype)
            V_r = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(layer.V.dtype)

            layer.set_svd_factors(U_r, V_r)
            if layer.bias is not None and bias_data is not None:
                n = min(layer.bias.numel(), bias_data.numel())
                layer.bias.data[:n].copy_(bias_data[:n].to(layer.bias.dtype))

    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted = {
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
                pass

        # ---- GPT-2 fused c_attn ----
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            c = src.c_attn
            base_c = getattr(c, "base_layer", c)
            wt = getattr(base_c, "weight", getattr(c, "weight", None))
            if wt is not None:
                wt = wt.data.clone()
                is_conv = (
                    "Conv1D" in c.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c, "nf")
                    or hasattr(base_c, "nf")
                    or (wt.dim() == 2 and wt.shape[0] < wt.shape[1])
                )
                if is_conv or (wt.dim() == 2 and wt.shape[1] == 3 * self.hidden_size):
                    wt = wt.t().contiguous()
                b = getattr(base_c, "bias", getattr(c, "bias", None))
                bd = b.data.clone() if b is not None else None
                split = wt.shape[0] // 3
                extracted["q"] = (wt[:split], bd[:split] if bd is not None else None)
                extracted["k"] = (
                    wt[split : 2 * split],
                    bd[split : 2 * split] if bd is not None else None,
                )
                extracted["v"] = (
                    wt[2 * split :],
                    bd[2 * split :] if bd is not None else None,
                )

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            o = src.c_proj
            base_o = getattr(o, "base_layer", o)
            wo = getattr(base_o, "weight", getattr(o, "weight", None))
            if wo is not None:
                wo = wo.data.clone()
                is_conv_o = (
                    "Conv1D" in o.__class__.__name__
                    or "Conv1D" in base_o.__class__.__name__
                    or hasattr(o, "nf")
                    or hasattr(base_o, "nf")
                    or (
                        wo.dim() == 2
                        and wo.shape[0] != self.hidden_size
                        and wo.shape[1] == self.hidden_size
                    )
                    or (wo.dim() == 2 and wo.shape[0] < wo.shape[1])
                )
                if is_conv_o:
                    wo = wo.t().contiguous()
                bo = getattr(base_o, "bias", getattr(o, "bias", None))
                extracted["o"] = (wo, bo.data.clone() if bo is not None else None)

        # ---- BERT / RoBERTa nested ----
        if hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                if extracted[key][0] is None and hasattr(s, sub):
                    m = getattr(s, sub)
                    b = getattr(m, "bias", None)
                    extracted[key] = (m.weight.data, b.data if b is not None else None)

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            d = getattr(src.output.dense, "base_layer", src.output.dense)
            b = getattr(d, "bias", None)
            extracted["o"] = (d.weight.data, b.data if b is not None else None)

        # ---- flat modules ----
        named_map = {
            "q": ("q_proj", "q_lin", "query"),
            "k": ("k_proj", "k_lin", "key"),
            "v": ("v_proj", "v_lin", "value"),
            "o": ("o_proj", "out_proj", "out_lin", "dense"),
        }
        for key, names in named_map.items():
            if extracted[key][0] is None:
                for n in names:
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (
                                m.weight.data,
                                b.data if b is not None else None,
                            )
                            break

        # ---- nested encoder detection ----
        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, a) for a in ("dense", "out_proj", "c_proj", "out_lin", "o_proj")
        )
        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
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
                or (cls_name.endswith("SelfAttention") and not has_out_proj)
            )
            and not has_out_proj
        ):
            is_nested = True
        self.is_nested_encoder = is_nested

        # ---- tracking original bytes ----
        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        # ---- validation ----
        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AdaptiveSVDAttention.copy_weights: missing {missing} from {cls_name}."
            )

        for k_ in ("q", "k", "v"):
            w = extracted[k_][0]
            if w is not None:
                n = torch.linalg.norm(w.float()).item()
                if n < 1e-4:
                    logger.warning("AdaptiveSVDAttention: %s near-zero norm (%.2e)", k_, n)

        # ---- factorize ----
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
                        # identity
                        eye = torch.eye(
                            target.out_features,
                            target.in_features,
                            device=target.U.device,
                            dtype=target.U.dtype,
                        )
                        self._init_svd_part(target, eye, None)
                    continue
                self._init_svd_part(target, w, b)

                # ASVD-FIX-10: freeze all base params + bias
                target.U.requires_grad = False
                target.V.requires_grad = False
                if target.bias is not None:
                    target.bias.requires_grad = False
                if target.lora_A is not None:
                    target.lora_A.requires_grad = True
                if target.lora_B is not None:
                    target.lora_B.requires_grad = True

        # ---- track compressed bytes ----
        try:
            for m in (self.query, self.key, self.value, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        # ---- inherit auxiliary state from source ----
        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if hasattr(src, "relative_attention_bias") and src.relative_attention_bias is not None:
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
            self.is_causal = self.is_decoder

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
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # ASVD-FIX-06: always return (output, attn_weights)
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross = key_value_states is not None
        kv_input = key_value_states if is_cross else hidden_states

        B, N_q, _ = hidden_states.shape
        N_k = kv_input.shape[1]
        H, D = self.num_heads, self.head_dim

        Q = self.query(hidden_states).view(B, N_q, H, D).transpose(1, 2).contiguous()
        K = self.key(kv_input).view(B, N_k, self.num_kv_heads, D).transpose(1, 2).contiguous()
        V = self.value(kv_input).view(B, N_k, self.num_kv_heads, D).transpose(1, 2).contiguous()

        # ---- RoPE (ASVD-FIX-05: uses module-level helper) ----
        position_embeddings = kwargs.get("position_embeddings")
        if (
            position_embeddings is not None
            and isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) >= 2
        ):
            cos, sin = position_embeddings[0], position_embeddings[1]
            if cos.dim() == 3:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            while cos.dim() < Q.dim():
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            if cos.shape[-2] != Q.shape[-2]:
                cos, sin = cos[..., : Q.shape[-2], :], sin[..., : Q.shape[-2], :]
            rotary_dim = cos.shape[-1]
            if rotary_dim <= Q.shape[-1]:
                q_rot, q_pass = Q[..., :rotary_dim], Q[..., rotary_dim:]
                k_rot, k_pass = K[..., :rotary_dim], K[..., rotary_dim:]
                q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
                k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
                Q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
                K = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
            else:
                cos_cut = cos[..., : Q.shape[-1]].to(Q.dtype)
                sin_cut = sin[..., : Q.shape[-1]].to(Q.dtype)
                Q = (Q * cos_cut) + (_rotate_half(Q) * sin_cut)
                cos_k = cos[..., : K.shape[-1]].to(K.dtype)
                sin_k = sin[..., : K.shape[-1]].to(K.dtype)
                K = (K * cos_k) + (_rotate_half(K) * sin_k)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = D**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale

        # ---- Relative attention bias (T5) ----
        position_bias = kwargs.get("position_bias")
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(N_q, N_k, device=hidden_states.device)

        mask = attention_mask if attention_mask is not None else kwargs.get("mask")

        if position_bias is not None:
            if mask is not None and mask.dim() == 4:
                position_bias = position_bias + mask.to(scores.dtype)
            scores = scores + position_bias.to(scores.dtype)
        elif mask is not None:
            scores = apply_attention_mask(scores, mask)

        # ---- Causal mask (ASVD-FIX-08: use self.is_causal) ----
        if self.is_causal and not is_cross and N_q > 1:
            causal_diag = 1 + max(0, N_k - N_q)
            causal = (
                torch.triu(
                    torch.full(
                        (N_q, N_k),
                        float("-inf"),
                        device=scores.device,
                        dtype=scores.dtype,
                    ),
                    diagonal=causal_diag,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_weights = F.softmax(scores.float(), dim=-1).to(hidden_states.dtype)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            hm = head_mask.to(attn_weights.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            attn_weights = attn_weights * hm

        context = torch.matmul(attn_weights, V.to(attn_weights.dtype))
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        output = output.to(hidden_states.dtype)
        return output, (attn_weights if output_attentions else None)

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("out_proj", self.out_proj),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("adaptive_svd_attention_inject")
@ATTENTION_REGISTRY.register("adaptive_svd_inject")
@ATTENTION_REGISTRY.register("inject_adaptive_svd")
def inject_adaptive_svd(
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
        logger.warning("inject_adaptive_svd: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

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
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_adaptive_svd(
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

    # ---- Single-layer injection ----
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

    # ---- ASVD-FIX-07: idempotency guard ----
    if isinstance(src_attn, AdaptiveSVDAttention) or "AdaptiveSVD" in src_attn.__class__.__name__:
        return None

    # ===========================================================
    # ASVD-FIX-01 / ASVD-FIX-13: priority for rank
    #   1. explicit `kwargs["rank"]` (CLI)
    #   2. `comp_config.svd_rank` / `comp_config.rank`
    #   3. `model_config.svd_rank` / `model_config.rank`
    #   4. fallback: min(64, head_dim)
    # ===========================================================
    explicit_rank = kwargs.pop("rank", None)

    # resolve fallback head_dim/hidden_size
    _h_size = (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "d_model", None)
        or getattr(cfg, "n_embd", None)
        or getattr(cfg, "dim", None)
        or kwargs.get("hidden_size", 768)
    )
    _n_heads = (
        getattr(cfg, "num_attention_heads", None)
        or getattr(cfg, "num_heads", None)
        or getattr(cfg, "n_heads", None)
        or getattr(src_attn, "num_heads", None)
        or kwargs.get("num_heads", 12)
    )
    _head_dim = getattr(cfg, "head_dim", None) or (_h_size // max(1, _n_heads))
    _default_svd_rank = max(1, min(64, _head_dim))

    if explicit_rank is not None:
        svd_rank = int(explicit_rank)
    else:
        svd_rank = (
            getattr(comp_config, "svd_rank", None)
            or getattr(comp_config, "rank", None)
            or getattr(cfg, "svd_rank", None)
            or getattr(cfg, "rank", None)
            or _default_svd_rank
        )
        svd_rank = int(svd_rank)

    # Residual rank
    residual_rank = (
        getattr(comp_config, "residual_rank", None)
        or getattr(cfg, "residual_rank", None)
        or kwargs.pop("residual_rank", 16)
    )
    residual_rank = int(residual_rank)

    # Energy threshold — ASVD-FIX-02: default 0.999, not 0.92
    energy_threshold = float(
        getattr(comp_config, "energy_threshold", None)
        or getattr(cfg, "energy_threshold", None)
        or kwargs.pop("energy_threshold", 0.999)
    )

    # ---- Config for building the module ----
    build_cfg = cfg
    if build_cfg is None:
        build_cfg = type(
            "Config",
            (),
            {
                "hidden_size": int(_h_size),
                "num_attention_heads": int(_n_heads),
                "num_heads": int(_n_heads),
                "head_dim": int(_head_dim),
                "num_key_value_heads": int(_n_heads),
                "svd_rank": int(svd_rank),
                "rank": int(svd_rank),
                "residual_rank": int(residual_rank),
                "energy_threshold": float(energy_threshold),
                "dropout": 0.1,
            },
        )()

    # ---- Instantiate ----
    try:
        new_attn = AdaptiveSVDAttention(
            config=build_cfg,
            svd_rank=svd_rank,
            residual_rank=residual_rank,
            energy_threshold=energy_threshold,
            pretrained_attn=src_attn if src_attn is not target else None,
            is_decoder=kwargs.pop("is_decoder", None),
            **kwargs,
        )
    except Exception as e:
        logger.error("AdaptiveSVDAttention construction failed: %s", e)
        return None

    # ---- Move to source device/dtype ----
    if src_attn is not None and src_attn is not target:
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

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
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
        spliced = True
    elif hasattr(target, "attn"):
        new_attn.is_nested_encoder = False
        target.attn = new_attn
        spliced = True
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
        spliced = True

    if not spliced:
        logger.error(
            "inject_adaptive_svd: no writable attention attribute on %s",
            target.__class__.__name__,
        )
        return None

    return new_attn


__all__ = [
    "AdaptiveSVDLinear",
    "AdaptiveSVDAttention",
    "inject_adaptive_svd",
]
