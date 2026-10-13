r"""
Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.

Architecture:
    * Factorized SVD projections for Q, K, V (down + up with LoRA adapters)
    * Full KV-cache support for autoregressive generation
    * RoPE with past_len slicing
    * GQA support with repeat_kv
    * Multi-architecture weight transfer (DistilBERT, RoBERTa, BERT, GPT-2, Qwen, LLaMA)

"""

from __future__ import annotations
from neuralops.core.common import FactorizedLinear
from neuralops.core.common import _cfg_get
from neuralops.core.common import _read_past_kv

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
from neuralops.core.amp_utils import maybe_half
from neuralops.core.attention_mask import apply_attention_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)




def _write_past_kv(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    past_key_value: Any | None,
    layer_idx: int,
) -> Any:
    """Update cache with new K/V entries; return updated container."""
    if past_key_value is not None and hasattr(past_key_value, "update"):
        try:
            past_key_value.update(k_new, v_new, layer_idx)
            return past_key_value
        except Exception:
            pass

    # Concatenate to existing
    if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
        k_prev, v_prev = past_key_value[0], past_key_value[1]
        if isinstance(k_prev, torch.Tensor):
            k = torch.cat([k_prev.to(k_new.dtype), k_new], dim=-2)
            v = torch.cat([v_prev.to(v_new.dtype), v_new], dim=-2)
            return (k, v)

    return (k_new, v_new)


# ===========================================================================
# FactorizedLoRAUp
# ===========================================================================
class FactorizedLoRAUp(nn.Module):
    """Up-projection with frozen base weight and trainable LoRA adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # Freeze base by default
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if (
            self.lora_rank > 0
            and self.lora_A is not None
            and self.lora_B is not None
        ):
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out


# ===========================================================================
# SOAAttention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "soa_attn",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class SOAAttention(BaseAttention):
    r"""
    Shared Orthogonal Attention with per-projection latent bottlenecks.

    Factorizes dense Q, K, V projections into low-rank bottleneck pairs
    initialized via truncated SVD from pretrained transformer weights.
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        model_config: Any | None = None,
        rank: int | None = None,
        k_rank: int = 64,
        dropout: float = 0.1,
        pretrained_attn: Any | None = None,
        old_attn: Any | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or model_config or kwargs.get("config")
        src_attn = (
            pretrained_attn
            if pretrained_attn is not None
            else (old_attn if old_attn is not None else kwargs.get("source_attn"))
        )

        try:
            super().__init__(cfg, rank=k_rank, **kwargs)
        except TypeError:
            super().__init__(cfg)

        # ---- Layer index for cache routing ----
        self.layer_idx = int(
            layer_idx if layer_idx is not None else kwargs.get("layer_idx", 0)
        )

        # ---- Dimensions ----
        self.hidden_size: int = (
            getattr(
                cfg,
                "hidden_size",
                getattr(
                    cfg,
                    "dim",
                    kwargs.get("hidden_size", kwargs.get("dim", 768)),
                ),
            )
            if cfg is not None
            else kwargs.get("hidden_size", kwargs.get("dim", 768))
        )
        if hasattr(self.hidden_size, "hidden_size"):
            self.hidden_size = self.hidden_size.hidden_size

        self.num_heads: int = int(
            getattr(
                cfg,
                "num_attention_heads",
                getattr(
                    cfg,
                    "n_heads",
                    getattr(
                        cfg,
                        "num_heads",
                        kwargs.get(
                            "num_attention_heads", kwargs.get("num_heads", 12)
                        ),
                    ),
                ),
            )
            if cfg is not None
            else kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))
        ) or 12

        # ---- head_dim ----
        head_dim = kwargs.get("head_dim")
        if not head_dim and src_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                mod = getattr(src_attn, attr, None)
                if (
                    mod is not None
                    and hasattr(mod, "weight")
                    and mod.weight is not None
                    and mod.weight.ndim == 2
                ):
                    head_dim = mod.weight.shape[0] // max(1, self.num_heads)
                    break
            if not head_dim:
                head_dim = getattr(src_attn, "head_dim", None)
        if not head_dim and cfg is not None:
            head_dim = getattr(cfg, "head_dim", None)
        if not head_dim or head_dim <= 0:
            head_dim = self.hidden_size // max(1, self.num_heads)
        self.head_dim = int(head_dim)

        # ---- GQA ----
        cfg_kv = (
            getattr(
                cfg,
                "num_key_value_heads",
                getattr(
                    cfg,
                    "num_kv_heads",
                    kwargs.get(
                        "num_key_value_heads", kwargs.get("num_kv_heads")
                    ),
                ),
            )
            if cfg is not None
            else kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads"))
        )
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.num_key_value_groups: int = max(
            1, self.num_heads // max(1, self.num_kv_heads)
        )
        self.kv_repeat: int = self.num_key_value_groups

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.all_head_size: int = self.q_head_size

        # ---- Factorized rank ----
        comp_config = getattr(cfg, "compression_config", None)
        if k_rank is None or k_rank == 64:
            effective_rank = getattr(
                comp_config,
                "k_rank",
                getattr(
                    comp_config,
                    "soa_rank",
                    rank if rank is not None else max(128, self.hidden_size // 4),
                ),
            )
        else:
            effective_rank = min(k_rank, self.hidden_size)
        self.k_rank: int = int(effective_rank)
        self.rank = self.k_rank
        self.lora_rank: int = int(lora_rank)
        self.lora_alpha: float = float(lora_alpha)

        # ---- Decoder detection ----
        _mt = str(getattr(cfg, "model_type", "") or "").lower() if cfg else ""
        _decoder_prefixes = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )
        is_dec = (
            bool(_cfg_get(cfg, "is_decoder", False))
            or bool(_cfg_get(cfg, "is_causal", False))
            or any(_mt.startswith(p) for p in _decoder_prefixes)
            or (src_attn is not None and hasattr(src_attn, "c_attn"))
            or (src_attn is not None and getattr(src_attn, "is_causal", False))
            or (src_attn is not None and getattr(src_attn, "is_decoder", False))
            or bool(kwargs.get("is_causal", False))
            or bool(kwargs.get("is_decoder", False))
        )
        if is_decoder is not None:
            is_dec = bool(is_decoder)
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = self.is_decoder
        self.is_nested_encoder: bool = False

        # ---- Factorized SVD projections for Q, K, V ----
        self.q_down = nn.Linear(self.hidden_size, self.k_rank, bias=False)
        self.q_up = FactorizedLoRAUp(
            self.k_rank,
            self.q_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )

        self.k_down = nn.Linear(self.hidden_size, self.k_rank, bias=False)
        self.k_up = FactorizedLoRAUp(
            self.k_rank,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )

        self.v_down = nn.Linear(self.hidden_size, self.k_rank, bias=False)
        self.v_up = FactorizedLoRAUp(
            self.k_rank,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )

        # ---- Output projection ----
        self.out_proj: nn.Module = LoRALinear(
            self.q_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        # ---- Dropout ----
        dropout_p = (
            getattr(
                cfg,
                "dropout",
                getattr(cfg, "attention_probs_dropout_prob", dropout),
            )
            if cfg is not None
            else dropout
        )
        self.dropout = nn.Dropout(dropout_p)

        # ---- Target dtype ----
        target_dtype = kwargs.get("dtype")
        if target_dtype is None and src_attn is not None:
            try:
                p = next(src_attn.parameters())
                target_dtype = p.dtype
            except StopIteration:
                target_dtype = torch.float32

        # ---- Weight transfer ----
        if src_attn is not None:
            self.copy_weights(src_attn)

        if target_dtype is not None and isinstance(target_dtype, torch.dtype):
            self.to(target_dtype)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    @property
    def q_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.q_down, self.q_up)

    @property
    def k_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.k_down, self.k_up)

    @property
    def v_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.v_down, self.v_up)

    @property
    def query(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def q_lin(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def k_lin(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def v_lin(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def out_lin(self) -> nn.Module:
        return self.out_proj

    @property
    def dense(self) -> nn.Module:
        return self.out_proj

    @property
    def o_proj(self) -> nn.Module:
        return self.out_proj

    @property
    def c_proj(self) -> nn.Module:
        return self.out_proj

    # ------------------------------------------------------------------
    # Weight transfer (multi-architecture)
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: nn.Module) -> None:
        """Extract Q/K/V/O weights and SVD-factorize them into down/up pairs."""
        with torch.no_grad():
            w_q = b_q = None
            w_k = b_k = None
            w_v = b_v = None
            w_o = b_o = None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            has_out_lin = (
                hasattr(old_attn, "out_lin")
                or hasattr(old_attn, "c_proj")
                or hasattr(old_attn, "o_proj")
                or hasattr(old_attn, "dense")
            )

            # ---- Detect nested encoder ----
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query") and not has_out_lin
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    dense = getattr(old_attn.output.dense, "base_layer", old_attn.output.dense)
                    w_o = dense.weight.detach()
                    b_o = (
                        dense.bias.detach() if dense.bias is not None else None
                    )
            else:
                src = old_attn

            # ---- Pattern 1: GPT-2 fused c_attn (Conv1D) ----
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                self.is_decoder = True
                self.is_causal = True

                c_attn_mod = src.c_attn
                base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
                c_w = getattr(
                    base_c, "weight", getattr(c_attn_mod, "weight", None)
                ).detach()
                c_b = getattr(
                    base_c, "bias", getattr(c_attn_mod, "bias", None)
                )
                c_b = c_b.detach() if c_b is not None else None

                # Conv1D transpose
                is_conv = (
                    "Conv1D" in base_c.__class__.__name__
                    or "Conv1D" in c_attn_mod.__class__.__name__
                    or hasattr(base_c, "nf")
                    or hasattr(c_attn_mod, "nf")
                    or (c_w.dim() == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv:
                    c_w = c_w.t().contiguous()

                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size

                w_q = c_w[:q_dim, :]
                w_k = c_w[q_dim : q_dim + k_dim, :]
                w_v = c_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]

                if c_b is not None:
                    b_q = c_b[:q_dim]
                    b_k = c_b[q_dim : q_dim + k_dim]
                    b_v = c_b[q_dim + k_dim : q_dim + k_dim + v_dim]

                # Output
                c_proj_mod = src.c_proj
                base_p = getattr(c_proj_mod, "base_layer", c_proj_mod)
                out_w = getattr(
                    base_p, "weight", getattr(c_proj_mod, "weight", None)
                ).detach()
                b_o = getattr(
                    base_p, "bias", getattr(c_proj_mod, "bias", None)
                )
                b_o = b_o.detach() if b_o is not None else None

                is_conv_p = (
                    "Conv1D" in base_p.__class__.__name__
                    or "Conv1D" in c_proj_mod.__class__.__name__
                    or hasattr(base_p, "nf")
                    or hasattr(c_proj_mod, "nf")
                    or (out_w.dim() == 2 and out_w.shape[0] < out_w.shape[1])
                )
                if is_conv_p:
                    out_w = out_w.t().contiguous()
                w_o = out_w

            # ---- Pattern 2: BERT / RoBERTa nested ----
            elif (
                hasattr(src, "query")
                and hasattr(src, "key")
                and hasattr(src, "value")
            ):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            w_o = proj.weight.detach()
                            b_o = (
                                proj.bias.detach()
                                if proj.bias is not None
                                else None
                            )
                            break

            # ---- Pattern 3: LLaMA / Qwen ----
            elif (
                hasattr(src, "q_proj")
                and hasattr(src, "k_proj")
                and hasattr(src, "v_proj")
            ):
                w_q = src.q_proj.weight.detach()
                b_q = (
                    src.q_proj.bias.detach()
                    if src.q_proj.bias is not None
                    else None
                )
                w_k = src.k_proj.weight.detach()
                b_k = (
                    src.k_proj.bias.detach()
                    if src.k_proj.bias is not None
                    else None
                )
                w_v = src.v_proj.weight.detach()
                b_v = (
                    src.v_proj.bias.detach()
                    if src.v_proj.bias is not None
                    else None
                )

                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = (
                        src.o_proj.bias.detach()
                        if src.o_proj.bias is not None
                        else None
                    )

            # ---- Pattern 4: DistilBERT ----
            elif (
                hasattr(src, "q_lin")
                and hasattr(src, "k_lin")
                and hasattr(src, "v_lin")
            ):
                w_q = src.q_lin.weight.detach()
                b_q = (
                    src.q_lin.bias.detach()
                    if src.q_lin.bias is not None
                    else None
                )
                w_k = src.k_lin.weight.detach()
                b_k = (
                    src.k_lin.bias.detach()
                    if src.k_lin.bias is not None
                    else None
                )
                w_v = src.v_lin.weight.detach()
                b_v = (
                    src.v_lin.bias.detach()
                    if src.v_lin.bias is not None
                    else None
                )

                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = (
                        src.out_lin.bias.detach()
                        if src.out_lin.bias is not None
                        else None
                    )

            # ---- Pattern 5: UniversalAdapter fallback ----
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    w_q, b_q = adapter.get_weight_bias(src, "q")
                    w_k, b_k = adapter.get_weight_bias(src, "k")
                    w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # ---- Validation ----
            missing = []
            if w_q is None:
                missing.append("Q")
            if w_k is None:
                missing.append("K")
            if w_v is None:
                missing.append("V")
            if w_o is None and not self.is_nested_encoder:
                missing.append("Out")
            if missing:
                raise RuntimeError(
                    f"SOAAttention: Failed to extract weights {missing} "
                    f"from {type(old_attn).__name__}"
                )

            # ---- SVD Factorization ----
            def _factorize_svd(
                W: torch.Tensor,
                down: nn.Linear,
                up: FactorizedLoRAUp,
                bias: torch.Tensor | None,
                name: str,
            ) -> None:
                W_f = W.float()
                try:
                    U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
                except Exception as e:
                    logger.warning("SVD failed for %s: %s", name, e)
                    return

                S = S.clamp(min=1e-6)
                r = min(self.k_rank, U.size(1), Vt.size(0))

                down.weight.zero_()
                up.weight.zero_()
                sqrt_s = torch.sqrt(S[:r])

                down.weight.data[:r, :].copy_(
                    (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(down.weight.dtype)
                )
                up.weight.data[:, :r].copy_(
                    (U[:, :r] * sqrt_s).to(up.weight.dtype)
                )

                if bias is not None and up.bias is not None:
                    n = min(up.bias.numel(), bias.numel())
                    up.bias.data[:n].copy_(bias[:n].to(up.bias.dtype))
                elif up.bias is not None:
                    up.bias.zero_()

                # Frobenius verification
                rec = up.weight @ down.weight
                f_norm = float(torch.linalg.norm(rec.float()).item())
                if f_norm < 1e-4:
                    logger.warning(
                        "SOAAttention: reconstructed weight norm near zero for '%s' (%.2e)",
                        name,
                        f_norm,
                    )

            _factorize_svd(w_q, self.q_down, self.q_up, b_q, "q")
            _factorize_svd(w_k, self.k_down, self.k_up, b_k, "k")
            _factorize_svd(w_v, self.v_down, self.v_up, b_v, "v")

            # ---- Output projection ----
            if w_o is not None and not self.is_nested_encoder:
                if isinstance(self.out_proj, LoRALinear):
                    out_d = min(self.out_proj.weight.size(0), w_o.size(0))
                    in_d = min(self.out_proj.weight.size(1), w_o.size(1))
                    self.out_proj.weight.data[:out_d, :in_d].copy_(
                        w_o[:out_d, :in_d].to(self.out_proj.weight.dtype)
                    )
                    if (
                        b_o is not None
                        and self.out_proj.bias is not None
                    ):
                        mb = min(self.out_proj.bias.numel(), b_o.numel())
                        self.out_proj.bias.data[:mb].copy_(
                            b_o[:mb].to(self.out_proj.bias.dtype)
                        )
                    f_norm_o = float(
                        torch.linalg.norm(self.out_proj.weight.detach().float()).item()
                    )
                    if f_norm_o < 1e-4:
                        logger.warning(
                            "SOAAttention: out_proj norm near zero (%.2e)", f_norm_o
                        )
            elif self.is_nested_encoder:
                # Nested encoder bypass
                self.out_proj = nn.Identity()

            # ---- Freeze base weights ----
            for mod in (self.q_down, self.k_down, self.v_down):
                mod.weight.requires_grad = False
            for mod in (self.q_up, self.k_up, self.v_up):
                mod.freeze_base_weights()
            if hasattr(self.out_proj, "freeze_base_weights"):
                self.out_proj.freeze_base_weights()
            elif isinstance(self.out_proj, LoRALinear):
                self.out_proj.freeze_base_weights()

    @classmethod
    def from_distilbert_pretrained(
        cls,
        old_attn: nn.Module,
        model_config: Any | None = None,
        **kwargs: Any,
    ) -> "SOAAttention":
        return cls(model_config=model_config, pretrained_attn=old_attn, **kwargs)

    # ------------------------------------------------------------------
    # Forward (with full KV-cache support)
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
    ) -> tuple[torch.Tensor, Any]:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len_new = kv_input.shape[1]

        # ---- Factorized projections ----
        z_q = self.q_down(hidden_states)
        z_k = self.k_down(kv_input)
        z_v = self.v_down(kv_input)

        q = self.q_up(z_q)
        k_new = self.k_up(z_k)
        v_new = self.v_up(z_v)

        H = self.num_heads
        H_kv = self.num_kv_heads
        head_dim = self.head_dim

        q = q.view(B, N, H, head_dim).transpose(1, 2).contiguous()
        k_new = (
            k_new.view(B, kv_len_new, H_kv, head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v_new = (
            v_new.view(B, kv_len_new, H_kv, head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- [SOA-FIX-01] Read past KV cache ----
        past_k, past_v, past_len = (None, None, 0)
        if past_key_value is not None and not is_cross:
            past_k, past_v, past_len = _read_past_kv(past_key_value, self.layer_idx)

        # ---- [SOA-FIX-02] RoPE with past_len ----
        pos_emb = kwargs.get("position_embeddings")
        if (
            pos_emb is not None
            and isinstance(pos_emb, tuple)
            and len(pos_emb) == 2
            and not is_cross
        ):
            cos, sin = pos_emb
            q, k_new = apply_rotary_pos_emb(
                q, k_new, cos, sin, past_len=past_len
            )
        elif getattr(self, "rotary_emb", None) is not None and not is_cross:
            try:
                position_ids = kwargs.get("position_ids")
                if position_ids is None and past_len > 0:
                    position_ids = (
                        torch.arange(
                            past_len, past_len + N, device=hidden_states.device
                        )
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
                cos, sin = self.rotary_emb(
                    k_new, seq_len=past_len + kv_len_new
                )
                q, k_new = apply_rotary_pos_emb(
                    q, k_new, cos, sin, past_len=past_len
                )
            except Exception as e:
                logger.debug("SOAAttention rotary_emb failed: %s", e)

        # ---- [SOA-FIX-01] Concatenate with past cache ----
        if past_k is not None and past_v is not None and not is_cross:
            k = torch.cat([past_k.to(k_new.dtype), k_new], dim=-2)
            v = torch.cat([past_v.to(v_new.dtype), v_new], dim=-2)
        else:
            k, v = k_new, v_new

        N_kv = k.shape[-2]

        # ---- [SOA-FIX-01] Prepare new cache ----
        new_past_kv: Any = None
        use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        if use_cache and not is_cross:
            new_past_kv = _write_past_kv(
                k_new, v_new, past_key_value, self.layer_idx
            )

        # ---- GQA expansion ----
        if H_kv != H:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        scale = 1.0 / math.sqrt(head_dim)
        is_causal = self.is_decoder and not is_cross and N > 1

        # ---- [SOA-FIX-08] Mask normalization ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        key_padding_mask: torch.Tensor | None = None
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                if eff_mask.dtype == torch.bool:
                    key_padding_mask = ~eff_mask  # True = padding
                elif eff_mask.dtype.is_floating_point:
                    key_padding_mask = eff_mask <= -1e4
                else:
                    key_padding_mask = eff_mask == 0
            elif eff_mask.dim() == 4:
                if eff_mask.dtype == torch.bool:
                    key_padding_mask = ~eff_mask[:, 0, 0, :]
                elif eff_mask.dtype.is_floating_point:
                    key_padding_mask = eff_mask[:, 0, 0, :] <= -1e4
                else:
                    key_padding_mask = eff_mask[:, 0, 0, :] == 0

        attn_weights: torch.Tensor | None = None
        context: torch.Tensor | None = None

        # ---- [SOA-FIX-03, SOA-FIX-05] Dispatch: SDPA vs Manual ----
        use_sdpa = (not output_attentions) and (head_mask is None)

        if use_sdpa:
            try:
                # Build explicit additive bias (SDPA cannot mix is_causal + attn_mask
                # consistently across PyTorch versions)
                attn_bias = None

                if is_causal or (key_padding_mask is not None):
                    # Build a dense float mask
                    bias_val = _safe_mask_value(q.dtype)
                    attn_bias = torch.zeros(
                        (B, 1, N, N_kv), device=q.device, dtype=q.dtype
                    )

                    if is_causal:
                        # Causal with past_len offset
                        q_pos = (
                            past_len + torch.arange(N, device=q.device)
                        ).unsqueeze(1)
                        k_pos = torch.arange(
                            N_kv, device=q.device
                        ).unsqueeze(0)
                        causal_bool = k_pos <= q_pos  # (N, N_kv)
                        attn_bias = attn_bias.masked_fill(
                            ~causal_bool.unsqueeze(0).unsqueeze(0), bias_val
                        )

                    if key_padding_mask is not None:
                        attn_bias = attn_bias.masked_fill(
                            key_padding_mask.unsqueeze(1).unsqueeze(2),
                            bias_val,
                        )

                _qs = maybe_half(q)
                _ks = maybe_half(k)
                _vs = maybe_half(v)

                _sdpa_out = F.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    attn_mask=attn_bias,
                    is_causal=False,  # We build the mask explicitly
                    dropout_p=self.dropout.p if self.training else 0.0,
                )

                context = (
                    _sdpa_out.to(hidden_states.dtype)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, N, self.all_head_size)
                )
                context = torch.nan_to_num(
                    context, nan=0.0, posinf=1e4, neginf=-1e4
                )

            except Exception as e:
                logger.debug(
                    "SOA SDPA failed, falling back to manual: %s", e
                )
                context = None

        # ---- Manual path ----
        if context is None:
            scores = (
                torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous())
                * scale
            )

            # Causal mask with past_len offset
            if is_causal:
                q_pos = (
                    past_len + torch.arange(N, device=scores.device)
                ).unsqueeze(1)
                k_pos = torch.arange(
                    N_kv, device=scores.device
                ).unsqueeze(0)
                causal_bool = k_pos <= q_pos  # (N, N_kv)
                bias_val = _safe_mask_value(scores.dtype)
                scores = scores.masked_fill(
                    ~causal_bool.unsqueeze(0).unsqueeze(0), bias_val
                )

            if key_padding_mask is not None:
                bias_val = _safe_mask_value(scores.dtype)
                scores = scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), bias_val
                )

            scores = torch.nan_to_num(
                scores, nan=-1e4, posinf=1e4, neginf=-1e4
            )
            attn_weights = F.softmax(scores, dim=-1).to(v.dtype)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            if head_mask is not None:
                hm = head_mask.to(attn_weights.dtype)
                if hm.dim() == 1:
                    hm = hm.view(1, -1, 1, 1)
                elif hm.dim() == 2:
                    hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
                elif hm.dim() == 3:
                    hm = hm.unsqueeze(-1)
                attn_weights = attn_weights * hm

            attn_weights = self.dropout(attn_weights)
            context = (
                torch.matmul(attn_weights, v)
                .transpose(1, 2)
                .contiguous()
                .view(B, N, self.all_head_size)
            )

        # ---- Output projection ----
        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Return ----
        if self.is_decoder and not is_cross:
            if output_attentions:
                return output, new_past_kv, attn_weights
            return output, new_past_kv

        if output_attentions:
            return output, attn_weights
        return output, None




# ===========================================================================
# Injection
# ===========================================================================
def inject_soa_attn(
    model: nn.Module | None = None,
    layer: nn.Module | None = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    target_layer_idx: int | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> SOAAttention | nn.Module | None:
    """Replace self-attention with SOAAttention across model families."""
    # [SOA-FIX-09] Recursion guard
    if _depth > 8:
        logger.warning("inject_soa_attn: max recursion depth exceeded.")
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
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            if target_layer_idx is not None and idx != target_layer_idx:
                continue
            res = inject_soa_attn(
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

    # ---- Single layer injection ----
    try:
        src_attn = old_attn
        if src_attn is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    src_attn = getattr(cand, "self", cand)
                    break

        if src_attn is None and target is not None and hasattr(target, "forward"):
            src_attn = target

        if src_attn is None:
            return None

        # [SOA-FIX-09] Idempotency guard
        if isinstance(src_attn, SOAAttention) or "SOA" in src_attn.__class__.__name__:
            return None
        if hasattr(src_attn, "inner") and isinstance(src_attn.inner, SOAAttention):
            return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, SOAAttention) or "SOA" in cand.__class__.__name__:
                    return None

        # Skip hybrid layers (DeltaNet, LinearAttention)
        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        k_rank = getattr(
            comp_config,
            "soa_rank",
            getattr(
                comp_config,
                "rank",
                kwargs.pop("k_rank", 128),
            ),
        )
        dropout = getattr(
            comp_config, "dropout", kwargs.pop("dropout", 0.1)
        )

        new_attn = SOAAttention(
            config=cfg,
            k_rank=k_rank,
            dropout=dropout,
            pretrained_attn=src_attn,
            layer_idx=layer_idx,
            **kwargs,
        )

        # ---- Splice ----
        if target is not None:
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
                logger.error(
                    "inject_soa_attn: no writable attention attribute on %s",
                    type(target).__name__,
                )
                return None

            return new_attn

        return new_attn

    except Exception as e:
        logger.exception(
            "inject_soa_attn failed at layer %d: %s", layer_idx, e
        )
        return None


# ===========================================================================
# Backward-compat aliases (minimal — one canonical per the clean-naming policy)
# ===========================================================================
SOAFusedAttention = SOAAttention
inject_soa = inject_soa_attn
# Aliases for testing compatibility
inject_soa_attention = inject_soa_attn
inject_soafusedattention = inject_soa_attn



__all__ = [
    "SOAAttention",
    "SOAFusedAttention",
    "FactorizedLinear",
    "FactorizedLoRAUp",
    "LoRALinear",
    "inject_soa_attn",
    "inject_soa",
    "inject_soa_attention",
    "inject_soafusedattention",
    "apply_rotary_pos_emb",
]