"""Galerkin Attention: Petrov-Galerkin Projection Attention (Corrected v3).

Rollback notes:
  The previous "linear attention with sequence-space normalization" attempt
  introduced NaN in training because Φ Φᵀ ≠ I_N, so Φ (Φᵀ 1_N) is not a
  valid reconstruction of the all-ones vector. This version reverts to the
  original softmax-in-basis-space formulation, which is mathematically
  valid and numerically stable.

"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

# ============================================================================
# Helpers
# ============================================================================


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    even_d = (d // 2) * 2
    x_rot = x[..., :even_d]
    half = even_d // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if even_d < d:
        return torch.cat([rotated, x[..., even_d:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q = (q * cos.to(q.dtype)) + (_rotate_half(q) * sin.to(q.dtype))
        k = (k * cos.to(k.dtype)) + (_rotate_half(k) * sin.to(k.dtype))

    return q, k


# ============================================================================
# LoRALinear
# ============================================================================


class LoRALinear(nn.Module):
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
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
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

    def unfreeze_base_weights(self) -> None:
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


# ============================================================================
# GalerkinAttention
# ============================================================================


@ATTENTION_REGISTRY.register("galerkin", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("galerkin_attention", supports_causal=True, supports_gqa=True)
class GalerkinAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        dropout: float = 0.1,
        basis_type: str = "ortho_dct",
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---- Config discovery ----
        if config is None and source_attn is not None:

            class DummyConfig:
                pass

            config = DummyConfig()
            out_layer = getattr(
                source_attn,
                "out_lin",
                getattr(
                    source_attn,
                    "dense",
                    getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None)),
                ),
            )
            q_layer = getattr(
                source_attn,
                "q_lin",
                getattr(
                    source_attn,
                    "query",
                    getattr(source_attn, "c_attn", getattr(source_attn, "q_proj", None)),
                ),
            )
            if out_layer is not None and hasattr(out_layer, "out_features"):
                config.hidden_size = out_layer.out_features
            elif q_layer is not None and hasattr(q_layer, "in_features"):
                config.hidden_size = q_layer.in_features
            else:
                config.hidden_size = 768
            config.num_attention_heads = getattr(
                source_attn,
                "n_heads",
                getattr(
                    source_attn,
                    "num_attention_heads",
                    getattr(source_attn, "num_heads", 12),
                ),
            )

        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        self.num_heads: int = (
            getattr(config, "num_attention_heads", getattr(config, "num_heads", 12))
            if config is not None
            else 12
        )
        self.num_kv_heads: int = (
            getattr(config, "num_key_value_heads", self.num_heads)
            if config is not None
            else self.num_heads
        )
        hidden_size: int = (
            getattr(config, "hidden_size", getattr(config, "d_model", 768))
            if config is not None
            else 768
        )
        cfg_head_dim = getattr(config, "head_dim", None) if config is not None else None
        if cfg_head_dim is None and source_attn is not None:
            for attr in ("q_proj", "query", "q_lin", "W_q"):
                mod = getattr(source_attn, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    cfg_head_dim = mod.weight.shape[0] // max(1, self.num_heads)
                    break
        self.head_dim: int = (
            int(cfg_head_dim)
            if cfg_head_dim is not None
            else getattr(config, "d_kv", hidden_size // max(1, self.num_heads))
        )
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.rank: int = (
            rank
            if rank is not None
            else getattr(config, "galerkin_rank", getattr(config, "rank", 64))
        )
        self.basis_type: str = basis_type

        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (
                layer_idx
                if layer_idx is not None
                else kwargs.get("layer_id", kwargs.get("layer_idx", 0))
            )
        )
        self.layer_idx: int = int(resolved_layer_id)

        # ---- LoRA config ----
        self.lora_rank: int = getattr(
            config,
            "lora_rank",
            getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
        )
        self.lora_alpha: float = getattr(
            config,
            "lora_alpha",
            getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
        )
        self.freeze_base: bool = self.lora_rank > 0

        # ---- Decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(getattr(config, "model_type", "") or "").lower() if config is not None else ""
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
            )
            is_dec = False
            if config is not None:
                if getattr(config, "is_decoder", False) or getattr(config, "is_causal", False):
                    is_dec = True
                if any(_mt.startswith(p) for p in _decoder_prefixes):
                    is_dec = True
                if hasattr(config, "architectures") and any(
                    "GPT" in str(a) or "CausalLM" in str(a)
                    for a in (getattr(config, "architectures", []) or [])
                ):
                    is_dec = True
            if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
                is_dec = True
            self.is_decoder = bool(is_dec)
        self.is_causal: bool = bool(self.is_decoder)

        # ---- Nested encoder detection ----
        is_nested = kwargs.get("is_nested_encoder")
        if is_nested is None and source_attn is not None:
            src_cls = source_attn.__class__.__name__
            if (
                src_cls
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                )
                or (src_cls.endswith("SelfAttention") and "distil" not in src_cls.lower())
                or (hasattr(source_attn, "self") and hasattr(source_attn, "output"))
            ):
                is_nested = True
            elif (
                hasattr(source_attn, "out_lin")
                or hasattr(source_attn, "c_proj")
                or hasattr(source_attn, "o_proj")
            ):
                is_nested = False
        if is_nested is None:
            is_nested = False
        self.is_nested_encoder: bool = bool(is_nested)

        # ---- Projections ----
        self.query = LoRALinear(
            hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.key = LoRALinear(
            hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.value = LoRALinear(
            hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        dropout_p = (
            getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", dropout),
                ),
            )
            if config
            else dropout
        )
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = None
        self.comp_config: Any | None = comp_config

        # GAL-FIX-07: small cache for the basis matrix
        self._basis_cache: dict[tuple[int, int], torch.Tensor] = {}

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ---- Aliases ----
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
    def o_proj(self):
        return self.out_proj

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
    def W_q(self):
        return self.query

    @property
    def W_k(self):
        return self.key

    @property
    def W_v(self):
        return self.value

    @property
    def dense(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

    # ---- Weight transfer (unchanged from original, correct) ----
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        if has_out_lin:
            is_nested = False
        elif (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            and hasattr(src, "output")
            and hasattr(src.output, "dense")
        ):
            is_nested = True
            src = src.self
        elif cls_name in (
            "BertSelfAttention",
            "RobertaSelfAttention",
            "XLMRobertaSelfAttention",
            "ElectraSelfAttention",
            "DebertaSelfAttention",
        ) or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower()):
            is_nested = True
        else:
            is_nested = False
        self.is_nested_encoder = bool(is_nested)

        if hasattr(src, "rotary_emb"):
            self.rotary_emb = src.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                cls_c = c_attn_mod.__class__.__name__
                base_cls_c = base_c.__class__.__name__
                if (
                    "Conv1D" in cls_c
                    or "Conv1D" in base_cls_c
                    or hasattr(c_attn_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (
                        c_w.ndim == 2
                        and c_w.shape[0] < c_w.shape[1]
                        and "Linear" not in cls_c
                        and "Linear" not in base_cls_c
                    )
                ):
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

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    cls_o = c_proj_mod.__class__.__name__
                    base_cls_o = base_o.__class__.__name__
                    if (
                        "Conv1D" in cls_o
                        or "Conv1D" in base_cls_o
                        or hasattr(c_proj_mod, "nf")
                        or hasattr(base_o, "nf")
                        or (
                            o_w.ndim == 2
                            and o_w.shape[0] < o_w.shape[1]
                            and "Linear" not in cls_o
                            and "Linear" not in base_cls_o
                        )
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # 2. BERT / RoBERTa nested
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

        # 3. Flat modules
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query", "W_q"],
                "k": ["k_lin", "k_proj", "key", "W_k"],
                "v": ["v_lin", "v_proj", "value", "W_v"],
                "o": ["out_lin", "out_proj", "dense", "o_proj", "proj", "output"],
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

        # UniversalAdapter fallback
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
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback warning: %s", e)

        target_map = {
            "q": self.query,
            "k": self.key,
            "v": self.value,
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
                    w[:out_d, :in_d].to(
                        device=target_mod.weight.device, dtype=target_mod.weight.dtype
                    )
                )

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.linalg.norm(target_mod.weight.data.float()).item())
                if w_norm < 1e-4:
                    logger.warning("GalerkinAttention: %s near-zero norm (%.2e)", proj_key, w_norm)

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # Basis construction (GAL-FIX-07: cached)
    # ------------------------------------------------------------------
    def _get_orthonormal_basis(
        self,
        N: int,
        r: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        r_actual = max(1, min(r, N))
        key = (N, r_actual)
        cached = self._basis_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached

        n = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(1)
        k = torch.arange(r_actual, device=device, dtype=torch.float32).unsqueeze(0)

        phi = torch.cos((math.pi * k * (2.0 * n + 1.0)) / (2.0 * max(1, N)))
        phi[:, 0] = phi[:, 0] * (1.0 / math.sqrt(max(1, N)))
        if r_actual > 1:
            phi[:, 1:] = phi[:, 1:] * math.sqrt(2.0 / max(1, N))

        eye = torch.eye(N, r_actual, device=device, dtype=torch.float32)
        phi = phi + 1e-4 * eye
        phi, _ = torch.linalg.qr(phi)
        phi = phi.to(dtype=dtype)
        self._basis_cache[key] = phi
        return phi

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
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Linear projections
        Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        K = self.key(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.value(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # 2. RoPE (GAL-FIX-05)
        if position_embeddings is not None and not is_cross_attention:
            if isinstance(position_embeddings, (tuple, list)):
                cos, sin = position_embeddings[0], position_embeddings[1]
            else:
                cos, sin = position_embeddings, None
            if sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

        # 3. GQA
        if self.num_kv_heads != self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask

        # 4. Decoder causal path
        if self.is_decoder and not is_cross_attention:
            # GAL-FIX-03: only use SDPA when we don't need attn weights/head_mask
            if not output_attentions and head_mask is None:
                try:
                    _qs = Q.half() if Q.dtype == torch.float32 else Q
                    _ks = K.half() if K.dtype == torch.float32 else K
                    _vs = V.half() if V.dtype == torch.float32 else V
                    _is_causal = N > 1
                    _sdpa_out = F.scaled_dot_product_attention(
                        _qs,
                        _ks,
                        _vs,
                        is_causal=_is_causal,
                        dropout_p=self.dropout.p if self.training else 0.0,
                    )
                    _sdpa_out = _sdpa_out.to(hidden_states.dtype)
                    _sdpa_out = (
                        _sdpa_out.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
                    )
                    _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                    out = self.out_proj(_sdpa_out) if not self.is_nested_encoder else _sdpa_out
                    return out, None
                except Exception:
                    pass

            # Manual causal path
            scale = d**-0.5
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

            # GAL-FIX-04: explicit causal mask
            causal_diag = 1 + max(0, kv_len - N)
            causal_mask = torch.triu(
                torch.full((N, kv_len), float("-inf"), device=device, dtype=torch.float32),
                diagonal=causal_diag,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            if eff_mask is not None:
                if eff_mask.dim() == 2:
                    pad_m = (1.0 - eff_mask.unsqueeze(1).unsqueeze(2).float()) * -1e4
                    scores = scores + pad_m
                elif eff_mask.dim() == 3:
                    scores = scores + eff_mask.unsqueeze(1).float()
                elif eff_mask.dim() == 4:
                    scores = scores + eff_mask.float()

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            scores_max = scores.max(dim=-1, keepdim=True).values.detach()
            exp_scores = torch.exp((scores - scores_max).clamp(min=-50.0, max=0.0))
            denom = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            probs = (exp_scores / denom).to(dtype)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                probs = probs * head_mask.to(probs.dtype)

            probs = self.dropout(probs)
            context = torch.matmul(probs, V)
            attn_weights = probs if output_attentions else None

        # 5. Encoder: softmax in basis space (GAL-FIX-01: original formulation)
        else:
            valid_mask = get_token_valid_mask(eff_mask, B, kv_len, device)
            if valid_mask is None:
                valid_mask = torch.ones(B, kv_len, device=device, dtype=torch.bool)
            mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)

            K_masked = K * mask_expanded
            V_masked = V * mask_expanded

            r_actual = max(1, min(self.rank, kv_len))
            Phi = self._get_orthonormal_basis(kv_len, r_actual, device, dtype)
            # Phi: (N, r)

            # Project into basis
            Q_basis = torch.einsum("bhnd,nr->bhrd", Q, Phi)  # (B, H, r, d)
            K_basis = torch.einsum("bhnd,nr->bhrd", K_masked, Phi)
            V_basis = torch.einsum("bhnd,nr->bhrd", V_masked, Phi)

            scale = d**-0.5
            scores = (
                torch.matmul(Q_basis.float(), K_basis.float().transpose(-2, -1).contiguous())
                * scale
            )
            # (B, H, r, r)

            # Numerically stable softmax in basis space
            scores_max = scores.max(dim=-1, keepdim=True).values.detach()
            exp_scores = torch.exp((scores - scores_max).clamp(min=-50.0, max=0.0))
            denom = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            probs = exp_scores / denom
            probs = torch.nan_to_num(probs, nan=0.0)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                probs = probs * head_mask.to(probs.dtype)

            probs = self.dropout(probs)

            # Attention in basis space
            O_basis = torch.matmul(probs.to(dtype), V_basis)  # (B, H, r, d)

            # Reconstruct sequence representation
            context = torch.einsum("bhrd,nr->bhnd", O_basis, Phi)  # (B, H, N, d)

            # Apply padding mask on outputs
            if eff_mask is not None and not is_cross_attention:
                context = context * mask_expanded

            attn_weights = probs if output_attentions else None

        # 6. Reshape and output projection
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, attn_weights


# ============================================================================
# Injection (GAL-FIX-06: recursion guard)
# ============================================================================


@ATTENTION_REGISTRY.register("galerkin_inject")
@ATTENTION_REGISTRY.register("inject_galerkin")
@ATTENTION_REGISTRY.register("galerkin_attention_inject")
@ATTENTION_REGISTRY.register("inject_galerkin_attention")
def inject_galerkin(
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
        logger.warning("inject_galerkin: max recursion depth reached")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_galerkin(
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

    # Single layer
    layer = target
    src_attn = old_attn
    if src_attn is None:
        if hasattr(layer, "attention"):
            src_attn = layer.attention.self if hasattr(layer.attention, "self") else layer.attention
        elif hasattr(layer, "attn"):
            src_attn = layer.attn
        elif hasattr(layer, "self_attn"):
            src_attn = layer.self_attn
        elif hasattr(layer, "self"):
            src_attn = layer.self

    try:
        new_attn = GalerkinAttention(
            config=model_config if model_config is not None else cfg,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn,
            **kwargs,
        )
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                layer.attention = new_attn
                new_attn.is_nested_encoder = False
        elif hasattr(layer, "self") and hasattr(layer, "output"):
            layer.self = new_attn
            new_attn.is_nested_encoder = True
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
            new_attn.is_nested_encoder = False
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
            new_attn.is_nested_encoder = False
        return new_attn
    except Exception as e:
        logger.exception("Galerkin injection failed: %s", e)
        return None
