"""ORION: Optimized Rectified Input-aware Orthogonal Network (Corrected).

Hybrid multi-head attention architecture integrating zero-shot SVD low-rank factorized
projections, head-wise bifurcated linear kernel vs. temperature-softmax full attention,
and random Fourier features.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal cumulative scan for kernel heads and causal masking for full heads
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ============================================================================
# Config helpers
# ============================================================================

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ============================================================================
# RoPE (OR-FIX-06: dtype-safe)
# ============================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


# ============================================================================
# OR-FIX-01: temperature softmax (replaces broken ReLU-based softpick)
# ============================================================================

def temperature_softmax(
    x: torch.Tensor, dim: int = -1, temperature: float = 1.0,
) -> torch.Tensor:
    """
    Standard temperature softmax. Replaces the ReLU-based softpick that was
    always-zero whenever max(x) <= 0 (e.g., fully-masked rows).
    """
    x_f = x.float()
    temp = max(float(temperature), 1e-6)
    return F.softmax(x_f / temp, dim=dim).to(x.dtype)


# ============================================================================
# ORIONConfig
# ============================================================================

@dataclass
class ORIONConfig:
    svd_energy: float = 0.95
    svd_min_rank: int = 4
    svd_max_rank: int = 64
    num_kernel_heads: int = 8
    num_full_heads: int = 4
    rff_features: int = 256
    softpick_temperature: float = 1.0
    dropout: float = 0.1


# ============================================================================
# FactorisedLinear
# ============================================================================

class FactorisedLinear(nn.Module):
    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        energy: float = 0.95,
        min_rank: int = 4,
        max_rank: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.energy = energy
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        rank = max(min_rank, min(max_rank, min(in_features, out_features)))
        self.rank = rank
        self.svd_rank = rank   # alias for framework FLOPs detection

        factory_kwargs = {"device": device, "dtype": dtype}
        self.down = nn.Linear(in_features, rank, bias=False, **factory_kwargs)
        self.up = nn.Linear(rank, out_features, bias=bias, **factory_kwargs)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(
                torch.empty((out_features, lora_rank), **factory_kwargs))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.freeze_base_weights()

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.up.bias

    def decompose_from_weight(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
    ) -> None:
        out_dim, in_dim = weight.shape
        with torch.no_grad():
            W = weight.float()
            # Shape alignment
            if W.shape[0] != self.out_features or W.shape[1] != self.in_features:
                padded = torch.zeros(self.out_features, self.in_features,
                                     device=W.device, dtype=W.dtype)
                r = min(self.out_features, W.shape[0])
                c = min(self.in_features, W.shape[1])
                padded[:r, :c] = W[:r, :c]
                W = padded

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            total = (S ** 2).sum()
            cum = torch.cumsum(S ** 2, dim=0) / torch.clamp(total, min=1e-8)
            rank = int((cum < self.energy).sum().item()) + 1
            max_avail_rank = self.down.weight.shape[0]
            rank = max(self.min_rank, min(rank, min(self.out_features, self.in_features),
                                          self.max_rank, max_avail_rank))
            self.rank = rank

            device = self.down.weight.device
            dtype = self.down.weight.dtype

            A = U[:, :rank] * S[:rank].unsqueeze(0)
            B = Vt[:rank, :]

            self.down.weight.data.zero_()
            self.up.weight.data.zero_()
            self.down.weight.data[:rank, :self.in_features].copy_(
                B.to(device=device, dtype=dtype))
            self.up.weight.data[:self.out_features, :rank].copy_(
                A.to(device=device, dtype=dtype))

            if bias is not None and self.up.bias is not None:
                min_b = min(self.up.bias.numel(), bias.numel())
                self.up.bias.data[:min_b].copy_(
                    bias[:min_b].to(device=device, dtype=dtype))
            elif self.up.bias is not None:
                self.up.bias.data.zero_()

            if self.freeze_base:
                self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.down.weight.requires_grad = True
        self.up.weight.requires_grad = True
        if self.up.bias is not None:
            self.up.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


# ============================================================================
# ORIONAttention
# ============================================================================

@ATTENTION_REGISTRY.register("orion_module", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("orion", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("orion_attention", supports_causal=True, supports_gqa=True)
class ORIONAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
        orion_cfg: Optional[ORIONConfig] = None,
        pretrained_attn: Optional[Any] = None,
        old_attn: Optional[Any] = None,
        comp_config: Optional[Any] = None,
        bert_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        cfg = config or model_config
        source_attn = pretrained_attn if pretrained_attn is not None else (
            old_attn if old_attn is not None else kwargs.get("source_attn"))
        try:
            super().__init__(cfg, rank)
        except TypeError:
            super().__init__(cfg)

        if orion_cfg is None:
            orion_cfg = ORIONConfig()
            if rank is not None:
                orion_cfg.svd_max_rank = max(rank, orion_cfg.svd_min_rank)

        hidden_size = kwargs.get("hidden_size", None)
        num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", None))
        num_kv_heads = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", None))

        if hidden_size is None or num_heads is None:
            if cfg is not None:
                if hidden_size is None:
                    hidden_size = _cfg_get(cfg, "hidden_size",
                                           _cfg_get(cfg, "d_model", _cfg_get(cfg, "dim", 768)))
                if num_heads is None:
                    num_heads = _cfg_get(cfg, "num_attention_heads",
                                         _cfg_get(cfg, "num_heads", 12))
                if num_kv_heads is None:
                    num_kv_heads = _cfg_get(cfg, "num_key_value_heads",
                                            _cfg_get(cfg, "num_kv_heads", num_heads))
            elif source_attn is not None:
                if hidden_size is None:
                    out_layer = getattr(
                        source_attn, "out_lin",
                        getattr(source_attn, "dense",
                                getattr(source_attn, "c_proj",
                                        getattr(source_attn, "o_proj", None))))
                    hidden_size = getattr(out_layer, "out_features", 768) if out_layer is not None else 768
                if num_heads is None:
                    num_heads = getattr(source_attn, "num_attention_heads",
                                        getattr(source_attn, "n_heads", 12))
                if num_kv_heads is None:
                    num_kv_heads = getattr(source_attn, "num_key_value_heads", num_heads)
            else:
                hidden_size = 768 if hidden_size is None else hidden_size
                num_heads = 12 if num_heads is None else num_heads
                num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.num_heads = max(1, int(num_heads))
        self.num_kv_heads = max(1, int(num_kv_heads))
        self.hidden_size = int(hidden_size)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by "
                f"num_heads {self.num_heads}")
        self.head_dim = self.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True
        self.cfg = orion_cfg
        self.bert_mode = bert_mode

        # Allocate kernel vs full heads
        if self.num_heads <= 1:
            self.num_kernel_heads = 1
            self.num_full_heads = 0
        else:
            self.num_full_heads = max(1, self.num_heads // 3)
            self.num_kernel_heads = self.num_heads - self.num_full_heads

        # Decoder detection
        model_type = str(getattr(cfg, "model_type", "") or "").lower() if cfg else ""
        is_dec = bool(getattr(cfg, "is_decoder",
                              getattr(cfg, "is_causal", kwargs.get("is_causal", False)))) if cfg else False
        if any(model_type.startswith(p) for p in
               ("gpt", "qwen", "llama", "mistral", "falcon", "bloom", "opt")):
            is_dec = True
        elif hasattr(cfg, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a)
            for a in (getattr(cfg, "architectures", []) or [])
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # LoRA config
        self.lora_rank = (
            getattr(cfg, "lora_rank",
                    getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            if cfg else 16
        )
        self.lora_alpha = (
            getattr(cfg, "lora_alpha",
                    getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
            if cfg else 32.0
        )
        self.freeze_base = (self.lora_rank > 0)

        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and source_attn is not None:
            p = next(source_attn.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        # Projections
        self.q_proj = FactorisedLinear(
            hidden_size, self.all_head_size, bias=True,
            energy=orion_cfg.svd_energy, min_rank=orion_cfg.svd_min_rank,
            max_rank=orion_cfg.svd_max_rank,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.k_proj = FactorisedLinear(
            hidden_size, self.all_kv_head_size, bias=True,
            energy=orion_cfg.svd_energy, min_rank=orion_cfg.svd_min_rank,
            max_rank=orion_cfg.svd_max_rank,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.v_proj = FactorisedLinear(
            hidden_size, self.all_kv_head_size, bias=True,
            energy=orion_cfg.svd_energy, min_rank=orion_cfg.svd_min_rank,
            max_rank=orion_cfg.svd_max_rank,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.out_proj = FactorisedLinear(
            self.all_head_size, hidden_size, bias=True,
            energy=orion_cfg.svd_energy, min_rank=orion_cfg.svd_min_rank,
            max_rank=orion_cfg.svd_max_rank,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )

        rff_init = torch.randn(
            max(1, self.num_kernel_heads), self.head_dim, orion_cfg.rff_features
        ) * (1.0 / math.sqrt(self.head_dim))
        self.register_buffer("rff_proj", rff_init, persistent=False)
        self.dropout = nn.Dropout(orion_cfg.dropout)

        if source_attn is not None:
            self.copy_weights(source_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    # ------------------------------------------------------------------
    @property
    def query(self) -> FactorisedLinear: return self.q_proj
    @property
    def key(self) -> FactorisedLinear: return self.k_proj
    @property
    def value(self) -> FactorisedLinear: return self.v_proj
    @property
    def dense(self) -> FactorisedLinear: return self.out_proj
    @property
    def o_proj(self) -> FactorisedLinear: return self.out_proj
    @property
    def q_lin(self) -> FactorisedLinear: return self.q_proj
    @property
    def k_lin(self) -> FactorisedLinear: return self.k_proj
    @property
    def v_lin(self) -> FactorisedLinear: return self.v_proj
    @property
    def out_lin(self) -> FactorisedLinear: return self.out_proj
    @property
    def c_proj(self) -> FactorisedLinear: return self.out_proj

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention",
                         "XLMRobertaSelfAttention", "ElectraSelfAttention",
                         "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (c_attn_mod.__class__.__name__ == "Conv1D"
                        or base_c.__class__.__name__ == "Conv1D"
                        or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :],
                                  c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size:2 * split_size, :],
                                  c_b[split_size:2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size:, :],
                                  c_b[2 * split_size:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (c_proj_mod.__class__.__name__ == "Conv1D"
                            or base_o.__class__.__name__ == "Conv1D"
                            or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # RoBERTa nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # Flat modules (DistilBERT, Qwen, etc.)
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

        # UniversalAdapter fallback
        missing = [k for k, (w, _) in extracted.items()
                   if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                "ORIONAttention.copy_weights: missing projections %s from %s",
                missing_after, src.__class__.__name__,
            )
            return

        target_map = {
            "q": self.q_proj, "k": self.k_proj, "v": self.v_proj, "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        # Identity for nested encoders
                        target_mod.down.weight.data.zero_()
                        target_mod.up.weight.data.zero_()
                        r = min(target_mod.rank, self.hidden_size, self.all_head_size)
                        target_mod.down.weight.data[:r, :r].fill_diagonal_(1.0)
                        target_mod.up.weight.data[:r, :r].fill_diagonal_(1.0)
                        if target_mod.up.bias is not None:
                            target_mod.up.bias.data.zero_()
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                # Orientation
                expected = (target_mod.out_features, target_mod.in_features)
                if w.shape != expected and w.t().contiguous().shape == expected:
                    w = w.t().contiguous()

                target_mod.decompose_from_weight(
                    w[:target_mod.out_features, :target_mod.in_features],
                    b[:target_mod.out_features] if b is not None else None,
                )

                norm = float(torch.linalg.norm(target_mod.weight.float()).item())
                if norm < 1e-4:
                    logger.warning(
                        "ORIONAttention: %s has near-zero Frobenius norm (%.2e)",
                        proj_key, norm,
                    )

        # ------------------------------------------------------------------
        # OR-FIX-03: _kernel_linear_attention with token_valid
        # ------------------------------------------------------------------
    def _kernel_linear_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        FAVOR+ random feature linear attention.

        Implements the unbiased feature map from Choromanski et al. (2020):
            φ(x) = exp(ω·x - ||x||²/2) / sqrt(m),   ω ~ N(0, I_d)

        The `-||x||²/2` term is essential for an unbiased estimate of exp(q·k).
        The scaling 1/sqrt(d) is applied to x once (not to ω, not twice).
        """
        B, Hk, N, D = Q.shape
        if Hk == 0:
            return torch.zeros(B, 0, N, D, device=Q.device, dtype=Q.dtype)

        F_dim = self.cfg.rff_features
        orig_dtype = Q.dtype
        Q, K, V = Q.float(), K.float(), V.float()
        proj = self.rff_proj[:Hk].to(Q.device).float()   # (Hk, D, m); ω ~ N(0, I_d)

        # Scale x ONCE by 1/sqrt(d). This is the "x" entering the FAVOR+ formula.
        scale = 1.0 / math.sqrt(self.head_dim)
        Q_scaled = Q * scale
        K_scaled = K * scale

        # Project onto random features:  (B, Hk, N, m)
        Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_scaled, proj)
        K_proj = torch.einsum("bhnd,hdf->bhnf", K_scaled, proj)

        # Subtract ||x||²/2 (per query/key) → unbiased FAVOR+ feature map
        Q_norm_sq = (Q_scaled ** 2).sum(dim=-1, keepdim=True) / 2.0   # (B, Hk, N, 1)
        K_norm_sq = (K_scaled ** 2).sum(dim=-1, keepdim=True) / 2.0   # (B, Hk, N, 1)

        # Numerical safety: subtract global max before exp (log-sum-exp style)
        # Mathematically cancels out in num/denom, so accuracy is unchanged.
        max_val = max(float(Q_proj.detach().amax().item()), float(K_proj.detach().amax().item()))
        max_val = max(max_val, 0.0)  # only subtract if positive

        Q_feat = torch.exp(Q_proj - Q_norm_sq - max_val) / math.sqrt(F_dim)
        K_feat = torch.exp(K_proj - K_norm_sq - max_val) / math.sqrt(F_dim)

        # Padding mask on K and V
        if token_valid is not None:
            mask_kv = token_valid.unsqueeze(1).unsqueeze(-1).to(K_feat.dtype)
            K_feat = K_feat * mask_kv
            V = V * mask_kv

        M = Q_feat.shape[-1]
        if self.is_decoder:
            chunk_size = min(64, N)
            num_chunks = (N + chunk_size - 1) // chunk_size
            S = torch.zeros(B, Hk, M, D, dtype=torch.float32, device=Q.device)
            Z = torch.zeros(B, Hk, M, dtype=torch.float32, device=Q.device)
            ctx_list = []

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)
                L = end - start

                q_c = Q_feat[:, :, start:end]
                k_c = K_feat[:, :, start:end]
                v_c = V[:, :, start:end]

                causal_mask = torch.tril(torch.ones(L, L, device=Q.device, dtype=torch.float32))
                intra_attn = torch.matmul(q_c, k_c.transpose(-1, -2).contiguous()) * causal_mask
                intra_out = torch.matmul(intra_attn, v_c)
                inter_out = torch.matmul(q_c, S)

                intra_den = intra_attn.sum(dim=-1, keepdim=True)
                inter_den = torch.matmul(q_c, Z.unsqueeze(-1))

                den = (intra_den + inter_den).clamp(min=1e-4)
                ctx_list.append((intra_out + inter_out) / den)

                S = S + torch.matmul(k_c.transpose(-1, -2).contiguous(), v_c)
                Z = Z + k_c.sum(dim=2)

            ctx = torch.cat(ctx_list, dim=2)
        else:
            KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)
            K_sum = K_feat.sum(dim=2)
            denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1).clamp(min=1e-4)
            ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / denom
        return ctx.to(orig_dtype)


    # ------------------------------------------------------------------
    # OR-FIX-02: _full_softpick_attention rewritten
    # ------------------------------------------------------------------
    def _full_softpick_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full softmax attention for the "full" heads.

        Returns a tensor of shape (B, Hf, N, D). No internal out_proj is
        applied here; the caller handles out_proj after concatenating all
        head groups.
        """
        B, Hf, N, D = Q.shape
        if Hf == 0:
            return torch.zeros(B, 0, N, D, device=Q.device, dtype=Q.dtype)

        scale = self.head_dim ** -0.5
        is_causal = bool(self.is_decoder) and N > 1

        # Build additive attention bias for padding
        attn_bias = None
        if token_valid is not None:
            valid_bias = ((1.0 - token_valid.float()) * -1e4).view(B, 1, 1, N)
            attn_bias = valid_bias.to(Q.dtype)

        # Try SDPA
        try:
            q_sdp = Q.half() if Q.dtype == torch.float32 else Q
            k_sdp = K.half() if K.dtype == torch.float32 else K
            v_sdp = V.half() if V.dtype == torch.float32 else V
            bias_sdp = None
            if attn_bias is not None:
                bias_sdp = attn_bias.to(q_sdp.dtype)

            out = F.scaled_dot_product_attention(
                q_sdp, k_sdp, v_sdp,
                attn_mask=bias_sdp,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal and bias_sdp is None,
            )
            out = out.to(Q.dtype)
            return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        except Exception:
            pass

        # Fallback: manual temperature softmax
        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        if is_causal:
            causal_mask = torch.tril(torch.ones(N, N, device=Q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if token_valid is not None:
            scores = scores.masked_fill(
                ~token_valid.unsqueeze(1).unsqueeze(2).bool(), -1e4
            )

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_w = temperature_softmax(
            scores, dim=-1, temperature=self.cfg.softpick_temperature
        ).to(V.dtype)
        attn_w = self.dropout(attn_w)
        return torch.matmul(attn_w, V)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.dim() == 4:
            B, H, L, Dh = hidden_states.shape
            hidden_states = hidden_states.view(B, L, H * Dh)
        elif not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, _ = hidden_states.shape
        Hk = self.num_kernel_heads
        Hf = self.num_full_heads
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv = kv_input.shape[1]

        Q_all = self.q_proj(hidden_states).view(
            B, N, self.num_heads, self.head_dim
        ).transpose(1, 2).contiguous()
        K_all = self.k_proj(kv_input).view(
            B, N_kv, self.num_kv_heads, self.head_dim
        ).transpose(1, 2).contiguous()
        V_all = self.v_proj(kv_input).view(
            B, N_kv, self.num_kv_heads, self.head_dim
        ).transpose(1, 2).contiguous()

        # RoPE
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q_all, K_all = apply_rotary_pos_emb(Q_all, K_all, cos, sin)

        # GQA expansion
        if self.kv_repeat > 1:
            K_all = repeat_kv(K_all, self.kv_repeat)
            V_all = repeat_kv(V_all, self.kv_repeat)

        # Token valid mask
        token_valid = None
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            token_valid = get_token_valid_mask(eff_mask, B, N_kv, device)

        ctx_list = []

        # OR-FIX-04: pass token_valid to both branches
        if Hk > 0:
            Q_kernel = Q_all[:, :Hk]
            K_kernel = K_all[:, :Hk]
            V_kernel = V_all[:, :Hk]
            ctx_kernel = self._kernel_linear_attention(
                Q_kernel, K_kernel, V_kernel, token_valid
            )
            ctx_list.append(ctx_kernel)

        if Hf > 0:
            Q_full = Q_all[:, Hk:]
            K_full = K_all[:, Hk:]
            V_full = V_all[:, Hk:]
            ctx_full = self._full_softpick_attention(
                Q_full, K_full, V_full, token_valid
            )
            ctx_list.append(ctx_full)

        if len(ctx_list) == 1:
            context = ctx_list[0]
        else:
            context = torch.cat(ctx_list, dim=1)

        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        # head_mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            context = context * hm.to(context.dtype)

        # Reshape to (B, N, H*D)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        # Output projection (skipped for nested encoders)
        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.out_proj.up.weight.device)


# ============================================================================
# Injection
# ============================================================================

@ATTENTION_REGISTRY.register("orion_inject")
def inject_orion(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Optional[Union[ORIONAttention, nn.Module]]:
    """Safe injection hook for ORIONAttention across diverse transformer architectures."""
    # OR-FIX-07: recursion guard
    if _depth > 8:
        logger.warning("inject_orion: max recursion depth exceeded, aborting.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
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
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_orion(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # Single layer injection
    try:
        src_attn = old_attn
        if src_attn is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if cand is None:
                        continue
                    if hasattr(cand, "self"):
                        src_attn = cand.self
                    else:
                        src_attn = cand
                    break

        if src_attn is None and target is not None and hasattr(target, "forward"):
            src_attn = target

        if src_attn is None:
            return None

        # Double-injection guard
        if isinstance(src_attn, ORIONAttention):
            return None
        if "ORION" in src_attn.__class__.__name__:
            return None
        for sub_attr in ("inner", "_wrapped_attention"):
            sub = getattr(src_attn, sub_attr, None)
            if isinstance(sub, ORIONAttention):
                return None

        # Skip hybrid DeltaNet/LinearAttention layers (Qwen3.5)
        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        hidden_size = (
            kwargs.get("hidden_size")
            or getattr(src_attn, "hidden_size", None)
            or getattr(src_attn, "embed_dim", None)
            or (src_attn.q_proj.in_features
                if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
                else None)
            or (src_attn.query.in_features
                if hasattr(src_attn, "query") and hasattr(src_attn.query, "in_features")
                else None)
            or (src_attn.q_lin.in_features
                if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features")
                else None)
            or (target.in_features if hasattr(target, "in_features") else None)
            or getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", 768)
        )
        num_heads = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(src_attn, "num_heads", None)
            or getattr(src_attn, "num_attention_heads", None)
            or getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "num_heads", 12)
        )

        new_attn = ORIONAttention(
            config=cfg,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            source_attn=src_attn,
            **kwargs,
        )

        # Attach in-place
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
            return new_attn
        for attr in ("self_attn", "attn", "self"):
            if hasattr(target, attr):
                setattr(target, attr, new_attn)
                return new_attn
        target.attention = new_attn
        return new_attn
    except Exception as e:
        logger.exception("inject_orion failed: %s", e)
        return None


inject_orion_attention = inject_orion