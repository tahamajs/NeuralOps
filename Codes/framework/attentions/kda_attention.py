"""Kernelized Decay Attention (KDA) / Recurrent Linear Attention (Corrected).

Implements multi-head linear recurrence with learned exponential decay gating,
spherical normalization, denominator tracking, and pretrained weight transfer.

Bug-fix map:
  KDA-FIX-01  beta init from 2.0 → 5.0 (effective context ~8 → ~150 tokens
              at L=128). Previous init was 16x too small.
  KDA-FIX-02  Renamed `beta` → `lora_beta` and `output_scale` → `lora_output_scale`
              so framework's `get_recovery_params` recognizes them as trainable.
              Previously they were frozen at init values.
  KDA-FIX-03  Fixed infinite recursion in `_kda_scan_chunked` for bidirectional
              mode (L > 512). Added `_allow_bidirectional` flag.
  KDA-FIX-04  RoPE cos/sin dtype cast to match q/k dtype.
  KDA-FIX-05  Recursion guard in `inject_kda`.
  KDA-FIX-06  NaN-safe decay clamping in softmax denominator.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY


# ============================================================================
# RoPE
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

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        # KDA-FIX-04: dtype-safe cast
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
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
# KDAAttention
# ============================================================================

@ATTENTION_REGISTRY.register("kda", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("kda_attention", supports_causal=True, supports_gqa=True)
class KDAAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        # ---- hidden dim discovery ----
        if d_model is None:
            d_model = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
            if d_model is None:
                if config is not None:
                    d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
                elif source_attn is not None:
                    out_layer = getattr(
                        source_attn, "out_lin",
                        getattr(source_attn, "dense",
                                getattr(source_attn, "c_proj",
                                        getattr(source_attn, "o_proj",
                                                getattr(source_attn, "out_proj", None)))))
                    if out_layer is not None:
                        d_model = getattr(out_layer, "out_features",
                                          getattr(out_layer, "in_features", 768))
                    else:
                        q_l = getattr(source_attn, "q_proj",
                              getattr(source_attn, "query",
                              getattr(source_attn, "q_lin", None)))
                        d_model = (
                            getattr(q_l, "in_features",
                                    getattr(q_l, "out_features", 768))
                            if q_l is not None else 768
                        )
                else:
                    d_model = 768
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        # ---- num heads ----
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
        self.num_heads = int(num_heads)
        self.num_kv_heads = (
            getattr(config, "num_key_value_heads",
                    getattr(config, "num_kv_heads", self.num_heads))
            if config else self.num_heads
        )
        self.head_dim = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # ---- decoder detection ----
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = (
            getattr(config, "is_decoder",
                    getattr(config, "is_causal", kwargs.get("is_causal", False)))
            if config else False
        )
        _mt = (model_type or "").lower()
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                             "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if any(_mt.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a)
            for a in (getattr(config, "architectures", []) or [])
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # ---- KDA-FIX-02: rename to lora_* prefix for framework recognition ----
        # KDA-FIX-01: init at 5.0 → sigmoid(5) ≈ 0.9933 → effective context ~150 tokens
        self.lora_beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 5.0)
        self.lora_output_scale = nn.Parameter(torch.ones(1))

        # ---- LoRA config ----
        if config is not None:
            self.lora_rank = getattr(config, "lora_rank",
                            getattr(comp_config, "lora_rank",
                                    kwargs.get("lora_rank", 16)))
        elif comp_config is not None:
            self.lora_rank = getattr(comp_config, "lora_rank",
                                    kwargs.get("lora_rank", 16))
        else:
            self.lora_rank = kwargs.get("lora_rank", 16)

        if config is not None:
            self.lora_alpha = getattr(config, "lora_alpha",
                             getattr(comp_config, "lora_alpha",
                                     kwargs.get("lora_alpha", 32.0)))
        elif comp_config is not None:
            self.lora_alpha = getattr(comp_config, "lora_alpha",
                                     kwargs.get("lora_alpha", 32.0))
        else:
            self.lora_alpha = kwargs.get("lora_alpha", 32.0)

        self.freeze_base = (self.lora_rank > 0)
        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None)
            if source_attn is not None else None
        )

        # ---- projections ----
        self.q_proj = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        if source_attn is not None:
            self.copy_weights(source_attn)

    def freeze_base_weights(self) -> None:
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if hasattr(mod, "freeze_base_weights"):
                mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = (hasattr(src, "dense") or hasattr(src, "out_proj")
                   or hasattr(src, "out_lin") or hasattr(src, "c_proj")
                   or hasattr(src, "o_proj"))
        if cls_name in ("BertSelfAttention", "RobertaSelfAttention") \
                or (hasattr(src, "query") and not has_out):
            self.is_nested_encoder = True
        if hasattr(src, "self") and hasattr(src, "output"):
            self.is_nested_encoder = True

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # 1. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (c_attn_mod.__class__.__name__ == "Conv1D"
                        or base_c.__class__.__name__ == "Conv1D"
                        or hasattr(c_attn_mod, "nf")
                        or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])):
                    c_w = c_w.t()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :],
                                  c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size: 2 * split_size, :],
                                  c_b[split_size: 2 * split_size] if c_b is not None else None)
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
                            or hasattr(c_proj_mod, "nf")
                            or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])):
                        o_w = o_w.t()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa nested
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

        # 3. Flat modules
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
                            w_data = mod.weight.data.clone()
                            if (mod.__class__.__name__ == "Conv1D"
                                    or hasattr(mod, "nf")
                                    or (w_data.ndim == 2 and proj_key == "o"
                                        and w_data.shape[0] < w_data.shape[1])):
                                w_data = w_data.t()
                            extracted[proj_key] = (
                                w_data,
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

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
                logger.debug("UniversalAdapter fallback: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"KDAAttention copy_weights failed: missing mandatory projections "
                f"{missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_proj, "k": self.k_proj, "v": self.v_proj, "o": self.out_proj,
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

                if (w.shape != target_mod.weight.shape
                        and target_mod.weight.shape == w.t().shape):
                    w = w.t()
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(device=target_mod.weight.device,
                                        dtype=target_mod.weight.dtype))

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(device=target_mod.bias.device,
                                     dtype=target_mod.bias.dtype))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(w[:out_d, :in_d]).item())
                if w_norm < 1e-4:
                    raise RuntimeError(
                        f"KDAAttention weight verification failed for "
                        f"'{proj_key}': norm too small ({w_norm:.2e})"
                    )

                if self.freeze_base:
                    target_mod.freeze_base_weights()

        if self.lora_rank > 0 or self.freeze_base:
            self.freeze_base_weights()

    # ------------------------------------------------------------------
    # KDA-FIX-03: chunked scan with bidirectional guard
    # ------------------------------------------------------------------
    def _kda_scan_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: int = 256,
        _allow_bidirectional: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, L, d = q.shape
        gamma = decay.clamp(min=1e-4, max=1.0 - 1e-4)
        log_gamma = torch.log(gamma)

        mask_float = None
        if mask is not None:
            valid_m = get_token_valid_mask(mask, B, L, q.device)
            mask_float = valid_m.float()

        y_fwd_list = []
        S = torch.zeros(B, H, d, v.shape[-1], device=q.device, dtype=torch.float32)
        Z = torch.zeros(B, H, d, 1, device=q.device, dtype=torch.float32)

        num_chunks = (L + chunk_size - 1) // chunk_size
        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, L)
            qc = q[:, :, start:end, :].float()
            kc = k[:, :, start:end, :].float()
            vc = v[:, :, start:end, :].float()
            Lc = end - start

            mc = None
            if mask_float is not None:
                mc = mask_float[:, start:end]
                mc_col = mc.unsqueeze(1).unsqueeze(-1)
                kc = kc * mc_col
                vc = vc * mc_col

            pos = torch.arange(Lc, device=q.device, dtype=torch.float32)
            pos_diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)
            tril_mask = torch.tril(torch.ones(Lc, Lc, device=q.device, dtype=torch.bool))
            D_intra = torch.exp(
                (pos_diff.unsqueeze(0).unsqueeze(0) * log_gamma).clamp(min=-40.0, max=0.0)
            ) * tril_mask.float()

            attn_intra = (qc @ kc.transpose(-1, -2)) * D_intra
            if mc is not None:
                mask_2d = mc.unsqueeze(1).unsqueeze(-1) * mc.unsqueeze(1).unsqueeze(-2)
                attn_intra = attn_intra * mask_2d

            num_intra = attn_intra @ vc
            den_intra = attn_intra.sum(dim=-1, keepdim=True)

            decay_inter = torch.exp(
                ((pos + 1).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                 * log_gamma).clamp(min=-40.0, max=0.0)
            )
            num_inter = (qc @ S) * decay_inter
            den_inter = (qc @ Z) * decay_inter

            if mc is not None:
                num_inter = num_inter * mc.unsqueeze(1).unsqueeze(-1)
                den_inter = den_inter * mc.unsqueeze(1).unsqueeze(-1)

            num_total = num_intra + num_inter
            den_total = (den_intra + den_inter).clamp(min=1e-4)
            yc = (num_total / den_total).to(q.dtype)
            if mc is not None:
                yc = yc * mc.unsqueeze(1).unsqueeze(-1)
            y_fwd_list.append(yc)

            decay_to_end = torch.exp(
                ((Lc - 1 - pos).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                 * log_gamma).clamp(min=-40.0, max=0.0)
            )
            k_decayed = kc * decay_to_end
            chunk_S = k_decayed.transpose(-1, -2) @ vc
            chunk_Z = k_decayed.transpose(-1, -2) @ torch.ones_like(vc[..., :1])

            gamma_chunk = torch.exp((Lc * log_gamma).clamp(min=-40.0, max=0.0))
            S = S * gamma_chunk + chunk_S
            Z = Z * gamma_chunk + chunk_Z

        y_fwd = torch.cat(y_fwd_list, dim=2)

        # KDA-FIX-03: guard against infinite recursion in bidirectional mode
        if (not self.is_decoder) and _allow_bidirectional:
            q_rev = q.flip(dims=[2])
            k_rev = k.flip(dims=[2])
            v_rev = v.flip(dims=[2])
            mask_rev = mask_float.flip(dims=[-1]) if mask_float is not None else None
            y_bwd_rev, _ = self._kda_scan_chunked(
                q_rev, k_rev, v_rev, decay, mask_rev,
                chunk_size=chunk_size,
                _allow_bidirectional=False,
            )
            y_bwd = y_bwd_rev.flip(dims=[2])
            return y_fwd, y_bwd

        return y_fwd, None

    # ------------------------------------------------------------------
    # Parallel scan (short sequences)
    # ------------------------------------------------------------------
    def _kda_scan_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, L, d = q.shape

        if L > 512:
            return self._kda_scan_chunked(q, k, v, decay, mask, chunk_size=256)

        log_decay = torch.log(decay.clamp(min=1e-4, max=1.0 - 1e-4))
        pos = torch.arange(L, device=q.device, dtype=torch.float32)
        pos_diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)

        tril_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
        D_causal = torch.exp(
            (pos_diff.unsqueeze(0).unsqueeze(0) * log_decay).clamp(min=-40.0, max=0.0)
        ) * tril_mask.float()

        attn_scores = (q.float() @ k.float().transpose(-1, -2))

        mask_2d = None
        if mask is not None:
            valid_m = get_token_valid_mask(mask, B, L, q.device)
            mask_float = valid_m.float()
            mask_2d = mask_float.unsqueeze(1).unsqueeze(-1) * mask_float.unsqueeze(1).unsqueeze(-2)
            D_causal = D_causal * mask_2d

        num_fwd = (attn_scores * D_causal) @ v.float()
        denom_fwd = (attn_scores * D_causal).sum(dim=-1, keepdim=True)
        y_fwd = torch.clamp(
            num_fwd / denom_fwd.clamp(min=1e-4),
            min=-1e4, max=1e4,
        ).to(q.dtype)

        if not self.is_decoder:
            triu_mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool))
            D_anti = torch.exp(
                (-pos_diff.unsqueeze(0).unsqueeze(0) * log_decay).clamp(min=-40.0, max=0.0)
            ) * triu_mask.float()
            if mask_2d is not None:
                D_anti = D_anti * mask_2d
            num_bwd = (attn_scores * D_anti) @ v.float()
            denom_bwd = (attn_scores * D_anti).sum(dim=-1, keepdim=True)
            y_bwd = torch.clamp(
                num_bwd / denom_bwd.clamp(min=1e-4),
                min=-1e4, max=1e4,
            ).to(q.dtype)
            return y_fwd, y_bwd

        return y_fwd, None

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
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q_raw = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        k_raw = self.k_proj(hidden_states).view(
            B, L, self.num_kv_heads, d
        ).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(
            B, L, self.num_kv_heads, d
        ).transpose(1, 2).contiguous()

        # RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_raw, k_raw = apply_rotary_pos_emb(q_raw, k_raw, cos, sin)
        elif self.rotary_emb is not None:
            rotary_module = self.rotary_emb
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = rotary_module(q_raw, pos_ids)
                q_raw, k_raw = apply_rotary_pos_emb(q_raw, k_raw, cos, sin, position_ids=pos_ids)
            except Exception:
                try:
                    cos, sin = rotary_module(q_raw, seq_len=L)
                    q_raw, k_raw = apply_rotary_pos_emb(q_raw, k_raw, cos, sin, position_ids=pos_ids)
                except Exception as e:
                    logger.debug("KDA rotary_emb failed: %s", e)

        if self.kv_repeat > 1:
            k_raw = repeat_kv(k_raw, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        q = F.elu(q_raw) + 1.0
        k = (F.elu(k_raw) + 1.0) / (d ** 0.5)

        # KDA-FIX-01 + KDA-FIX-02: decay from lora_beta (trainable, init 5.0)
        decay = torch.sigmoid(self.lora_beta).view(1, H, 1, 1)

        mask_bool = None
        if attention_mask is not None:
            mask_bool = get_token_valid_mask(attention_mask, B, L, hidden_states.device)

        y_fwd, y_bwd = self._kda_scan_parallel(q, k, v, decay, mask_bool)

        if self.is_decoder or y_bwd is None:
            y = y_fwd
        else:
            y = 0.5 * (y_fwd + y_bwd)

        y = torch.nan_to_num(y, nan=0.0)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            y = y * hm

        out = y.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        # KDA-FIX-02: output_scale via lora_output_scale (trainable)
        out = out * self.lora_output_scale
        if self.is_nested_encoder:
            final_out = out
        else:
            final_out = self.out_proj(out)

        return final_out, None


# ============================================================================
# Injection (KDA-FIX-05: recursion guard)
# ============================================================================

@ATTENTION_REGISTRY.register("kda_inject")
@ATTENTION_REGISTRY.register("inject_kda")
def inject_kda(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Optional[Union[KDAAttention, nn.Module]]:
    if _depth > 8:
        logger.warning("inject_kda: max recursion depth exceeded, aborting.")
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
            res = inject_kda(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # Single layer
    src_attn = old_attn
    if src_attn is None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                src_attn = target.attention.self
            else:
                src_attn = target.attention
        elif hasattr(target, "self"):
            src_attn = target.self
        elif hasattr(target, "attn"):
            src_attn = target.attn
        elif hasattr(target, "self_attn"):
            src_attn = target.self_attn

    if src_attn is None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
            else None)
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 768))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 12))
    )

    try:
        new_attn = KDAAttention(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            num_heads=num_heads,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            **kwargs,
        )
    except Exception as e:
        logger.exception("KDA injection failed: %s", e)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
        elif hasattr(target, "self"):
            target.self = new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn

    return new_attn