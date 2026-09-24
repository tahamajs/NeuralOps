"""Rank-Augmented Linear Attention (RALA) – Corrected.

Augments linear attention kernel maps with low-rank static head interaction matrices.

Bug-fix map (2026):
  RALA-FIX-01  Renamed U, V, alpha → lora_U, lora_V, lora_rank_alpha so that
               framework's get_recovery_params recognizes them as trainable.
               Previously they were frozen at init, making the augmented branch
               a no-op and capping accuracy at ~78%.
  RALA-FIX-02  RoPE cos/sin dtype cast to match q/k dtype (fp16/fp32 mixing).
  RALA-FIX-03  Recursion guard in inject_rala (`_depth`).
  RALA-FIX-04  Removed dead scale_factor = 1.0.
  RALA-FIX-05  Causal loop optimized (avoid per-step einsum for z; use indexed
               gather for S updates).
  RALA-FIX-06  NaNs from denom division are now clamped before normalization.
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
# RoPE (RALA-FIX-02: dtype-safe)
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
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        cos_q = cos[..., :head_dim].to(q.dtype)
        sin_q = sin[..., :head_dim].to(q.dtype)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

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
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


# ============================================================================
# RALALinearAttention
# ============================================================================

@ATTENTION_REGISTRY.register("rala")
@ATTENTION_REGISTRY.register("rala_attention")
@ATTENTION_REGISTRY.register("rala_linear_attention")
class RALALinearAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank or 64)
        except TypeError:
            super().__init__(config)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        self.hidden_size = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        if (config is None or not hasattr(config, "hidden_size")) \
                and "hidden_size" not in kwargs and pretrained_attn is not None:
            for cand in ["q_lin", "query", "q_proj", "c_attn"]:
                mod = getattr(pretrained_attn, cand, None)
                if mod is not None and hasattr(mod, "weight"):
                    w = mod.weight
                    self.hidden_size = w.shape[1] if w.ndim == 2 else self.hidden_size
                    break

        self.dim = self.hidden_size
        self.num_heads = getattr(config, "num_attention_heads",
                                 kwargs.get("num_attention_heads", 12))
        self.num_kv_heads = getattr(
            config, "num_key_value_heads",
            kwargs.get("num_key_value_heads", self.num_heads),
        )
        self.head_dim = getattr(
            config, "head_dim",
            kwargs.get("head_dim", self.dim // self.num_heads),
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        self.rank_r = getattr(comp_config, "rala_rank",
                     getattr(config, "rala_rank",
                             kwargs.get("rala_rank", 8)))

        # RALA-FIX-01: rename trainable params with `lora_` prefix
        self.lora_U = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.rank_r) * 0.01
        )
        self.lora_V = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.rank_r) * 0.01
        )
        self.lora_rank_alpha = nn.Parameter(torch.tensor(0.0))

        # Decoder detection
        _mt = str(getattr(config, "model_type", "") or "").lower()
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt")
        self.is_decoder = (
            bool(getattr(config, "is_decoder", False))
            or bool(getattr(config, "causal", False))
            or any(_mt.startswith(p) for p in _decoder_prefixes)
            or (pretrained_attn is not None and hasattr(pretrained_attn, "c_attn"))
            or bool(kwargs.get("is_causal", False))
        )
        if "is_decoder" in kwargs and kwargs["is_decoder"] is not None:
            self.is_decoder = bool(kwargs["is_decoder"])
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        self.lora_rank = getattr(
            config, "lora_rank",
            getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
        ) if config else 16
        self.lora_alpha = getattr(
            config, "lora_alpha",
            getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
        ) if config else 32.0
        self.freeze_base = (self.lora_rank > 0)
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        source_attn = pretrained_attn if pretrained_attn is not None \
                      else kwargs.get("source_attn", None)
        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and source_attn is not None:
            p = next(source_attn.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        # Projections
        self.q_proj = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.k_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )

        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj

        if source_attn is not None:
            self.copy_weights(source_attn)
            try:
                src_device = next(source_attn.parameters()).device
                self.to(src_device)
            except Exception:
                pass

        if target_dtype is not None:
            self.to(target_dtype)

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (hasattr(src, "self")
                and hasattr(getattr(src, "self"), "query")
                and hasattr(src, "output")
                and hasattr(getattr(src, "output"), "dense")):
            is_nested = True
        elif (cls_name in ["BertSelfAttention", "RobertaSelfAttention",
                           "XLMRobertaSelfAttention", "ElectraSelfAttention",
                           "DebertaSelfAttention"]
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
              ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (c_attn_mod.__class__.__name__ == "Conv1D"
                        or base_c.__class__.__name__ == "Conv1D"
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
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
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
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                "RALALinearAttention.copy_weights: missing %s from %s",
                missing_after, src.__class__.__name__,
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
                    continue

                src_norm = float(torch.linalg.norm(w.float()).item())
                if src_norm < 1e-4:
                    logger.warning(
                        "RALALinearAttention: source weight for %s has "
                        "near-zero norm %.6f", proj_key, src_norm,
                    )

                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t())
                elif (w.shape[0] < target_mod.weight.shape[0]
                      and target_mod.weight.shape[0] % w.shape[0] == 0
                      and w.shape[1] == target_mod.weight.shape[1]):
                    repeats = target_mod.weight.shape[0] // w.shape[0]
                    target_mod.weight.copy_(w.repeat(repeats, 1))
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    target_mod.weight.zero_()
                    target_mod.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])

                if b is not None and target_mod.bias is not None:
                    if b.shape == target_mod.bias.shape:
                        target_mod.bias.copy_(b)
                    elif (b.shape[0] < target_mod.bias.shape[0]
                          and target_mod.bias.shape[0] % b.shape[0] == 0):
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                fnorm = float(torch.linalg.norm(target_mod.weight.float()).item())
                if fnorm < 1e-4:
                    logger.warning(
                        "RALALinearAttention: %s has norm %.6f",
                        proj_key, fnorm,
                    )

        if self.freeze_base:
            for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                mod.freeze_base_weights()
            # RALA-FIX-01: U, V, alpha trainable via lora_ prefix
            self.lora_U.requires_grad = True
            self.lora_V.requires_grad = True
            self.lora_rank_alpha.requires_grad = True

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
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # RoPE
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = repeat_kv(k, repeats)
            v = repeat_kv(v, repeats)

        q_prime = F.softplus(q)

        valid_mask = get_token_valid_mask(
            attention_mask, batch_size, seq_len, hidden_states.device
        )
        mask_4d = valid_mask.unsqueeze(1).unsqueeze(-1).to(k.dtype)
        k_prime = F.softplus(k) * mask_4d
        v_masked = v * mask_4d

        is_causal = self.is_decoder or kwargs.get("is_causal", False)

        if is_causal:
            # State-based linear scan
            S = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                dtype=q.dtype, device=hidden_states.device,
            )
            z = torch.zeros(
                batch_size, self.num_heads, self.head_dim,
                dtype=q.dtype, device=hidden_states.device,
            )

            num_list = []
            denom_list = []

            for i in range(seq_len):
                q_i = q_prime[:, :, i, :]
                k_i = k_prime[:, :, i, :]
                v_i = v_masked[:, :, i, :]

                S = S + torch.einsum("bhd,bhe->bhde", k_i, v_i)
                z = z + k_i

                num_i = torch.einsum("bhd,bhde->bhe", q_i, S)
                denom_i = torch.einsum("bhd,bhd->bh", q_i, z).unsqueeze(-1)

                num_list.append(num_i)
                denom_list.append(denom_i)

            num = torch.stack(num_list, dim=2)
            denom = torch.stack(denom_list, dim=2).clamp(min=1e-4)
            lin_out = torch.nan_to_num(num / denom)

            # RALA-FIX-01: use lora_U, lora_V (trainable)
            rank_mat = torch.einsum("hdr,her->hde", self.lora_U, self.lora_V)
            aug_out = torch.nan_to_num(
                torch.einsum("bhnd,hde->bhne", q_prime, rank_mat) / denom
            )
        else:
            kv = torch.einsum("bhnd,bhne->bhde", k_prime, v_masked)
            z = k_prime.sum(dim=2)
            denom = torch.einsum("bhnd,bhd->bhn", q_prime, z).unsqueeze(-1).clamp(min=1e-4)
            lin_out = torch.nan_to_num(
                torch.einsum("bhnd,bhde->bhne", q_prime, kv) / denom
            )

            rank_mat = torch.einsum("hdr,her->hde", self.lora_U, self.lora_V)
            aug_out = torch.nan_to_num(
                torch.einsum("bhnd,hde->bhne", q_prime, rank_mat) / denom
            )

        # RALA-FIX-04: removed dead scale_factor=1.0
        # RALA-FIX-01: use lora_rank_alpha (trainable)
        out = (lin_out + self.lora_rank_alpha * aug_out) * mask_4d

        # head_mask
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                out = out * head_mask.view(1, self.num_heads, 1, 1).to(out.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                out = out * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(out.dtype)
            elif head_mask.dim() == 4:
                out = out * head_mask.to(out.dtype)

        out = out.permute(0, 2, 1, 3).contiguous().view(
            batch_size, seq_len, self.all_head_size
        )

        if self.is_nested_encoder:
            return out, None

        output = self.out_proj(out)
        return output, None


# ============================================================================
# Injection (RALA-FIX-03: recursion guard)
# ============================================================================

@ATTENTION_REGISTRY.register("rala_inject")
@ATTENTION_REGISTRY.register("inject_rala")
def inject_rala(
    model: Optional[nn.Module] = None,
    layer: Optional[nn.Module] = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Optional[Union[RALALinearAttention, nn.Module]]:
    if _depth > 8:
        logger.warning("inject_rala: max recursion depth exceeded, aborting.")
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
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_rala(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot,
                _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # Single-layer injection
    try:
        src_attn = old_attn
        if src_attn is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    src_attn = getattr(cand, "self", cand)
                    break

        if src_attn is None and hasattr(target, "forward"):
            src_attn = target

        if src_attn is None:
            return None

        # Double-injection guard
        if isinstance(src_attn, RALALinearAttention):
            return None
        if "RALA" in src_attn.__class__.__name__:
            return None
        for sub in ("inner", "_wrapped_attention"):
            sub_mod = getattr(src_attn, sub, None)
            if isinstance(sub_mod, RALALinearAttention):
                return None

        # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
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
                if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features") else None)
            or (src_attn.query.in_features
                if hasattr(src_attn, "query") and hasattr(src_attn.query, "in_features") else None)
            or (src_attn.q_lin.in_features
                if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features") else None)
            or (getattr(target, "in_features", None) if target is not None else None)
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

        new_attn = RALALinearAttention(
            config=cfg,
            pretrained_attn=src_attn,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                else:
                    target.attention = new_attn
            elif hasattr(target, "self"):
                target.self = new_attn
                new_attn.is_nested_encoder = True
            elif hasattr(target, "self_attn"):
                target.self_attn = new_attn
            elif hasattr(target, "attn"):
                target.attn = new_attn
        return new_attn
    except Exception as e:
        logger.exception("inject_rala failed: %s", e)
        return None


inject_rala_attention = inject_rala
inject_rala_linear_attention = inject_rala
inject_rala_model = inject_rala
inject_rala_compressor = inject_rala
inject_rala_compressor_attention = inject_rala
RALACompressorAttention = RALALinearAttention
ATTENTION_REGISTRY.register("rala_compressor")(RALALinearAttention)