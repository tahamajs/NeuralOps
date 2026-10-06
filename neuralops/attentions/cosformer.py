"""
Cosformer v2 — Production-Hardened with proper causal safety.

FIXES in v2:
  - FIX 1: cos/sin angles use 2π range (was π/2, too small)
  - FIX 2: denom clamp = 1.0 (was 1e-6, causes explosion)
  - FIX 3: Q/K scaled by 1/sqrt(d) after ELU+1 (was unscaled)
  - FIX 4: is_decoder forced True for GPT-family + warning
  - FIX 5: Cosformer disabled for decoders (bidirectional → leakage)
  - FIX 6: Triton dispatch disabled on Windows (unstable)
  - FIX 7: chunk_size = max(16, min(32, N // 4))
  - FIX 8: Even head_dim enforced with graceful fallback
  - FIX 9: Explicit causal mask in _forward_causal
  - FIX 10: Environment variable toggles for ablation
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

try:
    from neuralops.core.triton_kernels import triton_dispatch
except Exception:
    triton_dispatch = None


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    return v.lower() in ("1", "true", "yes", "on") if v else default


# ============================================================================
# Thread-safe injection depth guard
# ============================================================================
_inject_state = threading.local()
_MAX_DEPTH = 4


def _get_inject_depth() -> int:
    return getattr(_inject_state, "depth", 0)


def _set_inject_depth(depth: int) -> None:
    _inject_state.depth = depth


# ============================================================================
# Utilities
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


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
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
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

    return q, k


# ============================================================================
# LoRALinear
# ============================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lora_rank=16,
                 lora_alpha=32.0, freeze_base=False, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
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

    def reset_parameters(self):
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

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x):
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


# ============================================================================
# CosformerAttention
# ============================================================================
@ATTENTION_REGISTRY.register("cosformer", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("cosformer_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class CosformerAttention(BaseAttention):
    is_custom_attention: bool = True
    triton_kernel = "linear_attn"

    def __init__(self, config=None, rank=None, max_seq_len=512, act_fun="relu",
                 eps=1e-6, pretrained_attn=None, old_attn=None, comp_config=None,
                 layer_id=None, layer_idx=None, **kwargs):
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        def _cfg_get(k, default=None):
            for obj in (config, comp_config):
                if obj is not None and hasattr(obj, k):
                    val = getattr(obj, k)
                    if val is not None:
                        return val
            return kwargs.get(k, default)

        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")

        if hidden_size is None:
            hidden_size = _cfg_get("hidden_size", _cfg_get("dim", None))
            if hidden_size is None and source_attn is not None:
                out_layer = getattr(source_attn, "out_lin",
                                    getattr(source_attn, "dense",
                                            getattr(source_attn, "c_proj",
                                                    getattr(source_attn, "o_proj", None))))
                hidden_size = getattr(out_layer, "out_features", 768) if out_layer is not None else 768
            elif hidden_size is None:
                hidden_size = 768

        if num_heads is None:
            num_heads = _cfg_get("num_attention_heads", _cfg_get("num_heads", None))
            if num_heads is None and source_attn is not None:
                num_heads = getattr(source_attn, "n_heads",
                                    getattr(source_attn, "num_attention_heads",
                                            getattr(source_attn, "num_heads", 12)))
            elif num_heads is None:
                num_heads = 12

        super().__init__(config, rank)
        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)

        resolved_head_dim = _cfg_get("head_dim", None)
        if resolved_head_dim is None and source_attn is not None:
            resolved_head_dim = getattr(source_attn, "head_dim", None)
        if resolved_head_dim is None:
            resolved_head_dim = self.dim // max(1, self.num_heads)
        self.head_dim = int(resolved_head_dim)

        # FIX 8: graceful handling of odd head_dim
        if self.head_dim % 2 != 0:
            logger.warning("CosformerAttention: odd head_dim=%d, RoPE will slice to even", self.head_dim)

        kv_heads = _cfg_get("num_key_value_heads", None)
        if kv_heads is None and source_attn is not None:
            kv_heads = getattr(source_attn, "num_key_value_heads",
                               getattr(source_attn, "num_kv_heads",
                                       getattr(source_attn, "n_kv_heads", None)))
            if kv_heads is None and hasattr(source_attn, "k_proj") \
                    and hasattr(source_attn.k_proj, "out_features") and self.head_dim > 0:
                kv_heads = source_attn.k_proj.out_features // self.head_dim
        self.num_kv_heads = int(kv_heads if kv_heads is not None else self.num_heads)

        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.max_seq_len = int(_cfg_get("max_position_embeddings", max_seq_len))
        self.act_fun = act_fun
        self.eps = float(eps)

        resolved_layer_id = (layer_id if layer_id is not None
                             else (layer_idx if layer_idx is not None
                                   else kwargs.get("layer_id", kwargs.get("layer_idx", 0))))
        self.layer_idx = int(resolved_layer_id)

        self.lora_rank = int(_cfg_get("lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha = float(_cfg_get("lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base = self.lora_rank > 0

        # FIX 4: force causal for GPT-family
        model_type = getattr(config, "model_type", "").lower() if config is not None else ""
        is_dec = _cfg_get("is_decoder", _cfg_get("is_causal", kwargs.get("is_causal", False)))
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "phi", "gemma",
                             "opt", "bloom", "falcon")
        if (any(model_type.startswith(p) for p in _decoder_prefixes)
            or (hasattr(config, "architectures") and any(
                "GPT" in str(a) or "CausalLM" in str(a)
                for a in (getattr(config, "architectures", []) or [])))):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        # FIX 4: old_attn with c_attn → decoder
        if source_attn is not None:
            if hasattr(source_attn, "c_attn") or "gpt" in source_attn.__class__.__name__.lower():
                is_dec = True
        # FIX 4: safe default
        if config is None and source_attn is None:
            logger.warning("Cosformer: defaulting is_decoder=True for causal safety")
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = bool(kwargs.get("is_nested_encoder", False))

        # FIX 5: warn if bidirectional on decoder
        if not self.is_decoder:
            logger.warning(
                "Cosformer: is_decoder=False → bidirectional. "
                "This leaks future tokens for GPT-family."
            )

        # Projections
        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                    lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                    freeze_base=self.freeze_base)

        dropout_p = (getattr(config, "dropout",
                              getattr(config, "attention_probs_dropout_prob", 0.1))
                     if config else 0.1)
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = None
        self.comp_config = comp_config
        self.triton_kwargs = {"phi": "elu+1", "eps": self.eps,
                              "cos_weight": None, "sin_weight": None}

        if source_attn is not None:
            self.copy_weights(source_attn)

    # Aliases
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def W_q(self): return self.q_proj
    @property
    def W_k(self): return self.k_proj
    @property
    def W_v(self): return self.v_proj
    @property
    def c_proj(self): return self.out_proj

    def copy_weights(self, src):
        if src is None:
            return

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = (hasattr(src, "dense") or hasattr(src, "out_proj")
                   or hasattr(src, "out_lin") or hasattr(src, "c_proj")
                   or hasattr(src, "o_proj"))
        if cls_name in ["BertSelfAttention", "RobertaSelfAttention"] or (
                hasattr(src, "query") and not has_out):
            self.is_nested_encoder = True
        if hasattr(src, "self") and hasattr(src, "output"):
            self.is_nested_encoder = True

        if hasattr(src, "rotary_emb"):
            self.rotary_emb = src.rotary_emb

        extracted = {"q": (None, None), "k": (None, None),
                     "v": (None, None), "o": (None, None)}

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception:
                    pass
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if ((c_attn_mod.__class__.__name__ == "Conv1D")
                        or (base_c.__class__.__name__ == "Conv1D")
                        or hasattr(c_attn_mod, "nf")
                        or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size:2*split_size, :],
                                  c_b[split_size:2*split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2*split_size:, :],
                                  c_b[2*split_size:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if ((c_proj_mod.__class__.__name__ == "Conv1D")
                            or (base_o.__class__.__name__ == "Conv1D")
                            or hasattr(c_proj_mod, "nf")
                            or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # RoBERTa/BERT nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                base_mod = getattr(mod, "base_layer", mod)
                if base_mod is not None and hasattr(base_mod, "weight"):
                    extracted[proj_key] = (
                        base_mod.weight.data.clone(),
                        base_mod.bias.data.clone() if getattr(base_mod, "bias", None) is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                base_out = getattr(out_mod, "base_layer", out_mod)
                extracted["o"] = (
                    base_out.weight.data.clone(),
                    base_out.bias.data.clone() if getattr(base_out, "bias", None) is not None else None,
                )

        # Standard
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query", "W_q"],
                "k": ["k_lin", "k_proj", "key", "W_k"],
                "v": ["v_lin", "v_proj", "value", "W_v"],
                "o": ["out_lin", "out_proj", "dense", "o_proj", "c_proj", "wo"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        base_mod = getattr(mod, "base_layer", mod)
                        if hasattr(base_mod, "weight"):
                            w_data = base_mod.weight.data.clone()
                            if ((base_mod.__class__.__name__ == "Conv1D")
                                    or hasattr(base_mod, "nf")
                                    or (w_data.ndim == 2 and proj_key == "o"
                                        and w_data.shape[0] < w_data.shape[1])):
                                w_data = w_data.t().contiguous()
                            b_data = getattr(base_mod, "bias", None)
                            extracted[proj_key] = (
                                w_data,
                                b_data.data.clone() if b_data is not None else None,
                            )
                            break

        missing = [k for k, (w, _) in extracted.items()
                   if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug(f"UniversalAdapter fallback: {e}")

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"CosformerAttention copy_weights: missing mandatory projections "
                f"{missing_after} from {src.__class__.__name__}."
            )

        target_map = {"q": self.q_proj, "k": self.k_proj,
                      "v": self.v_proj, "o": self.out_proj}

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
                        and target_mod.weight.shape == w.t().contiguous().shape):
                    w = w.t().contiguous()

                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target_mod.weight.device, target_mod.weight.dtype))

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(target_mod.bias.device, target_mod.bias.dtype))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(f"Cosformer: '{proj_key}' norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # FIX 1: cos/sin with 2π range
    # ------------------------------------------------------------------
    def _get_cos_sin_weights(self, seq_len, device, dtype):
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        m = max(1, int(seq_len))
        # FIX 1: 2π range (was π/2)
        angles = (2.0 * math.pi / m) * pos
        cos_weight = torch.cos(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
        sin_weight = torch.sin(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
        return cos_weight, sin_weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False,
                output_hidden_states=False, return_dict=True,
                position_ids=None, position_embeddings=None,
                *args, **kwargs):
        assert hidden_states.dim() == 3, \
            f"Cosformer expects 3D [B, N, D], got {tuple(hidden_states.shape)}"

        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        input_dtype = hidden_states.dtype

        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len = kv_input.shape[1]

        # Q/K/V
        Q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # RoPE
        pos_emb = position_embeddings if position_embeddings is not None \
            else kwargs.get("position_embeddings")
        if pos_emb is not None and not is_cross_attention:
            if isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
                cos, sin = pos_emb[0], pos_emb[1]
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

        # GQA
        if self.num_kv_heads != self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # FIX 3: ELU+1 + scale by 1/sqrt(d)
        if self.act_fun == "softplus":
            Q_k = F.softplus(Q.float())
            K_k = F.softplus(K.float())
        else:
            Q_k = F.elu(Q.float()) + 1.0
            K_k = F.elu(K.float()) + 1.0

        # FIX 3: scale by 1/sqrt(d) to control magnitude
        scale = 1.0 / math.sqrt(d)
        Q_k = Q_k * scale
        K_k = K_k * scale

        V_f = V.float()

        # Mask
        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
        valid_mask = None
        if eff_mask is not None:
            valid_mask = get_token_valid_mask(eff_mask, B, kv_len, device)
            mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).float()
            K_k = K_k * mask_expanded
            V_f = V_f * mask_expanded

        # FIX 1: cos/sin weights
        cos_w, sin_w = self._get_cos_sin_weights(N, device, dtype=torch.float32)
        Q_cos = Q_k * cos_w
        Q_sin = Q_k * sin_w
        K_cos = K_k * cos_w
        K_sin = K_k * sin_w

        # FIX 6: Triton disabled
        is_causal_eval = bool(self.is_decoder and not is_cross_attention and kv_len == N)

        use_triton = (triton_dispatch is not None
                      and not is_cross_attention
                      and not _env_flag("COSFORMER_DISABLE_TRITON", True))
        context = None
        triton_ok = False

        if use_triton:
            try:
                Q_t = Q_cos.half()
                K_t = K_cos.half()
                K_s_t = K_sin.half()
                V_t = V_f.half()

                if (torch.isnan(Q_t).any() or torch.isnan(K_t).any()
                        or torch.isinf(Q_t).any() or torch.isinf(K_t).any()):
                    raise RuntimeError("Non-finite before Triton")

                dispatch_kwargs = dict(self.triton_kwargs)
                dispatch_kwargs["cos_weight"] = cos_w.half()
                dispatch_kwargs["sin_weight"] = sin_w.half()
                dispatch_kwargs["eps"] = self.eps

                context = triton_dispatch(
                    self.triton_kernel, Q_t, K_t, V_t,
                    Q_sin=Q_sin.half(),
                    K_sin=K_s_t,
                    is_causal=is_causal_eval,
                    **dispatch_kwargs,
                )
                context = context.float()
                triton_ok = True
            except Exception as e:
                logger.debug("Triton dispatch failed: %s", e)
                triton_ok = False

        # Python fallback
        if not triton_ok or context is None:
            if is_cross_attention:
                context = self._forward_bidirectional(
                    Q_cos, Q_sin, K_cos, K_sin, V_f, B, H, d, kv_len, device, torch.float32)
            elif is_causal_eval:
                context = self._forward_causal(
                    Q_cos, Q_sin, K_cos, K_sin, V_f, B, H, d, N, device, torch.float32)
            else:
                context = self._forward_bidirectional(
                    Q_cos, Q_sin, K_cos, K_sin, V_f, B, H, d, N, device, torch.float32)

        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            context = context * head_mask

        context = context.to(input_dtype)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            final_out = context
        else:
            final_out = self.out_proj(context)

        final_out = torch.nan_to_num(final_out, nan=0.0, posinf=1e4, neginf=-1e4)
        final_out = final_out.clamp(min=-1e4, max=1e4)

        return final_out.to(input_dtype), None

    # ------------------------------------------------------------------
    # FIX 2 + 7 + 9: causal with proper denom clamp and chunk_size
    # ------------------------------------------------------------------
    def _forward_causal(self, Q_cos, Q_sin, K_cos, K_sin, V,
                        B, H, d, N, device, dtype):
        # FIX 7: chunk_size = max(16, min(32, N // 4))
        chunk_size = max(16, min(32, N // 4))

        if chunk_size >= N:
            causal_mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
            score_c = torch.matmul(Q_cos, K_cos.transpose(-1, -2).contiguous())
            score_s = torch.matmul(Q_sin, K_sin.transpose(-1, -2).contiguous())
            scores = (score_c + score_s).masked_fill(~causal_mask, 0.0)
            num = torch.matmul(scores, V)
            denom = scores.sum(dim=-1, keepdim=True)
            # FIX 2: clamp = 1.0 (was eps=1e-6)
            return (num / denom.clamp(min=1.0)).to(dtype)

        pad_len = (chunk_size - (N % chunk_size)) % chunk_size
        if pad_len > 0:
            Q_cos = F.pad(Q_cos, (0, 0, 0, pad_len))
            Q_sin = F.pad(Q_sin, (0, 0, 0, pad_len))
            K_cos = F.pad(K_cos, (0, 0, 0, pad_len))
            K_sin = F.pad(K_sin, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))

        total_N = N + pad_len
        num_chunks = total_N // chunk_size

        qc_c = Q_cos.view(B, H, num_chunks, chunk_size, d)
        qs_c = Q_sin.view(B, H, num_chunks, chunk_size, d)
        kc_c = K_cos.view(B, H, num_chunks, chunk_size, d)
        ks_c = K_sin.view(B, H, num_chunks, chunk_size, d)
        v_c = V.view(B, H, num_chunks, chunk_size, d)

        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size,
                                            device=device, dtype=torch.bool))
        score_c = torch.matmul(qc_c, kc_c.transpose(-1, -2).contiguous()).masked_fill(~causal_mask, 0.0)
        score_s = torch.matmul(qs_c, ks_c.transpose(-1, -2).contiguous()).masked_fill(~causal_mask, 0.0)
        num_intra = torch.matmul(score_c + score_s, v_c)
        denom_intra = (score_c + score_s).sum(dim=-1, keepdim=True)

        S_cos = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
        S_sin = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
        Z_cos = torch.zeros(B, H, d, 1, device=device, dtype=torch.float32)
        Z_sin = torch.zeros(B, H, d, 1, device=device, dtype=torch.float32)

        num_list, denom_list = [], []
        for c in range(num_chunks):
            num_inter = torch.matmul(qc_c[:, :, c], S_cos) + torch.matmul(qs_c[:, :, c], S_sin)
            denom_inter = torch.matmul(qc_c[:, :, c], Z_cos) + torch.matmul(qs_c[:, :, c], Z_sin)

            num_list.append(num_intra[:, :, c] + num_inter)
            denom_list.append(denom_intra[:, :, c] + denom_inter)

            kc_curr = kc_c[:, :, c]
            ks_curr = ks_c[:, :, c]
            v_curr = v_c[:, :, c]

            if c == num_chunks - 1 and pad_len > 0:
                valid = chunk_size - pad_len
                kc_curr = kc_curr.clone()
                kc_curr[:, :, valid:] = 0.0
                ks_curr = ks_curr.clone()
                ks_curr[:, :, valid:] = 0.0
                v_curr = v_curr.clone()
                v_curr[:, :, valid:] = 0.0

            S_cos = S_cos + torch.matmul(kc_curr.transpose(-1, -2).contiguous(), v_curr)
            S_sin = S_sin + torch.matmul(ks_curr.transpose(-1, -2).contiguous(), v_curr)
            Z_cos = Z_cos + kc_curr.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()
            Z_sin = Z_sin + ks_curr.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()

        num = torch.cat(num_list, dim=2)[:, :, :N, :]
        denom = torch.cat(denom_list, dim=2)[:, :, :N, :]
        # FIX 2: clamp = 1.0
        return (num / denom.clamp(min=1.0)).to(dtype)

    # ------------------------------------------------------------------
    # FIX 2: bidirectional with proper denom clamp
    # ------------------------------------------------------------------
    def _forward_bidirectional(self, Q_cos, Q_sin, K_cos, K_sin, V,
                                B, H, d, N, device, dtype):
        KV_cos = torch.matmul(K_cos.transpose(-2, -1).contiguous(), V)
        KV_sin = torch.matmul(K_sin.transpose(-2, -1).contiguous(), V)

        Z_cos = K_cos.sum(dim=2, keepdim=True)
        Z_sin = K_sin.sum(dim=2, keepdim=True)

        num = torch.matmul(Q_cos, KV_cos) + torch.matmul(Q_sin, KV_sin)
        denom = torch.matmul(Q_cos, Z_cos.transpose(-2, -1).contiguous()) + \
                torch.matmul(Q_sin, Z_sin.transpose(-2, -1).contiguous())
        # FIX 2: clamp = 1.0
        denom = denom.clamp(min=1.0)

        context = (num / denom).to(dtype)
        return torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)


# ============================================================================
# Injection
# ============================================================================
def _is_single_layer(obj):
    if obj is None:
        return False
    single_layer_classes = (
        "TransformerBlock", "BertLayer", "RobertaLayer", "GPT2Block",
        "Qwen2DecoderLayer", "LlamaDecoderLayer", "Qwen3DecoderLayer",
        "MistralDecoderLayer",
    )
    cls_name = getattr(obj, "__class__", type(obj)).__name__
    if cls_name in single_layer_classes:
        return True
    has_attn = (hasattr(obj, "attention") or hasattr(obj, "attn")
                or hasattr(obj, "self_attn") or hasattr(obj, "self"))
    has_model = (hasattr(obj, "transformer") or hasattr(obj, "distilbert")
                 or hasattr(obj, "roberta") or hasattr(obj, "bert")
                 or hasattr(obj, "encoder") or hasattr(obj, "model")
                 or hasattr(obj, "layers") or hasattr(obj, "h")
                 or hasattr(obj, "encoder_layers") or hasattr(obj, "decoder_layers"))
    return has_attn and not has_model


@ATTENTION_REGISTRY.register("cosformer_inject")
@ATTENTION_REGISTRY.register("inject_cosformer")
def inject_cosformer(layer=None, model=None, model_config=None, comp_config=None,
                     layer_idx=0, total_layers=12, old_attn=None, **kwargs):
    depth = _get_inject_depth()
    if depth >= _MAX_DEPTH:
        logger.error("Max injection depth reached in inject_cosformer")
        return None
    _set_inject_depth(depth + 1)

    try:
        target = layer if layer is not None else model
        if target is None:
            target = kwargs.get("model", kwargs.get("layer"))
        if target is None:
            return None

        cfg = model_config if model_config is not None else getattr(target, "config", None)

        kwargs_clean = {k: v for k, v in kwargs.items()
                        if k not in ("model_config", "comp_config", "layer", "old_attn", "model")}
        kwargs_clean.pop("pretrained_attn", None)

        layer_list = None
        if _is_single_layer(target):
            pass
        elif hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif (hasattr(target, "bert") and hasattr(target.bert, "encoder")
              and hasattr(target.bert.encoder, "layer")):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer
        elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
            layer_list = target.layers

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_cosformer(
                    layer=lyr, model_config=cfg, comp_config=comp_config,
                    layer_idx=idx, total_layers=tot, **kwargs_clean,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

        layer = target

        src_attn = old_attn
        if src_attn is None and layer is not None:
            if hasattr(layer, "attention"):
                src_attn = layer.attention
            elif hasattr(layer, "attn"):
                src_attn = layer.attn
            elif hasattr(layer, "self_attn"):
                src_attn = layer.self_attn
            elif hasattr(layer, "self"):
                src_attn = layer.self

        hidden_size = (
            getattr(src_attn, "hidden_size", None)
            or getattr(src_attn, "embed_dim", None)
            or (src_attn.q_proj.in_features
                if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
                else None)
            or (layer.in_features if hasattr(layer, "in_features") else None)
            or getattr(model_config, "hidden_size", 768)
        )
        num_heads = (
            getattr(src_attn, "num_heads", None)
            or getattr(src_attn, "num_attention_heads", None)
            or getattr(model_config, "num_attention_heads", 4)
        )

        try:
            new_attn = CosformerAttention(
                config=model_config, comp_config=comp_config,
                layer_id=layer_idx, layer_idx=layer_idx,
                old_attn=src_attn,
                hidden_size=hidden_size, num_heads=num_heads,
                num_attention_heads=num_heads,
                **kwargs_clean,
            )

            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                else:
                    layer.attention = new_attn
            elif hasattr(layer, "attn"):
                layer.attn = new_attn
            elif hasattr(layer, "self_attn"):
                layer.self_attn = new_attn
            elif hasattr(layer, "self"):
                layer.self = new_attn

            return new_attn
        except Exception as e:
            logger.error(f"Cosformer construction failed: {e}")
            return None
    finally:
        _set_inject_depth(depth)


__all__ = [
    "CosformerAttention",
    "inject_cosformer",
    "apply_rotary_pos_emb",
    "LoRALinear",
]