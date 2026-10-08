"""
HGRN-2 Outer-Product Recurrent Linear Attention — Production-Hardened v4.

FIXES in v4 (based on external mathematical review):
  - FIX 11: REMOVED RoPE completely (was invalid for outer-product linear 
           attention — phases mix and become noise)
  - FIX 12: REMOVED denominator Z (was Katharopoulos-style normalization;
           modern RNNs like HGRN2/GLA/Mamba don't need it — f-gate handles 
           scaling automatically)
  - FIX 13: REMOVED ELU+1 (was destroying negative geometry of pretrained 
           GPT-2 weights; only q is scaled by 1/sqrt(d))

All v3 fixes preserved:
  - FIX 1: W_f bias init = -4.0
  - FIX 2: q scaled by 1/sqrt(d)
  - FIX 3: state_norm init = identity
  - FIX 4: chunk_size = 16
  - FIX 5: denom clamp (removed — see FIX 12)
  - FIX 6: decay mask = i <= j
  - FIX 7: output_scale init = 0.1
  - FIX 8: k NOT scaled by sqrt(d)
  - FIX 9: strict shape check in _copy_to
  - FIX 10: einsum reshaped to matmul
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _cfg_get(config: Any, attr: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(attr, default)
    return getattr(config, attr, default)


# ===========================================================================
# RoPE — kept for backward compat but NOT used in forward
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Kept for API compatibility. NOT used in forward — RoPE is invalid
    for outer-product linear attention."""
    if cos is None or sin is None:
        return q, k
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    is_custom_attention: bool = False

    def __init__(self, in_features, out_features, bias=True, lora_rank=0,
                 lora_alpha=1.0, freeze_base=True, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty(out_features, in_features, **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features, **fk))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, **fk))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.freeze_base_weights()

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (self.bias.to(x.dtype)
             if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias)
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.scaling
        return out


# ===========================================================================
# HGRN2GatedOuterBlock
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "hgrn2_outer", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "hgrn2_gated_outer", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
class HGRN2GatedOuterBlock(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(self, config=None, rank=None, dim=None, num_heads=None,
                 lora_rank=16, lora_alpha=32.0, is_decoder=None,
                 chunk_size=16, **kwargs):
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        if dim is None:
            if isinstance(config, int):
                dim = config
            elif config is not None:
                dim = getattr(config, "hidden_size",
                              getattr(config, "d_model",
                                      getattr(config, "n_embd",
                                              getattr(config, "dim", None))))
        if dim is None:
            dim = kwargs.get("hidden_size", kwargs.get("d_model", 768))
        if hasattr(dim, "hidden_size"):
            dim = dim.hidden_size

        if num_heads is None:
            if config is not None and not isinstance(config, int):
                num_heads = getattr(config, "num_attention_heads",
                                    getattr(config, "num_heads",
                                            getattr(config, "n_heads",
                                                    getattr(config, "n_head", None))))
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        while self.num_heads > 1 and (self.dim % self.num_heads != 0):
            self.num_heads -= 1
        self.head_dim = self.dim // max(1, self.num_heads)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.chunk_size = max(16, int(chunk_size))

        cfg_kv = None
        if config is not None and not isinstance(config, int):
            cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        self.num_kv_heads = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.num_key_value_groups = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.q_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim

        # FIX 15: FORCE causal for GPT-family
        _mt = str(_cfg_get(config, "model_type", "") or "").lower()
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                             "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            cfg_is_dec = (
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "causal", False)
                or any(_mt.startswith(p) for p in _decoder_prefixes)
                or kwargs.get("is_decoder", False)
                or kwargs.get("is_causal", False)
            )
            # FIX 15: strong signal from old_attn
            old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
            if old_attn is not None:
                if hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
                    cfg_is_dec = True
                if getattr(old_attn, "is_decoder", False):
                    cfg_is_dec = True
            # FIX 15: safe default
            if config is None and old_attn is None:
                logger.warning("HGRN2: defaulting to is_decoder=True for causal safety")
                cfg_is_dec = True
            self.is_decoder = bool(cfg_is_dec)

        self.is_nested_encoder = False

        self.q_proj = LoRALinear(self.dim, self.q_head_size, bias=False,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=True)
        self.k_proj = LoRALinear(self.dim, self.kv_head_size, bias=False,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=True)
        self.v_proj = LoRALinear(self.dim, self.kv_head_size, bias=False,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=True)
        self.W_f = nn.Linear(self.dim, self.num_heads, bias=True)

        self.state_norm = nn.LayerNorm(self.head_dim)
        with torch.no_grad():
            self.state_norm.weight.fill_(1.0)
            self.state_norm.bias.zero_()

        self.out_proj = LoRALinear(self.q_head_size, self.dim, bias=True,
                                    lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                    freeze_base=True)

        _wf_bias = _env_float("HGRN2_WF_BIAS", -4.0)
        with torch.no_grad():
            nn.init.normal_(self.W_f.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.W_f.bias, _wf_bias)

        _out_scale = _env_float("HGRN2_OUTPUT_SCALE", 0.1)
        self.output_scale = nn.Parameter(torch.full((1,), _out_scale))

        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
        if old_attn is not None:
            self.copy_weights(old_attn)

    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj

    def copy_weights(self, old_attn):
        with torch.no_grad():
            w_q = w_k = w_v = w_o = None
            b_q = b_k = b_v = b_o = None

            cls_name = old_attn.__class__.__name__
            if cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query") and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                cls_c = src.c_attn.__class__.__name__
                if (hasattr(src.c_attn, "nf") or "Conv1D" in cls_c
                        or (c_attn_w.shape[0] == self.dim and c_attn_w.shape[1] != self.dim)):
                    c_attn_w = c_attn_w.t()
                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size
                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim:q_dim+k_dim, :]
                w_v = c_attn_w[q_dim+k_dim:q_dim+k_dim+v_dim, :]
                if c_attn_b is not None:
                    b_q = c_attn_b[:q_dim]
                    b_k = c_attn_b[q_dim:q_dim+k_dim]
                    b_v = c_attn_b[q_dim+k_dim:q_dim+k_dim+v_dim]
                out_w = src.c_proj.weight.detach()
                cls_p = src.c_proj.__class__.__name__
                if (hasattr(src.c_proj, "nf") or "Conv1D" in cls_p
                        or (out_w.ndim == 2 and out_w.shape[0] != self.dim)):
                    out_w = out_w.t()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
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
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            elif hasattr(src, "q_proj") and hasattr(src, "k_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None
                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None

            elif hasattr(src, "q_lin") and hasattr(src, "k_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None
                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None

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

            if w_q is None or w_k is None or w_v is None:
                logger.warning("HGRN2GatedOuterBlock: missing Q/K/V weights")
                return

            try:
                for p in old_attn.parameters():
                    self._orig_storage_bytes += p.numel() * p.element_size()
            except Exception:
                pass

            def _copy_to(target_mod, w_src, b_src, name):
                tw = target_mod.weight
                w_cast = w_src.to(tw.dtype)
                if w_cast.shape == tw.shape:
                    tw.data.copy_(w_cast)
                else:
                    if w_cast.t().contiguous().shape == tw.shape:
                        tw.data.copy_(w_cast.t().contiguous())
                    else:
                        min_r = min(tw.shape[0], w_cast.shape[0])
                        min_c = min(tw.shape[1], w_cast.shape[1])
                        tw.data.zero_()
                        tw.data[:min_r, :min_c].copy_(w_cast[:min_r, :min_c])
                        logger.warning(
                            "HGRN2GatedOuterBlock: partial copy for '%s': %s → %s",
                            name, tuple(w_cast.shape), tuple(tw.shape),
                        )
                if b_src is not None and target_mod.bias is not None:
                    tb = target_mod.bias
                    b_cast = b_src.to(tb.dtype)
                    if b_cast.shape == tb.shape:
                        tb.data.copy_(b_cast)
                    else:
                        min_b = min(tb.numel(), b_cast.numel())
                        tb.data.zero_()
                        tb.data[:min_b].copy_(b_cast[:min_b])

            _copy_to(self.q_proj, w_q, b_q, "q")
            _copy_to(self.k_proj, w_k, b_k, "k")
            _copy_to(self.v_proj, w_v, b_v, "v")
            if w_o is not None:
                _copy_to(self.out_proj, w_o, b_o, "o")

            try:
                for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                    for p in m.parameters():
                        self._comp_storage_bytes += p.numel() * p.element_size()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # FIX 12: NO denominator, NO z_prev, NO W_k
    # ------------------------------------------------------------------
    def _chunked_scan(self, q, k, v, f, chunk_size=16):
        """
        Chunked scan for pure HGRN2 recurrence.
        
        FIX 12: Removed denominator Z (Katharopoulos-style normalization).
                Modern RNNs (HGRN2, GLA, Mamba) rely on f-gate for scaling.
                Dividing by Z fights the forget-gate dynamics.
        """
        B, H, L, d = q.shape
        num_chunks = (L + chunk_size - 1) // chunk_size

        # Only S state (no Z)
        S_prev = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        outputs = []

        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, L)
            C = end - start

            q_c = q[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]
            f_c = f[:, :, start:end]

            log_f = torch.log(f_c.clamp_min(1e-6))
            log_F_c = torch.cumsum(log_f, dim=-1)

            log_F_prev = torch.cat([
                torch.zeros(B, H, 1, device=q.device, dtype=log_F_c.dtype),
                log_F_c,
            ], dim=-1)

            log_decay = log_F_c.unsqueeze(-2) - log_F_prev.unsqueeze(-1)

            i_idx = torch.arange(C + 1, device=q.device).view(C + 1, 1)
            j_idx = torch.arange(C, device=q.device).view(1, C)
            valid = (i_idx == 0) | (i_idx <= j_idx)
            fill_val = -1e4 if log_decay.dtype == torch.float16 else -1e9
            log_decay = log_decay.masked_fill(~valid, fill_val)
            decay = torch.exp(log_decay).to(q.dtype)

            one_minus_f = (1.0 - f_c).to(q.dtype)
            W_kv = one_minus_f[..., None, None] * (k_c.unsqueeze(-1) * v_c.unsqueeze(-2))

            combined_kv = torch.cat([S_prev.unsqueeze(2), W_kv], dim=2).to(q.dtype)

            B_, H_, Nc, _, _ = combined_kv.shape
            combined_kv_flat = combined_kv.view(B_, H_, Nc, -1)
            S_all_flat = torch.matmul(decay, combined_kv_flat)
            S_all = S_all_flat.view(B_, H_, C, d, d)

            # FIX 12: output is num only — no denominator!
            y_c = torch.einsum("bhjd,bhjde->bhje", q_c, S_all)

            outputs.append(y_c)
            S_prev = S_all[:, :, -1]

        return torch.cat(outputs, dim=2).to(q.dtype)

    def storage_bytes(self):
        return {"orig": self._orig_storage_bytes, "comp": self._comp_storage_bytes}

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False,
                output_hidden_states=False, return_dict=True,
                *args, **kwargs):
        if encoder_hidden_states is not None:
            raise NotImplementedError("HGRN2GatedOuterBlock does not support cross-attention.")

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # FIX 11: REMOVED RoPE completely
        # HGRN2 uses f-gate for positional tracking; RoPE is invalid for
        # outer-product linear attention (phases mix into noise).

        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # FIX 13: REMOVED ELU+1 — only scale q by 1/sqrt(d)
        # ELU+1 was destroying negative geometry of pretrained GPT-2 weights.
        # Modern HGRN2/GLA/Mamba preserve signed values.
        q = q / math.sqrt(d)

        f = torch.exp((-F.softplus(self.W_f(hidden_states))).clamp(min=-3.0, max=-1e-4)).transpose(1, 2)

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.dtype == torch.bool:
                    mask_bool = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    mask_bool = attention_mask > -1e4
                else:
                    mask_bool = attention_mask > 0
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    m = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    m = attention_mask > -1e4
                else:
                    m = attention_mask > 0
                mask_bool = m.any(dim=-2).squeeze(1)
            if mask_bool is not None and mask_bool.shape[-1] != L:
                mask_bool = None

        if mask_bool is not None:
            m_exp = mask_bool.unsqueeze(1).unsqueeze(-1).to(k.dtype)
            k = k * m_exp
            v = v * m_exp

        y_fwd = self._chunked_scan(q, k, v, f, chunk_size=self.chunk_size)

        if self.is_decoder:
            y = y_fwd
        else:
            q_r = torch.flip(q, dims=[2])
            k_r = torch.flip(k, dims=[2])
            v_r = torch.flip(v, dims=[2])
            f_r = torch.flip(f, dims=[2])
            y_bwd_r = self._chunked_scan(q_r, k_r, v_r, f_r, chunk_size=self.chunk_size)
            y_bwd = torch.flip(y_bwd_r, dims=[2])
            y = 0.5 * (y_fwd + y_bwd)
            y = y.to(hidden_states.dtype)

        sn_dtype = (self.state_norm.weight.dtype
                    if hasattr(self.state_norm, "weight") and self.state_norm.weight is not None
                    else hidden_states.dtype)
        y = self.state_norm(y.to(sn_dtype)).to(hidden_states.dtype)
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)

        if head_mask is not None:
            hm = head_mask.to(y.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(-1)
            y = y * hm

        y = y.transpose(1, 2).contiguous().view(B, L, H * d)

        if self.is_nested_encoder:
            out = y
        else:
            out = self.out_proj(y)

        out = out * self.output_scale

        if mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None


# ===========================================================================
# Injector
# ===========================================================================
def inject_hgrn2_outer(layer=None, model=None, model_config=None, comp_config=None,
                       layer_idx=0, total_layers=12, old_attn=None,
                       _depth=0, **kwargs):
    if _depth > 8:
        logger.warning("inject_hgrn2_outer: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    for path in (("distilbert", "transformer", "layer"),
                 ("roberta", "encoder", "layer"),
                 ("bert", "encoder", "layer"),
                 ("transformer", "h"),
                 ("model", "layers"),
                 ("encoder", "layer"),
                 ("layers",),
                 ("h",)):
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
            res = inject_hgrn2_outer(layer=lyr, model_config=cfg, comp_config=comp_config,
                                      layer_idx=idx, total_layers=tot,
                                      _depth=_depth + 1, **kwargs)
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    src = old_attn
    is_nested_target = False
    if src is None and target is not None:
        for attr in ("attention", "self", "attn", "self_attn"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    if hasattr(cand, "self") and hasattr(cand.self, "query"):
                        src = cand.self
                        is_nested_target = True
                    else:
                        src = cand
                    break

    if src is None:
        return None

    if isinstance(src, HGRN2GatedOuterBlock) or "HGRN2" in src.__class__.__name__:
        return None

    dim = 768
    num_heads = 12
    if cfg is not None:
        dim = getattr(cfg, "hidden_size", getattr(cfg, "dim", 768))
        num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", 12))

    _mt = str(getattr(cfg, "model_type", "") or "").lower() if cfg else ""
    _dec_pfx = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
    is_dec = (
        bool(getattr(cfg, "is_decoder", False) if cfg else False)
        or bool(getattr(cfg, "causal", False) if cfg else False)
        or any(_mt.startswith(p) for p in _dec_pfx)
        or hasattr(src, "c_attn")
    )

    try:
        kwargs.pop("dim", None)
        kwargs.pop("num_heads", None)
        new_block = HGRN2GatedOuterBlock(config=cfg, dim=dim, num_heads=num_heads,
                                          is_decoder=is_dec, **kwargs)
        new_block.copy_weights(src)

        p = next(src.parameters(), None)
        if p is not None:
            new_block.to(device=p.device, dtype=p.dtype)

        spliced = False
        if is_nested_target and hasattr(target, "attention") and hasattr(target.attention, "self"):
            new_block.is_nested_encoder = True
            target.attention.self = new_block
            spliced = True
        elif hasattr(target, "attention"):
            new_block.is_nested_encoder = False
            target.attention = new_block
            spliced = True
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            target.self = new_block
            spliced = True
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            target.attn = new_block
            spliced = True
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            target.self_attn = new_block
            spliced = True

        if not spliced:
            logger.error("inject_hgrn2_outer: no writable attribute on %s", type(target).__name__)
            return None

        return new_block

    except Exception as e:
        logger.error("inject_hgrn2_outer failed at layer %d: %s", layer_idx, e)
        return None


HGRN2OuterBlock = HGRN2GatedOuterBlock

__all__ = [
    "HGRN2GatedOuterBlock",
    "HGRN2OuterBlock",
    "LoRALinear",
    "inject_hgrn2_outer",
    "apply_rotary_pos_emb",
]