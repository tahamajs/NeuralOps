"""MambaSVD-Flash: Hybrid Bidirectional SSM-FlashAttention Module (Production-Hardened v2).

Fuses bidirectional selective state-space dynamics with online low-rank SVD
of state transitions, coupled with a Multi-Head / GQA FlashAttention path.

"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
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
        q = torch.cat([(q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass], dim=-1)
        k = torch.cat([(k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ===========================================================================
# LoRALinear (dtype-safe)
# ===========================================================================
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

    def unfreeze_base_weights(self):
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x):
        # MSF-FIX-09: dtype-safe
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register("mamba_svd_flash", supports_causal=True, supports_gqa=True)
class MambaSVDFlash(BaseAttention):
    """
    Hybrid bidirectional selective SSM + multi-head attention.

    The SSM path is a low-rank factorized state transition:
        h_t = U_t · diag(σ_t) · V_t · h_{t-1} + B_t · x_t
        y_t = C_t · h_t
    with per-token U_t, σ_t, V_t, B_t, C_t predicted from x_t.

    The attention path is standard MHA / GQA with optional RoPE.
    A learnable gate mixes the two paths.
    """
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 16,
        d_model: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        comp_config: Optional[Any] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # MSF-FIX-18: config None-safe
        dim = (kwargs.get("d_model") or kwargs.get("hidden_size")
               or kwargs.get("dim") or d_model or hidden_size)
        if dim is None:
            if config is not None:
                if isinstance(config, int):
                    dim = config
                elif isinstance(config, dict):
                    dim = config.get("hidden_size", config.get("dim", 128))
                else:
                    dim = getattr(config, "hidden_size", getattr(config, "dim", 128))
            elif source_attn is not None:
                cand = getattr(source_attn, "hidden_size",
                                getattr(source_attn, "embed_dim",
                                        getattr(source_attn, "all_head_size", None)))
                if cand is None and hasattr(source_attn, "q_proj"):
                    q = getattr(source_attn, "q_proj", None)
                    if q is not None and hasattr(q, "in_features"):
                        cand = q.in_features
                dim = cand or 128

        config_obj = None if isinstance(config, int) else config
        super().__init__(config_obj, rank if rank is not None else 16)

        self.d_model = int(dim or 128)
        self.dim = self.d_model
        self.hidden_size = self.d_model

        # MSF-FIX-02: respect the `rank` parameter
        self.rank = max(1, int(rank if rank is not None else 16))

        # ---- LoRA config ----
        if config_obj is not None:
            self.lora_rank = int(getattr(config_obj, "lora_rank",
                                  getattr(comp_config, "lora_rank",
                                          kwargs.get("lora_rank", 16))))
            self.lora_alpha = float(getattr(config_obj, "lora_alpha",
                                     getattr(comp_config, "lora_alpha",
                                             kwargs.get("lora_alpha", 32.0))))
        elif comp_config is not None:
            self.lora_rank = int(getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            self.lora_alpha = float(getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))
        self.freeze_base = (self.lora_rank > 0)

        # ---- decoder detection ----
        model_type = (str(getattr(config_obj, "model_type", "") or "").lower()
                       if config_obj is not None else "")
        is_dec = False
        if config_obj is not None:
            is_dec = bool(getattr(config_obj, "is_decoder", False)
                          or getattr(config_obj, "is_causal", False))
        if any(model_type.startswith(p) for p in
               ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt",
                "bloom", "falcon", "olmo", "mpt", "stablelm")):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        # Also detect via source attention
        if source_attn is not None:
            if hasattr(source_attn, "c_attn") or getattr(source_attn, "is_decoder", False):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # ---- heads and GQA ----
        n_heads = num_heads or kwargs.get("n_heads") or kwargs.get("num_attention_heads")
        if n_heads is None:
            if config_obj is not None:
                n_heads = getattr(config_obj, "num_attention_heads",
                                   getattr(config_obj, "n_heads", 12))
            elif source_attn is not None:
                n_heads = getattr(source_attn, "num_attention_heads",
                                   getattr(source_attn, "num_heads", 12))
            else:
                n_heads = 12
        self.num_heads = int(n_heads)
        if self.d_model % self.num_heads != 0:
            logger.warning("MambaSVDFlash: d_model %% num_heads != 0; forcing num_heads=1")
            self.num_heads = 1
        self.head_dim = self.d_model // max(1, self.num_heads)

        cfg_kv = None
        if config_obj is not None:
            cfg_kv = getattr(config_obj, "num_key_value_heads",
                              getattr(config_obj, "num_kv_heads", None))
        if cfg_kv is None:
            cfg_kv = kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = max(1, min(int(cfg_kv), self.num_heads))
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        # ---- SSM low-rank projections ----
        self.proj_u = nn.Linear(self.d_model, self.d_model * self.rank, bias=False)
        self.proj_sigma = nn.Linear(self.d_model, self.rank, bias=True)
        self.proj_v = nn.Linear(self.d_model, self.rank * self.d_model, bias=False)
        self.proj_b = LoRALinear(self.d_model, self.d_model, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.proj_c = LoRALinear(self.d_model, self.d_model, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)

        # ---- attention projections ----
        self.q_proj = LoRALinear(self.d_model, self.all_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.d_model, self.all_kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.d_model, self.all_kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.d_model, bias=True,
                                    lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                    freeze_base=self.freeze_base)

        # ---- gating ----
        self.gating = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.gating.weight)
        nn.init.constant_(self.gating.bias, -2.0)

        # ---- MSF-FIX-08: learnable bidirectional mixing ----
        self.bidirectional_mix = nn.Parameter(torch.tensor(0.0))

        # sigma bias: contractive stability (singular values ~0.85)
        nn.init.constant_(self.proj_sigma.bias, 1.8)

        # ---- MSF-FIX-01: define dropout ----
        dropout_p = 0.1
        if config_obj is not None:
            dropout_p = float(getattr(config_obj, "dropout",
                              getattr(config_obj, "attention_probs_dropout_prob", 0.1)))
        elif comp_config is not None:
            dropout_p = float(getattr(comp_config, "dropout", 0.1))
        else:
            dropout_p = float(kwargs.get("dropout", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        self.source_attn = source_attn
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        # MSF-FIX-11: storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ---- compatibility aliases ----
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj

    # ------------------------------------------------------------------
    def freeze_base_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj,
                   self.proj_b, self.proj_c):
            m.freeze_base_weights()

    def unfreeze_base_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj,
                   self.proj_b, self.proj_c):
            m.unfreeze_base_weights()

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        if old_attn is None:
            return

        try:
            for p in old_attn.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        cls_name = old_attn.__class__.__name__
        has_out_proj = any(hasattr(old_attn, a) for a in
                            ("out_proj", "out_lin", "c_proj", "dense", "o_proj"))
        is_nested = False
        if (hasattr(old_attn, "self") and hasattr(getattr(old_attn, "self"), "query")
                and hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense")):
            is_nested = True
        elif (cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                            "XLMRobertaSelfAttention", "ElectraSelfAttention",
                            "DebertaSelfAttention")
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())) \
                and not has_out_proj:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # 1. GPT-2 fused c_attn / c_proj
        if hasattr(old_attn, "c_attn") and old_attn.c_attn is not None:
            c = old_attn.c_attn
            base_c = getattr(c, "base_layer", c)
            cw = getattr(base_c, "weight", getattr(c, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                is_conv = (c.__class__.__name__ == "Conv1D"
                            or base_c.__class__.__name__ == "Conv1D"
                            or (cw.ndim == 2 and cw.shape[0] < cw.shape[1]))
                if is_conv:
                    cw = cw.t().contiguous()
                split = cw.size(0) // 3
                bmod = getattr(base_c, "bias", getattr(c, "bias", None))
                cb = bmod.data.clone() if bmod is not None else None
                extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
                extracted["k"] = (cw[split:2*split], cb[split:2*split] if cb is not None else None)
                extracted["v"] = (cw[2*split:], cb[2*split:] if cb is not None else None)

            if hasattr(old_attn, "c_proj") and old_attn.c_proj is not None:
                o = old_attn.c_proj
                base_o = getattr(o, "base_layer", o)
                ow = getattr(base_o, "weight", getattr(o, "weight", None))
                if ow is not None:
                    ow = ow.data.clone()
                    is_conv_o = (o.__class__.__name__ == "Conv1D"
                                  or base_o.__class__.__name__ == "Conv1D"
                                  or (ow.ndim == 2 and ow.shape[0] < ow.shape[1]))
                    if is_conv_o:
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # 2. Nested (BERT/RoBERTa)
        elif hasattr(old_attn, "self") and hasattr(getattr(old_attn, "self"), "query"):
            s = old_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (m.weight.data, b.data if b is not None else None)
            if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
                d = old_attn.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (d.weight.data, b.data if b is not None else None)

        # 3. Flat modules
        else:
            cand = {
                "q": ("q_proj", "q_lin", "query"),
                "k": ("k_proj", "k_lin", "key"),
                "v": ("v_proj", "v_lin", "value"),
                "o": ("out_proj", "out_lin", "dense", "o_proj"),
            }
            for key, names in cand.items():
                for n in names:
                    if hasattr(old_attn, n):
                        m = getattr(old_attn, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (m.weight.data, b.data if b is not None else None)
                            break

        # 4. UniversalAdapter fallback
        missing = [k for k, (w, _) in extracted.items()
                    if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(old_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                          if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning("MambaSVDFlash: missing projections %s from %s",
                            missing_after, cls_name)

        target_map = {"q": self.q_proj, "k": self.k_proj,
                       "v": self.v_proj, "o": self.out_proj}

        with torch.no_grad():
            for key, target in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if w.shape != target.weight.shape \
                        and w.t().contiguous().shape == target.weight.shape:
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data[:mb].copy_(
                        b[:mb].to(target.bias.device, target.bias.dtype)
                    )
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                if self.freeze_base:
                    target.freeze_base_weights()

            # Initialize SSM B, C from K, V
            w_k, b_k = extracted["k"]
            if w_k is not None:
                out_b = min(self.proj_b.weight.size(0), w_k.size(0))
                in_b = min(self.proj_b.weight.size(1), w_k.size(1))
                self.proj_b.weight.data[:out_b, :in_b].copy_(
                    w_k[:out_b, :in_b].to(self.proj_b.weight.device, self.proj_b.weight.dtype)
                )
                if b_k is not None and self.proj_b.bias is not None:
                    mbk = min(self.proj_b.bias.numel(), b_k.numel())
                    self.proj_b.bias.data[:mbk].copy_(b_k[:mbk])
                if self.freeze_base:
                    self.proj_b.freeze_base_weights()

            w_v, b_v = extracted["v"]
            if w_v is not None:
                out_c = min(self.proj_c.weight.size(0), w_v.size(0))
                in_c = min(self.proj_c.weight.size(1), w_v.size(1))
                self.proj_c.weight.data[:out_c, :in_c].copy_(
                    w_v[:out_c, :in_c].to(self.proj_c.weight.device, self.proj_c.weight.dtype)
                )
                if b_v is not None and self.proj_c.bias is not None:
                    mbv = min(self.proj_c.bias.numel(), b_v.numel())
                    self.proj_c.bias.data[:mbv].copy_(b_v[:mbv])
                if self.freeze_base:
                    self.proj_c.freeze_base_weights()

        # Track compressed storage
        try:
            for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj,
                       self.proj_b, self.proj_c, self.proj_u, self.proj_v,
                       self.proj_sigma):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        if hasattr(old_attn, "rotary_emb") and old_attn.rotary_emb is not None:
            self.rotary_emb = old_attn.rotary_emb

        if self.lora_rank > 0 or self.freeze_base:
            self.freeze_base_weights()

    # ------------------------------------------------------------------
    # SVD-SSM scan (chunked sequential, but bounded)
    # ------------------------------------------------------------------
    def _svd_ssm_scan(
        self,
        U: torch.Tensor,
        Sigma: torch.Tensor,
        V: torch.Tensor,
        B_seq: torch.Tensor,
        C_seq: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """
        Sequential low-rank SSM scan. Bounded by chunk_size for memory.

        Recurrence:
            v_t = V_t @ h_{t-1}          # (B, r)
            h_t = U_t @ diag(σ_t) @ v_t + B_t * x_t
            y_t = C_t * h_t
        """
        b, l, d = hidden_states.shape
        if l == 0:
            return torch.zeros_like(hidden_states)
        if l == 1:
            # Single token: no recurrence contribution
            bx = (B_seq * hidden_states).float()
            return (C_seq.float() * bx).to(hidden_states.dtype)

        # MSF-FIX-12: normalize U over the rank dimension (dim=-2), V over rank dim (dim=-2)
        U_norm = F.normalize(U.float(), p=2, dim=-2, eps=1e-8)  # (B,L,D,r)
        V_norm = F.normalize(V.float(), p=2, dim=-2, eps=1e-8)  # (B,L,r,D)
        US = U_norm * Sigma.float().unsqueeze(-2)                # (B,L,D,r) * (B,L,1,r)
        bx = (B_seq * hidden_states).float()                     # (B,L,D)

        chunk_size = max(16, min(64, int(chunk_size)))
        num_chunks = (l + chunk_size - 1) // chunk_size

        h = torch.zeros(b, d, device=hidden_states.device, dtype=torch.float32)
        outputs: list = []

        rng = range(num_chunks) if not reverse else range(num_chunks - 1, -1, -1)
        for c in rng:
            s_idx = c * chunk_size
            e_idx = min(l, s_idx + chunk_size)
            step_range = (range(s_idx, e_idx) if not reverse
                           else range(e_idx - 1, s_idx - 1, -1))

            chunk_out: Dict[int, torch.Tensor] = {}
            for t in step_range:
                v_t = torch.matmul(V_norm[:, t], h.unsqueeze(-1)).squeeze(-1)   # (B, r)
                h_trans = torch.matmul(US[:, t], v_t.unsqueeze(-1)).squeeze(-1) # (B, D)
                h_next = h_trans + bx[:, t]
                if mask is not None:
                    m_t = mask[:, t].view(b, 1).to(h_next.dtype)
                    h = torch.where(m_t.bool(), h_next, h)
                else:
                    h = h_next
                h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
                chunk_out[t] = h

            if reverse:
                for t in sorted(chunk_out.keys()):
                    outputs.append(chunk_out[t])
            else:
                for t in sorted(chunk_out.keys()):
                    outputs.append(chunk_out[t])

        H = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        return torch.nan_to_num(C_seq * H, nan=0.0)

    # ------------------------------------------------------------------
    # Mask normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_valid_mask(
        attention_mask: Optional[torch.Tensor],
        B: int, L: int, device: torch.device,
    ) -> torch.Tensor:
        """
        MSF-FIX-04 / MSF-FIX-13: single path, always returns (B, L) bool.
        Defaults to all-True.
        """
        if attention_mask is None:
            return torch.ones(B, L, device=device, dtype=torch.bool)

        def _to_bool(t):
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if attention_mask.dim() == 2:
            m = _to_bool(attention_mask)
            if m.shape == (B, L):
                return m
        elif attention_mask.dim() == 3:
            m = _to_bool(attention_mask)
            if m.shape[1] == L:
                return m.any(dim=0) if m.shape[0] != B else m[0] if m.shape[0] == 1 else m[:, -1, :]
        elif attention_mask.dim() == 4:
            # (B, 1, L, L) or (B, H, L, L)
            m = _to_bool(attention_mask)
            # Reduce over the second-to-last dim (queries) — a valid key is one
            # that at least one query can attend to.
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m.squeeze(1)
            if m.shape == (B, L):
                return m

        # Fallback: use framework helper
        try:
            m = get_token_valid_mask(attention_mask, B, L, device)
            if isinstance(m, torch.Tensor) and m.shape == (B, L):
                return m.bool()
        except Exception:
            pass

        return torch.ones(B, L, device=device, dtype=torch.bool)

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
        if encoder_hidden_states is not None:
            raise NotImplementedError("MambaSVDFlash does not support cross-attention.")

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        while isinstance(attention_mask, (tuple, list)):
            attention_mask = attention_mask[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        dtype_in = hidden_states.dtype

        # MSF-FIX-13: normalize mask once
        mask_bool = self._normalize_valid_mask(attention_mask, B, L, hidden_states.device)

        # ---------------- 1. SSM path ----------------
        U = self.proj_u(hidden_states).view(B, L, D, self.rank)
        Sigma = torch.sigmoid(self.proj_sigma(hidden_states)).clamp(min=1e-6) * 0.95
        V = self.proj_v(hidden_states).view(B, L, self.rank, D)
        B_t_seq = self.proj_b(hidden_states)
        C_t_seq = self.proj_c(hidden_states)

        if self.is_decoder:
            ssm_out = self._svd_ssm_scan(U, Sigma, V, B_t_seq, C_t_seq,
                                          hidden_states, mask=mask_bool,
                                          reverse=False, chunk_size=64)
        else:
            ssm_fwd = self._svd_ssm_scan(U, Sigma, V, B_t_seq, C_t_seq,
                                          hidden_states, mask=mask_bool,
                                          reverse=False, chunk_size=64)
            ssm_bwd = self._svd_ssm_scan(U, Sigma, V, B_t_seq, C_t_seq,
                                          hidden_states, mask=mask_bool,
                                          reverse=True, chunk_size=64)
            # MSF-FIX-08: learnable mix
            w = torch.sigmoid(self.bidirectional_mix)
            ssm_out = w * ssm_fwd + (1 - w) * ssm_bwd

        if isinstance(ssm_out, tuple):
            ssm_out = ssm_out[0]

        # ---------------- 2. Attention path ----------------
        Q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V_attn = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        elif self.rotary_emb is not None:
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = self.rotary_emb(K, seq_len=L)
            except TypeError:
                try:
                    cos, sin = self.rotary_emb(K, pos_ids)
                except Exception:
                    cos, sin = None, None
            if cos is not None and sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=pos_ids)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V_attn = repeat_kv(V_attn, self.kv_repeat)

        scale = 1.0 / math.sqrt(max(1, self.head_dim))
        attn_out: Optional[torch.Tensor] = None
        probs: Optional[torch.Tensor] = None

        # Fast path: SDPA when no mask / head_mask / output_attentions
        if head_mask is None and not output_attentions and mask_bool.all():
            try:
                _qs = Q.half() if Q.dtype == torch.float32 else Q
                _ks = K.half() if K.dtype == torch.float32 else K
                _vs = V_attn.half() if V_attn.dtype == torch.float32 else V_attn
                _is_causal = self.is_decoder and L > 1
                _sdpa_out = F.scaled_dot_product_attention(
                    _qs, _ks, _vs, is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                attn_out = _sdpa_out.to(dtype_in).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
                attn_out = torch.nan_to_num(attn_out, nan=0.0)
            except Exception:
                attn_out = None

        if attn_out is None:
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

            if self.is_decoder:
                causal = torch.triu(
                    torch.full((L, L), -1e4, device=hidden_states.device, dtype=torch.float32),
                    diagonal=1,
                )
                scores = scores + causal

            # MSF-FIX-04: mask keys via (B, L) bool
            if not mask_bool.all():
                scores = scores.masked_fill(~mask_bool[:, None, None, :], -1e4)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(dtype_in)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = self.dropout(probs)

            # MSF-FIX-14: head_mask uniform
            if head_mask is not None:
                hm = head_mask.to(probs.dtype)
                if hm.dim() == 1:
                    hm = hm.view(1, -1, 1, 1)
                elif hm.dim() == 2:
                    hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
                elif hm.dim() == 3:
                    hm = hm.unsqueeze(1)
                probs = probs * hm

            attn_out = torch.matmul(probs, V_attn).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
            attn_out = torch.nan_to_num(attn_out, nan=0.0)

        # ---------------- 3. Gated fusion ----------------
        gamma = torch.sigmoid(self.gating(hidden_states))
        Y = gamma * ssm_out + (1.0 - gamma) * attn_out
        Y = torch.nan_to_num(Y, nan=0.0, posinf=1e4, neginf=-1e4)

        out = Y if self.is_nested_encoder else self.out_proj(Y)

        # MSF-FIX-05: only apply final mask if we didn't already do it in scores
        # (when the fast path was used, probs was None and masking hasn't been
        #  applied to output yet)
        if probs is None and not mask_bool.all():
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, (probs if output_attentions else None)

    # ------------------------------------------------------------------
    def storage_bytes(self) -> Dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (("q_proj", self.q_proj), ("k_proj", self.k_proj),
                         ("v_proj", self.v_proj), ("out_proj", self.out_proj),
                         ("proj_b", self.proj_b), ("proj_c", self.proj_c),
                         ("proj_u", self.proj_u), ("proj_v", self.proj_v),
                         ("proj_sigma", self.proj_sigma),
                         ("gating", self.gating)):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("mamba_svd_flash_inject")
@ATTENTION_REGISTRY.register("inject_mamba_svd_flash")
def inject_mamba_svd_flash(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Optional[Union[MambaSVDFlash, nn.Module]]:
    if _depth > 8:
        logger.warning("inject_mamba_svd_flash: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # ---- Whole-model traversal ----
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
            res = inject_mamba_svd_flash(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # ---- Single-layer ----
    src_attn = old_attn
    if src_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    # MSF-FIX-10: idempotency guard
    if src_attn is not None and (isinstance(src_attn, MambaSVDFlash)
                                   or "MambaSVDFlash" in src_attn.__class__.__name__):
        return None
    if src_attn is not None and ("DeltaNet" in src_attn.__class__.__name__
                                   or "LinearAttention" in src_attn.__class__.__name__):
        return None

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None) or
        getattr(src_attn, "embed_dim", None) or
        getattr(src_attn, "dim", None) or
        (src_attn.q_proj.in_features if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features") else None) or
        (src_attn.q_lin.in_features if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features") else None) or
        (getattr(target, "in_features", None) if target is not None else None) or
        getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None) or
        getattr(src_attn, "num_attention_heads", None) or
        getattr(src_attn, "n_heads", None) or
        getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))
    )

    try:
        new_block = MambaSVDFlash(
            config=cfg,
            rank=kwargs.pop("rank", 16),
            comp_config=comp_config,
            old_attn=src_attn,
            hidden_size=hidden_size,
            d_model=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
    except Exception as e:
        logger.error("Failed to instantiate MambaSVDFlash: %s", e)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_block.is_nested_encoder = True
                target.attention.self = new_block
            else:
                new_block.is_nested_encoder = False
                target.attention = new_block
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            target.self = new_block
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            target.self_attn = new_block

    return new_block


__all__ = [
    "MambaSVDFlash",
    "inject_mamba_svd_flash",
    "apply_rotary_pos_emb",
    "LoRALinear",
]