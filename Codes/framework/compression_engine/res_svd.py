"""Spectral Residual Low-Rank SVD Attention (Res-SVD Attention) — v2.

W = U_base V_base^T + alpha * B A
where B A is initialized from the top-k residual singular directions of W - W_base.

Bug-fix map (v2):
  RES-01  Removed broken SDPA try/except with undefined vars.
  RES-02  copy_weights uses shape-aware init.
  RES-03  temp_linear constructed on same device/dtype as target.
  RES-04  Raise on missing mandatory weights.
  RES-05  RoPE dtype-safe + past_len offset.
  RES-06  Robust attention-mask helper.
  RES-07  Unified head_mask.
  RES-08  Recursion guard in inject_res_svd.
  RES-09  is_nested_encoder consistently set.
  RES-10  Guard bias before zero_().
  RES-11  sample_x on correct device.
  RES-12  No dangling self.config.
  RES-13  Cross-attention raises NotImplementedError.
  RES-14  rank always passed to super().
  RES-15  **KV-cache concat implemented for past_key_value (tuple + DynamicCache).**
  RES-16  **4D attention_mask is added directly to scores (preserves causal
           structure) instead of being reduced to a valid-token mask.**
  RES-17  **Safe base_U/weight device/dtype resolution with fallback.**
  RES-18  **Causal mask extended to N × (N + past_len) when cache is present.**
  RES-19  **Missing aliases added (o_proj, dense, c_proj, out_lin).**
  RES-20  **load_state_dict(strict=False) with explicit missing/unexpected
           key logging.**
  RES-21  **ResSVDLinear.from_linear guarded with try/except.**
  RES-22  **target_layer_idx used to filter which layers to inject.**
  RES-23  **RoPE slicing checks availability before slicing.**
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.compression_engine.res_svd import ResSVDLinear
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(q, k, cos, sin):
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
        raise ValueError(
            f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}"
        )

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut = cos[..., :head_dim].to(q.dtype)
        sin_cut = sin[..., :head_dim].to(q.dtype)
        q = (q * cos_cut) + (_rotate_half(q) * sin_cut)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# Helpers
# ===========================================================================
def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
    """RES-17: resolve device/dtype of the first available tensor in module."""
    for name in ("base_U", "base_V", "residual_A", "residual_B", "weight"):
        t = getattr(module, name, None)
        if isinstance(t, torch.Tensor):
            return t.device, t.dtype
    for p in module.parameters():
        return p.device, p.dtype
    return torch.device("cpu"), torch.float32


# ===========================================================================
# ResSVDAttention
# ===========================================================================
@ATTENTION_REGISTRY.register("res_svd")
@ATTENTION_REGISTRY.register("ressvd")
@ATTENTION_REGISTRY.register("res_svd_attention")
@ATTENTION_REGISTRY.register("spectral_residual_svd")
class ResSVDAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any,
        base_rank: Optional[int] = None,
        residual_rank: Optional[int] = None,
        rank: Optional[int] = 16,
        scaling: float = 1.0,
        factorize_base: bool = True,
        freeze_base: bool = True,
        dropout: Optional[float] = None,
        pretrained_attn: Optional[nn.Module] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        effective_rank = int(rank) if rank is not None else 16
        try:
            super().__init__(config, effective_rank)
        except TypeError:
            super().__init__(config)

        comp_config = getattr(config, "compression_config", None) if config else None

        self.hidden_size: int = int(getattr(config, "hidden_size",
                                    kwargs.get("hidden_size", 768)))
        self.dim = self.hidden_size
        self.num_heads: int = int(getattr(config, "num_attention_heads",
                                   kwargs.get("num_attention_heads", 12)))
        self.num_kv_heads: int = int(getattr(config, "num_key_value_heads",
                                      kwargs.get("num_key_value_heads", self.num_heads)))

        cfg_head_dim = getattr(config, "head_dim", None)
        kw_head_dim = kwargs.get("head_dim", None)
        inferred = cfg_head_dim or kw_head_dim
        if inferred is None and pretrained_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                if hasattr(pretrained_attn, attr):
                    layer_q = getattr(pretrained_attn, attr)
                    if hasattr(layer_q, "weight"):
                        inferred = layer_q.weight.shape[0] // max(1, self.num_heads)
                        break

        self.head_dim = int(inferred or (self.dim // max(1, self.num_heads)))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim

        self.base_rank: int = int(
            base_rank if base_rank is not None
            else getattr(comp_config, "base_rank", effective_rank)
        )
        self.residual_rank: int = int(
            residual_rank if residual_rank is not None
            else getattr(comp_config, "residual_rank", max(4, effective_rank // 4))
        )
        self.scaling = float(scaling)
        self.factorize_base = bool(factorize_base)
        self.freeze_base = bool(freeze_base)

        # Decoder detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            mt = str(getattr(config, "model_type", "") or "").lower()
            self.is_decoder = (
                bool(getattr(config, "is_decoder", False))
                or bool(getattr(config, "is_causal", False))
                or any(mt.startswith(p) for p in (
                    "gpt", "qwen", "llama", "mistral", "gemma",
                    "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))
                or (pretrained_attn is not None and hasattr(pretrained_attn, "c_attn"))
                or bool(kwargs.get("is_causal", False))
                or bool(kwargs.get("is_decoder", False))
            )
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # Projections
        self.query = ResSVDLinear(
            in_features=self.hidden_size,
            out_features=self.all_head_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )
        self.key = ResSVDLinear(
            in_features=self.hidden_size,
            out_features=self.all_kv_head_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )
        self.value = ResSVDLinear(
            in_features=self.hidden_size,
            out_features=self.all_kv_head_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )
        self.out_proj = ResSVDLinear(
            in_features=self.all_head_size,
            out_features=self.hidden_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )

        dropout_p = (
            float(dropout) if dropout is not None
            else float(getattr(config, "dropout",
                               getattr(config, "attention_probs_dropout_prob", 0.1)))
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        logger.info(
            "ResSVDAttention: hidden=%d, heads=%d, head_dim=%d, is_decoder=%s, "
            "base_rank=%d, residual_rank=%d",
            self.hidden_size, self.num_heads, self.head_dim,
            self.is_decoder, self.base_rank, self.residual_rank,
        )

    # ------------------------------------------------------------------
    # RES-19: aliases
    # ------------------------------------------------------------------
    @property
    def q_proj(self): return self.query
    @property
    def k_proj(self): return self.key
    @property
    def v_proj(self): return self.value
    @property
    def q_lin(self): return self.query
    @property
    def k_lin(self): return self.key
    @property
    def v_lin(self): return self.value
    @property
    def out_lin(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # RES-21: shape-aware weight transfer
    # ------------------------------------------------------------------
    def _init_from_weight(
        self, target: ResSVDLinear,
        w: torch.Tensor, b: Optional[torch.Tensor],
    ) -> None:
        # RES-17: safe device/dtype
        device, dtype = _module_device_dtype(target)

        with torch.no_grad():
            temp_linear = nn.Linear(
                target.in_features, target.out_features,
                bias=(b is not None),
                device=device, dtype=dtype,
            )
            temp_linear.weight.data.copy_(w.to(device=device, dtype=dtype))
            if b is not None and temp_linear.bias is not None:
                temp_linear.bias.data.copy_(b.to(device=device, dtype=dtype))

            try:
                new_layer = ResSVDLinear.from_linear(
                    temp_linear,
                    base_rank=self.base_rank,
                    residual_rank=self.residual_rank,
                    scaling=self.scaling,
                    factorize_base=self.factorize_base,
                    freeze_base=self.freeze_base,
                )
            except Exception as e:
                logger.error("ResSVDLinear.from_linear failed: %s", e)
                return

            # RES-20: strict=False with explicit logging
            missing, unexpected = target.load_state_dict(
                new_layer.state_dict(), strict=False,
            )
            if missing:
                logger.debug("ResSVD state_dict missing keys: %s", missing[:5])
            if unexpected:
                logger.debug("ResSVD state_dict unexpected keys: %s", unexpected[:5])

    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (hasattr(src, "self") and hasattr(getattr(src, "self"), "query")
                and hasattr(src, "output") and hasattr(getattr(src, "output"), "dense")):
            is_nested = True
        elif (cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                            "XLMRobertaSelfAttention", "ElectraSelfAttention",
                            "DebertaSelfAttention")
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
              ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or hasattr(c_attn_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv:
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split, :], c_b[:split] if c_b is not None else None)
                extracted["k"] = (c_w[split:2 * split, :], c_b[split:2 * split] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split:, :], c_b[2 * split:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or hasattr(c_proj_mod, "nf")
                        or hasattr(base_o, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # RoBERTa nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in (("q", "query"), ("k", "key"), ("v", "value")):
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

        # Flat modules
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
                logger.debug("UniversalAdapter failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"ResSVDAttention.copy_weights: missing {missing_after} "
                f"from {cls_name}. Refusing random init."
            )

        # Output projection
        if self.is_nested_encoder and extracted["o"][0] is None:
            # out_proj is bypassed in forward; use rectangular identity to keep
            # state_dict consistent (if any).
            eye = torch.eye(
                self.all_head_size, self.hidden_size,
                device=_module_device_dtype(self.out_proj)[0],
                dtype=_module_device_dtype(self.out_proj)[1],
            )
            self._init_from_weight(self.out_proj, eye, None)
        elif extracted["o"][0] is not None:
            self._init_from_weight(self.out_proj, extracted["o"][0], extracted["o"][1])

        # q, k, v with GQA resize
        for proj_key, target in (("q", self.query), ("k", self.key), ("v", self.value)):
            w, b = extracted[proj_key]
            if w is None:
                continue

            out_t = target.out_features
            in_t = target.in_features
            src_out, src_in = w.shape

            # GQA resize
            if proj_key in ("k", "v"):
                if src_out > out_t and src_out % out_t == 0:
                    rep = src_out // out_t
                    w = w.view(out_t, rep, -1).mean(dim=1)
                    if b is not None:
                        b = b.view(out_t, rep).mean(dim=1)
                elif src_out < out_t and out_t % src_out == 0:
                    rep = out_t // src_out
                    w = w.repeat(rep, 1)
                    if b is not None:
                        b = b.repeat(rep)

            # Shape alignment
            if w.shape[0] != out_t or w.shape[1] != in_t:
                padded = torch.zeros(out_t, in_t, dtype=w.dtype, device=w.device)
                r = min(out_t, w.shape[0])
                c = min(in_t, w.shape[1])
                padded[:r, :c] = w[:r, :c]
                w = padded

            self._init_from_weight(target, w, b)

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    # ------------------------------------------------------------------
    # RES-06: mask helper
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(attention_mask, B, N, device):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 3:
            if attention_mask.dtype == torch.bool:
                return attention_mask.squeeze(1)
            if attention_mask.dtype.is_floating_point:
                return (attention_mask > -1e4).squeeze(1)
            return (attention_mask > 0).squeeze(1)
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    # RES-15: KV-cache extraction helper
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_past_kv(past_key_value, layer_idx):
        if past_key_value is None:
            return None, None, 0
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            past_k, past_v = past_key_value[0], past_key_value[1]
            return past_k, past_v, past_k.shape[-2]
        if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
            idx = min(layer_idx, len(past_key_value.key_cache) - 1)
            past_k = past_key_value.key_cache[idx]
            past_v = past_key_value.value_cache[idx]
            return past_k, past_v, past_k.shape[-2]
        return None, None, 0

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
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "ResSVDAttention: cross-attention (encoder_hidden_states) is not supported."
            )

        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.query(hidden_states).view(B, N, H, D).permute(0, 2, 1, 3)
        K_new = self.key(hidden_states).view(B, N, self.num_kv_heads, D).permute(0, 2, 1, 3)
        V_new = self.value(hidden_states).view(B, N, self.num_kv_heads, D).permute(0, 2, 1, 3)

        # ---- RoPE with past_len offset (RES-05 + RES-23) ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)

        past_k, past_v, past_len = self._extract_past_kv(past_key_value, layer_idx)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            # RES-23: safe slicing
            if cos.shape[-2] >= past_len + N:
                cos_new = cos[..., past_len: past_len + N, :]
                sin_new = sin[..., past_len: past_len + N, :]
            elif cos.shape[-2] == N:
                # Already sliced to new positions
                cos_new, sin_new = cos, sin
            else:
                raise ValueError(
                    f"RoPE mismatch: cos has seq={cos.shape[-2]}, "
                    f"but past_len+N = {past_len + N}"
                )
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos_new, sin_new)

        # ---- KV-cache concat (RES-15) ----
        if past_k is not None and past_v is not None:
            # Broadcast-cat along key dim: past_k may be (B, H_kv, past_len, D)
            K_cat = torch.cat([past_k, K_new], dim=-2)
            V_cat = torch.cat([past_v, V_new], dim=-2)
        else:
            K_cat = K_new
            V_cat = V_new

        N_kv = K_cat.shape[-2]

        # GQA expansion
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            K_full = repeat_kv(K_cat, repeats)
            V_full = repeat_kv(V_cat, repeats)
        else:
            K_full = K_cat
            V_full = V_cat

        scale = D ** -0.5
        scores = torch.matmul(Q.float(), K_full.float().transpose(-2, -1).contiguous()) * scale

        # ---- Causal mask with past_len (RES-18) ----
        if self.is_decoder and N > 1:
            if past_len == 0:
                causal = torch.triu(
                    torch.full((N, N_kv), float("-inf"), device=device, dtype=torch.float32),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)
            else:
                abs_q = (past_len + torch.arange(N, device=device)).unsqueeze(1)
                abs_k = torch.arange(N_kv, device=device).unsqueeze(0)
                causal = torch.where(
                    abs_k <= abs_q,
                    torch.zeros((N, N_kv), device=device, dtype=torch.float32),
                    torch.full((N, N_kv), float("-inf"), device=device, dtype=torch.float32),
                ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal

        # ---- Attention mask (RES-16: preserve 4D causal) ----
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                # Add directly to scores. Shapes: (B, H, N_q, N_kv) or (B, 1, N_q, N_kv)
                mask4 = attention_mask
                # If N_kv != attention_mask.shape[-1], try to slice matching tail
                if mask4.shape[-1] > N_kv:
                    mask4 = mask4[..., -N_kv:]
                elif mask4.shape[-1] < N_kv:
                    # Pad with zeros on the left
                    pad = torch.zeros(
                        *mask4.shape[:-1], N_kv - mask4.shape[-1],
                        device=mask4.device, dtype=mask4.dtype,
                    )
                    mask4 = torch.cat([pad, mask4], dim=-1)
                # If N_q != attention_mask.shape[-2], slice matching tail
                if mask4.shape[-2] > N:
                    mask4 = mask4[..., -N:, :]
                elif mask4.shape[-2] < N:
                    pad = torch.zeros(
                        *mask4.shape[:-2], N - mask4.shape[-2], mask4.shape[-1],
                        device=mask4.device, dtype=mask4.dtype,
                    )
                    mask4 = torch.cat([pad, mask4], dim=-2)
                scores = scores + mask4.to(scores.dtype)
            else:
                valid_mask = self._prepare_valid_mask(attention_mask, B, N_kv, device)
                if valid_mask is not None:
                    scores = scores.masked_fill(
                        ~valid_mask[:, None, None, :], -1e4
                    )

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # ---- head_mask (RES-07) ----
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                # Could be (num_layers, num_heads) or (B, H)
                if head_mask.shape[1] == self.num_heads:
                    head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
                elif head_mask.shape[0] == self.num_heads:
                    head_mask = head_mask.t().unsqueeze(0).unsqueeze(-1)
            elif head_mask.dim() == 3:
                # (B, H, N) → mask per head; reduce over N
                head_mask = head_mask.mean(dim=-1, keepdim=True).unsqueeze(-1)
            if head_mask is not None and head_mask.size(1) == self.num_heads:
                attn = attn * head_mask.to(attn.dtype)

        context = torch.matmul(attn, V_full)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        return output, (attn if output_attentions else None)


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("res_svd_inject")
@ATTENTION_REGISTRY.register("inject_res_svd")
@ATTENTION_REGISTRY.register("inject_res_svd_attention")
def inject_res_svd(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    target_layer_idx: Optional[int] = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_res_svd: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
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
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
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
        # RES-22: filter by target_layer_idx if specified
        for idx, lyr in enumerate(layer_list):
            if target_layer_idx is not None and idx != target_layer_idx:
                continue
            res = inject_res_svd(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot,
                _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # Single-layer injection
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = cand
                        break
        if src is None:
            src = target

        if isinstance(src, ResSVDAttention) or "ResSVDAttention" in src.__class__.__name__:
            return None

        new_attn = ResSVDAttention(
            config=cfg or getattr(src, "config", None),
            pretrained_attn=src if src is not target else None,
            **kwargs,
        )

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
        return new_attn
    except Exception as e:
        logger.exception("inject_res_svd failed at layer %d: %s", layer_idx, e)
        return None


inject_res_svd_attention = inject_res_svd


__all__ = [
    "ResSVDAttention",
    "inject_res_svd",
    "inject_res_svd_attention",
]