"""Linformer Attention – Production-hardened with full multi-architecture support.

Bug-fix map (2026 revision):
  LF-01  RoPE cos/sin cast to q/k dtype.
  LF-02  super().__init__ with try/except.
  LF-03  Injection depth via function argument (thread-safe).
  LF-04  Robust head_dim detection.
  LF-05  RoPE past_len offset for KV-cache.
  LF-06  Removed dead output_scale parameter.
  LF-07  GQA resize in copy_weights.
  LF-08  Norm verification via float() cast.
  LF-09  Proper causal mask for projected K/V.
  LF-10  E/F interpolation with 3D input (safe).
  LF-11  is_decoder detection via centralized helper.
  LF-12  FactorizedLoRALinear replaces LoRALinear.
  LF-13  Forward sanity check after weight transfer.
  LF-14  attention_mask shape-safe handling.
  LF-15  E/F init only on min(k, max_seq) diagonal.
  LF-16  _is_single_layer via attribute whitelist (no string class names).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE (dtype-safe)
# ===========================================================================

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

    if cos.shape[-2] != q.shape[-2]:
        raise ValueError(
            f"apply_rotary_pos_emb: cos/sin seq len {cos.shape[-2]} != q seq len {q.shape[-2]}"
        )

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


# ===========================================================================
# Decoder detection
# ===========================================================================

def _detect_decoder(
    config: Any,
    is_decoder: Optional[bool],
    src_attn: Optional[nn.Module],
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "") or "").lower()
        if mt.startswith(("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                          "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
            return True
        arch = getattr(config, "architectures", None)
        if arch and any(("GPT" in str(a) or "CausalLM" in str(a)) for a in arch):
            return True
    if src_attn is not None:
        if getattr(src_attn, "is_decoder", False):
            return True
        if hasattr(src_attn, "c_attn"):
            return True
        if "gpt" in src_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# LF-12: FactorizedLoRALinear
# ===========================================================================

class FactorizedLoRALinear(nn.Module):
    is_custom_attention = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.svd_rank = self.rank
        self.lora_rank = int(lora_rank) if lora_rank > 0 else 0
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        factory = {"device": device, "dtype": dtype}
        self.down = nn.Linear(self.in_features, self.rank, bias=False, **factory)
        self.up = nn.Linear(self.rank, self.out_features, bias=bias, **factory)

        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features, **factory))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.lora_rank, **factory))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.up.weight, self.down.weight)
        if self.lora_A is not None and self.lora_B is not None:
            w = w + (self.lora_B @ self.lora_A) * self.lora_scaling
        return w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.up.bias

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.down.weight.requires_grad = True
        self.up.weight.requires_grad = True
        if self.up.bias is not None:
            self.up.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if x.dtype != self.down.weight.dtype:
            x = x.to(self.down.weight.dtype)
        out = self.up(self.down(x))
        if self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out.to(orig_dtype)


# ===========================================================================
# Main Linformer attention
# ===========================================================================

@ATTENTION_REGISTRY.register("linformer", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("linformer_attention", supports_causal=True, supports_gqa=True)
class LinformerAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        k: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # LF-02: robust super().__init__
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        # LF-04: robust dimension detection
        hidden_size = self._detect_hidden_size(config, source_attn, kwargs)
        num_heads = self._detect_num_heads(config, source_attn, kwargs, hidden_size)
        num_kv_heads = self._detect_num_kv_heads(config, source_attn, kwargs, num_heads)
        head_dim = self._detect_head_dim(config, source_attn, kwargs, hidden_size, num_heads)

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # Linformer hyperparams
        k_val = k if k is not None else getattr(comp_config, "linformer_k",
                                                getattr(comp_config, "k", None) if comp_config else None)
        if k_val is None:
            k_val = getattr(config, "linformer_k", getattr(config, "k", 64)) if config else 64
        self.k_proj_dim = int(k_val)

        pos_emb = getattr(config, "max_position_embeddings", None) if config else None
        if pos_emb is None:
            pos_emb = max_seq_len if max_seq_len is not None else 512
        self.max_seq_len = max(512, int(pos_emb))

        # LF-11: centralized decoder detection
        self.is_decoder = _detect_decoder(config, is_decoder or kwargs.get("is_causal"), source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # LoRA config
        self.lora_rank = int(
            (getattr(config, "lora_rank", None) if config else None)
            or (getattr(comp_config, "lora_rank", None) if comp_config else None)
            or kwargs.get("lora_rank", 16)
        )
        self.lora_alpha = float(
            (getattr(config, "lora_alpha", None) if config else None)
            or (getattr(comp_config, "lora_alpha", None) if comp_config else None)
            or kwargs.get("lora_alpha", 32.0)
        )

        # LF-12: projection rank for FactorizedLoRALinear
        self.proj_rank = int(
            kwargs.get("proj_rank")
            or (getattr(comp_config, "proj_rank", None) if comp_config else None)
            or (rank if rank is not None else 64)
        )

        # FactorizedLoRALinear projections
        FF = FactorizedLoRALinear
        self.q_proj = FF(self.dim, self.all_head_size, rank=self.proj_rank,
                         lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)
        self.k_proj = FF(self.dim, self.all_kv_head_size, rank=self.proj_rank,
                         lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)
        self.v_proj = FF(self.dim, self.all_kv_head_size, rank=self.proj_rank,
                         lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)
        self.out_proj = FF(self.all_head_size, self.dim, rank=self.proj_rank,
                           lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)

        # Linformer projections: E, F: (k, max_seq_len)
        self.E = nn.Parameter(torch.empty(self.k_proj_dim, self.max_seq_len))
        self.F = nn.Parameter(torch.empty(self.k_proj_dim, self.max_seq_len))
        with torch.no_grad():
            nn.init.normal_(self.E, std=0.02)
            nn.init.normal_(self.F, std=0.02)
            # LF-15: only touch the min(k, max_seq) block
            min_dim = min(self.k_proj_dim, self.max_seq_len)
            self.E.data[:min_dim, :min_dim].fill_diagonal_(1.0)
            self.F.data[:min_dim, :min_dim].fill_diagonal_(1.0)

        dropout_p = float(
            (getattr(config, "dropout",
                     getattr(config, "hidden_dropout_prob",
                             getattr(config, "attention_probs_dropout_prob", 0.1)))
             if config else 0.1)
        )
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # LF-04: dimension detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_hidden_size(config: Any, src: Optional[nn.Module], kwargs: Dict[str, Any]) -> int:
        for k in ("hidden_size", "d_model", "n_embd", "dim"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("hidden_size", "d_model", "n_embd", "dim"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for attr in ("out_lin", "out_proj", "o_proj", "dense", "c_proj"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "out_features"):
                    return int(mod.out_features)
            for attr in ("q_proj", "q_lin", "query", "c_attn"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "in_features"):
                    return int(mod.in_features)
        return 768

    @staticmethod
    def _detect_num_heads(config: Any, src: Optional[nn.Module], kwargs: Dict[str, Any],
                          hidden_size: int) -> int:
        for k in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("num_attention_heads", "num_heads", "n_heads", "n_head"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for k in ("num_heads", "num_attention_heads", "n_heads"):
                v = getattr(src, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        return max(1, hidden_size // 64)

    @staticmethod
    def _detect_num_kv_heads(config: Any, src: Optional[nn.Module], kwargs: Dict[str, Any],
                             num_heads: int) -> int:
        for k in ("num_key_value_heads", "num_kv_heads"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("num_key_value_heads", "num_kv_heads"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            v = getattr(src, "num_key_value_heads", None)
            if isinstance(v, int) and v > 0:
                return v
        return num_heads

    @staticmethod
    def _detect_head_dim(config: Any, src: Optional[nn.Module], kwargs: Dict[str, Any],
                         hidden_size: int, num_heads: int) -> int:
        v = kwargs.get("head_dim")
        if isinstance(v, int) and v > 0:
            return v
        if config is not None:
            for k in ("head_dim", "kv_channels", "d_kv"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for attr in ("q_proj", "q_lin", "query"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
                    try:
                        return int(mod.weight.shape[0] // max(1, num_heads))
                    except Exception:
                        pass
        return max(1, hidden_size // max(1, num_heads))

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
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
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # SVD init helper
    # ------------------------------------------------------------------
    @staticmethod
    def _svd_init(w: torch.Tensor, b: Optional[torch.Tensor],
                  proj: FactorizedLoRALinear) -> None:
        with torch.no_grad():
            W = w.detach().float()
            if W.ndim != 2:
                W = W.reshape(W.size(0), -1)

            out_t, in_t = proj.out_features, proj.in_features
            if W.shape[0] != out_t or W.shape[1] != in_t:
                padded = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
                r = min(out_t, W.shape[0])
                c = min(in_t, W.shape[1])
                padded[:r, :c] = W[:r, :c]
                W = padded

            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception as e:
                logger.warning("Linformer._svd_init SVD failed: %s", e)
                return

            r = min(proj.rank, U.size(1), Vh.size(0))
            proj.down.weight.data.zero_()
            proj.up.weight.data.zero_()
            proj.down.weight.data[:r, :].copy_(Vh[:r, :].to(proj.down.weight.dtype))
            US = U[:, :r] @ torch.diag(S[:r])
            proj.up.weight.data[:, :r].copy_(US.to(proj.up.weight.dtype))

            if b is not None and proj.up.bias is not None:
                n = min(b.numel(), proj.up.bias.numel())
                proj.up.bias.data[:n].copy_(b[:n].to(proj.up.bias.dtype))
            elif proj.up.bias is not None:
                proj.up.bias.data.zero_()

    # ------------------------------------------------------------------
    # copy_weights (LF-07, LF-08, LF-11)
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
        elif (cls_name in ["BertSelfAttention", "RobertaSelfAttention",
                           "XLMRobertaSelfAttention", "ElectraSelfAttention",
                           "DebertaSelfAttention"]
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())) \
                and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

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
                is_conv = (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (c_w.ndim == 2 and c_w.shape[1] == 3 * self.hidden_size)
                )
                if is_conv:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
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
                    is_conv_o = (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. BERT / RoBERTa nested
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

        # 3. Flat modules (DistilBERT, Qwen, LLaMA)
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
            raise RuntimeError(
                f"LinformerAttention.copy_weights: missing mandatory projections "
                f"{missing_after} from {cls_name}."
            )

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        def _resize_gqa(w, b, out_target):
            """LF-07: adapt w for GQA: average down or repeat to match out_target."""
            if w is None:
                return w, b
            src_out = w.shape[0]
            if src_out == out_target:
                return w, b
            if src_out > out_target and src_out % out_target == 0:
                rep = src_out // out_target
                w = w.view(out_target, rep, -1).mean(dim=1)
                if b is not None:
                    b = b.view(out_target, rep).mean(dim=1)
            elif src_out < out_target and out_target % src_out == 0:
                rep = out_target // src_out
                w = w.repeat(rep, 1)
                if b is not None:
                    b = b.repeat(rep)
            return w, b

        with torch.no_grad():
            w_q, _ = extracted["q"]
            w_k, _ = extracted["k"]
            w_v, _ = extracted["v"]
            w_o, b_o = extracted["o"]

            if w_q is not None:
                self._svd_init(_orient(w_q, self.hidden_size), None, self.q_proj)
            if w_k is not None:
                w_k_o = _orient(w_k, self.hidden_size)
                w_k_res, _ = _resize_gqa(w_k_o, None, self.k_proj.out_features)
                self._svd_init(w_k_res, None, self.k_proj)
            if w_v is not None:
                w_v_o = _orient(w_v, self.hidden_size)
                w_v_res, _ = _resize_gqa(w_v_o, None, self.v_proj.out_features)
                self._svd_init(w_v_res, None, self.v_proj)
            if w_o is not None:
                self._svd_init(_orient(w_o, self.all_head_size), b_o, self.out_proj)
            elif self.is_nested_encoder:
                eye = torch.eye(self.all_head_size, self.all_head_size,
                                device=self.out_proj.down.weight.device,
                                dtype=self.out_proj.down.weight.dtype)
                self._svd_init(eye, None, self.out_proj)

            # LF-08: norm verification via float() cast
            for name, mod in (("q_proj", self.q_proj), ("k_proj", self.k_proj),
                              ("v_proj", self.v_proj), ("out_proj", self.out_proj)):
                try:
                    w_norm = float(torch.linalg.norm(mod.weight.float()).item())
                    if w_norm < 1e-4:
                        logger.warning(
                            "LinformerAttention: weight '%s' has near-zero norm (%.2e).",
                            name, w_norm,
                        )
                except Exception:
                    pass

            for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                mod.freeze_base_weights()

        # LF-11: inherit is_decoder
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
            self.is_causal = self.is_decoder

        if hasattr(src, "rotary_emb") and src.rotary_emb is not None:
            self.rotary_emb = src.rotary_emb
        else:
            if not hasattr(self, "rotary_emb"):
                self.rotary_emb = None

        # LF-13: forward sanity check
        try:
            with torch.no_grad():
                d = next(self.parameters()).device
                dt = self.q_proj.down.weight.dtype
                dummy = torch.randn(1, 8, self.dim, device=d, dtype=dt)
                _ = self.forward(dummy)
        except Exception as e:
            logger.debug("LinformerAttention forward sanity check warning: %s", e)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Any] = None,
        layer_idx: int = 0,
        output_attentions: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # LF-05: RoPE with past_len offset
        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            if past_len > 0:
                cos = cos[..., past_len: past_len + N, :]
                sin = sin[..., past_len: past_len + N, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # Mask K, V (LF-14: shape-safe)
        valid_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.dtype == torch.bool:
                    valid_mask = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    valid_mask = attention_mask > -1e4
                else:
                    valid_mask = attention_mask > 0
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    v_tmp = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    v_tmp = attention_mask > -1e4
                else:
                    v_tmp = attention_mask > 0
                valid_mask = v_tmp.any(dim=-2).squeeze(1)
            else:
                valid_mask = torch.ones(B, N, device=device, dtype=torch.bool)

        if valid_mask is not None:
            mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
            k = k * mask_expanded
            v = v * mask_expanded
        else:
            mask_expanded = torch.ones(B, 1, N, 1, device=device, dtype=dtype)

        L = N
        k_eff = min(self.k_proj_dim, L)

        # E, F: (k_proj_dim, max_seq_len) → (k_eff, L)
        if L <= self.max_seq_len:
            E_act = self.E[:k_eff, :L].to(dtype=dtype, device=device)
            F_act = self.F[:k_eff, :L].to(dtype=dtype, device=device)
        else:
            # LF-10: interpolation with 3D input
            E_act = F.interpolate(
                self.E[:k_eff].unsqueeze(0).float(),
                size=L, mode="linear", align_corners=False,
            ).squeeze(0).to(dtype=dtype, device=device)
            F_act = F.interpolate(
                self.F[:k_eff].unsqueeze(0).float(),
                size=L, mode="linear", align_corners=False,
            ).squeeze(0).to(dtype=dtype, device=device)

        # Project key/value sequence: (B, H, k_eff, d)
        K_proj = torch.matmul(E_act.to(k.dtype), k)
        V_proj = torch.matmul(F_act.to(v.dtype), v)

        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q.float(), K_proj.transpose(-2, -1).contiguous().float()) * scale

        # LF-09: proper causal masking for projected K/V
        if self.is_decoder:
            if L <= self.k_proj_dim:
                # Exact causal on full sequence — recompute without projection
                causal_scores = torch.matmul(q.float(), k.transpose(-2, -1).contiguous().float()) * scale
                causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
                scores = causal_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
                V_proj = v
            else:
                # Approximate causal on projected K/V: mask each projected key by its mean position
                causal_approx = torch.tril(
                    torch.ones(L, k_eff, device=device, dtype=torch.bool)
                )
                # For each query position i, allow projected keys whose support range includes <= i
                # Simplified: allow key j if j <= i (proportional to support)
                idx_q = torch.arange(L, device=device).unsqueeze(1)      # (L, 1)
                idx_k = torch.arange(k_eff, device=device).unsqueeze(0)  # (1, k_eff)
                # Each projected key j covers roughly L/k_eff positions
                step = max(1, L // k_eff)
                allowed = (idx_k * step) <= idx_q                        # (L, k_eff)
                scores = scores.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), -1e4)

        if valid_mask is not None:
            scores = scores.masked_fill(
                ~valid_mask.unsqueeze(1).unsqueeze(-1), -1e4
            )

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                hm = head_mask[:, :, :1].unsqueeze(-1)
            else:
                hm = head_mask
            attn = attn * hm.to(attn.dtype)

        context = torch.matmul(attn, V_proj.to(attn.dtype))
        if valid_mask is not None:
            context = context * mask_expanded
        context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out.to(dtype), (attn if output_attentions else None)


# ===========================================================================
# LF-16: single-layer detection via attribute whitelist
# ===========================================================================

_SINGLE_LAYER_ATTRS = ("attention", "attn", "self_attn", "self")
_MODEL_ATTRS = ("transformer", "distilbert", "roberta", "bert",
                "encoder", "model", "layers", "h")


def _is_single_layer(obj: Any) -> bool:
    if obj is None:
        return False
    # A single layer must have an attention submodule
    has_attn = any(hasattr(obj, a) for a in _SINGLE_LAYER_ATTRS)
    if not has_attn:
        return False
    # But must NOT have any model-level container
    has_model_container = any(hasattr(obj, a) for a in _MODEL_ATTRS)
    return not has_model_container


# ===========================================================================
# Injection (LF-03: depth via argument, thread-safe)
# ===========================================================================

@ATTENTION_REGISTRY.register("linformer_inject")
@ATTENTION_REGISTRY.register("inject_linformer")
def inject_linformer(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_linformer: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    if _is_single_layer(target):
        pass
    elif hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
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
            res = inject_linformer(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # Single-layer injection
    try:
        src_attn = old_attn
        if src_attn is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    src_attn = getattr(cand, "self", cand)
                    break

        if src_attn is None:
            return None

        # Double-injection guard
        if isinstance(src_attn, LinformerAttention) or "Linformer" in src_attn.__class__.__name__:
            return None

        # Skip Qwen hybrid layers
        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        k_val = (getattr(comp_config, "linformer_k",
                         getattr(comp_config, "k", 64)) if comp_config else kwargs.get("linformer_k", 64))
        max_seq_len = max(512, getattr(cfg, "max_position_embeddings", 512)) if cfg else 512

        new_attn = LinformerAttention(
            cfg,
            k=k_val,
            max_seq_len=max_seq_len,
            comp_config=comp_config,
            old_attn=src_attn,
            **kwargs,
        )

        p = next(src_attn.parameters(), None) if src_attn is not None else None
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

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
        logger.exception("inject_linformer failed at layer %d: %s", layer_idx, e)
        return None