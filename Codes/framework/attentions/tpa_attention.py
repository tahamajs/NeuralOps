r"""
TPAttention: Tensor Product Attention with Factorised Q-K Projections (Fast CP init).

Fixes in this revision:
  TPA-FIX-01  Fast CP init via marginalization + few ALS steps (no masks,
              no einsum on (H, r, r, in_dim) with mask). Init time reduced
              from minutes to under a second for typical shapes.
  TPA-FIX-02  Requires r * r == head_dim for the fast path; otherwise the
              identity-select qk_proj is used.
  TPA-FIX-03  Recursion guard in injector.
  TPA-FIX-04  RoPE dtype cast.
  TPA-FIX-05  Removed per-forward `del Q, K, v` (was a no-op).
  TPA-FIX-06  Fallback to orthogonal init if CP yields NaN.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _HAS_SDPA_KERNEL = True
except ImportError:
    _HAS_SDPA_KERNEL = False

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    if rotary_dim == 0:
        return x
    x_rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


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
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim, k.shape[-1]) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_sub = cos[..., : q.shape[-2], :rotary_dim]
    sin_sub = sin[..., : q.shape[-2], :rotary_dim]

    # TPA-FIX-04: dtype-safe cast
    cos_q = cos_sub.to(device=q.device, dtype=q.dtype)
    sin_q = sin_sub.to(device=q.device, dtype=q.dtype)
    cos_k = cos_sub.to(device=k.device, dtype=k.dtype)
    sin_k = sin_sub.to(device=k.device, dtype=k.dtype)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_out[..., :rotary_dim] = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_out[..., :rotary_dim] = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
    if q_pass.numel() > 0:
        q_out[..., rotary_dim:] = q_pass
        k_out[..., rotary_dim:] = k_pass
    return q_out, k_out


# ============================================================================
# TPA-FIX-01: Fast CP init
# ============================================================================

def _cp_decompose_qk_fast(
    W: torch.Tensor,
    num_heads: int,
    rank: int,
    head_dim: int,
    n_iter: int = 3,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast CP decomposition of W (H*head_dim, in_dim) for TPA forward semantics:
        W[h*head_dim + a*rank + b, k] ≈ A[h, a, k] · B[h, b, k]

    Requires rank * rank == head_dim (true for r=8, head_dim=64).

    Init strategy:
      A[h, a] = mean over b of W[h, a*r+b]
      B[h, b] = mean over a of W[h, a*r+b]

    Then a few ALS iterations refine A, B. No mask, no loops over heads.
    """
    W_f = W.detach().float()
    H = num_heads
    in_dim = W_f.shape[1]

    if rank * rank != head_dim:
        raise ValueError(
            f"Fast CP requires rank*rank == head_dim, got "
            f"rank={rank}, head_dim={head_dim}"
        )

    # (H, r, r, in_dim)
    W_r = W_f.view(H, rank, rank, in_dim)

    # Initialize A, B via marginalization
    A = W_r.mean(dim=2)   # (H, r, in_dim)
    B = W_r.mean(dim=1)   # (H, r, in_dim)

    # Prevent exact-zero rows
    A = A + eps
    B = B + eps

    for _ in range(n_iter):
        # Update A given B:
        #   A[h,a,k] = Σ_b W[h,a,b,k]·B[h,b,k] / (Σ_b B[h,b,k]^2 + eps)
        num_A = torch.einsum("habk,hbk->hak", W_r, B)
        den_A = torch.einsum("hbk,hbk->hk", B, B).unsqueeze(1)
        A = num_A / (den_A + eps)

        # Update B given A:
        #   B[h,b,k] = Σ_a W[h,a,b,k]·A[h,a,k] / (Σ_a A[h,a,k]^2 + eps)
        num_B = torch.einsum("habk,hak->hbk", W_r, A)
        den_B = torch.einsum("hak,hak->hk", A, A).unsqueeze(1)
        B = num_B / (den_B + eps)

    # Rescale for balanced dynamic range (keeps numerical stability for RoPE)
    norm_A = A.norm(dim=-1, keepdim=True).clamp(min=eps)
    norm_B = B.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale = (norm_B / norm_A).sqrt()
    A = A * scale
    B = B / scale

    A_out = A.reshape(H * rank, in_dim).contiguous()
    B_out = B.reshape(H * rank, in_dim).contiguous()
    return A_out, B_out


# ============================================================================
# LoRALinear
# ============================================================================

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out


# ============================================================================
# TPAttention
# ============================================================================

@ATTENTION_REGISTRY.register("tpa", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("tpa_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("tp_attention", supports_causal=True, supports_gqa=True)
class TPAttention(BaseAttention):
    supports_gqa: bool = True
    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 8,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)
        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn", None)

        # ---- dim ----
        dim = None
        if config is not None and not isinstance(config, int):
            dim = _cfg_get(config, "hidden_size",
                  _cfg_get(config, "dim",
                  _cfg_get(config, "d_model", None)))
        if dim is None and src is not None:
            dim = getattr(src, "hidden_size", getattr(src, "embed_dim", None))
            if dim is None and hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features"):
                dim = src.q_proj.in_features
            elif dim is None and hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features"):
                dim = src.q_lin.in_features
            elif dim is None and hasattr(src, "query") and hasattr(src.query, "in_features"):
                dim = src.query.in_features
            elif dim is None and hasattr(src, "c_attn") and hasattr(src.c_attn, "weight"):
                w = src.c_attn.weight
                dim = w.shape[0] if w.shape[0] != 3 * w.shape[1] else w.shape[1]
        if dim is None:
            dim = kwargs.get("hidden_size", 768)

        # ---- heads ----
        num_heads = None
        if config is not None and not isinstance(config, int):
            num_heads = _cfg_get(config, "num_attention_heads",
                        _cfg_get(config, "n_heads",
                        _cfg_get(config, "num_heads", None)))
        if num_heads is None and src is not None:
            num_heads = getattr(src, "num_heads",
                        getattr(src, "num_attention_heads",
                        getattr(src, "n_head", None)))
        if num_heads is None:
            num_heads = 12

        self.dim: int = int(dim)
        self.num_heads: int = max(1, int(num_heads))

        # ---- head_dim ----
        cfg_hd = _cfg_get(config, "head_dim", None) if config is not None else None
        kw_hd = kwargs.get("head_dim", None)
        if cfg_hd is not None:
            self.head_dim: int = int(cfg_hd)
        elif kw_hd is not None:
            self.head_dim: int = int(kw_hd)
        elif src is not None and getattr(src, "head_dim", None) is not None:
            self.head_dim = int(getattr(src, "head_dim"))
        else:
            self.head_dim = self.dim // self.num_heads

        # ---- GQA ----
        cfg_kv = (
            _cfg_get(config, "num_key_value_heads",
                     _cfg_get(config, "num_kv_heads", None))
            if config is not None else None
        )
        if cfg_kv is None and src is not None:
            cfg_kv = getattr(src, "num_key_value_heads", getattr(src, "num_kv_heads", None))
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_key_value_groups: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # ---- rank ----
        raw_r = kwargs.get("rank", _cfg_get(config, "rank", rank))
        self.rank = max(1, int(raw_r))

        # If rank*rank != head_dim, use identity-select qk_proj.
        self._use_qk_proj = (self.rank * self.rank != self.head_dim)

        self.lora_rank: int = int(kwargs.get("lora_rank",
                                    _cfg_get(config, "lora_rank", lora_rank)))
        self.lora_alpha: float = float(kwargs.get("lora_alpha",
                                        _cfg_get(config, "lora_alpha", lora_alpha)))

        # ---- decoder ----
        _mt = str(_cfg_get(config, "model_type", "") or "").lower() if config is not None else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                             "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        elif config is not None and (
            _cfg_get(config, "is_decoder", False)
            or _cfg_get(config, "is_causal", False)
            or any(_mt.startswith(p) for p in _decoder_prefixes)
        ):
            self.is_decoder = True
        else:
            self.is_decoder = bool(kwargs.get("is_decoder", False))

        self.is_nested_encoder: bool = False
        self.gradient_checkpointing: bool = bool(kwargs.get("gradient_checkpointing", True))

        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and src is not None:
            p = next(src.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        # ---- projections ----
        self.q_proj1 = LoRALinear(
            self.dim, self.num_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.q_proj2 = LoRALinear(
            self.dim, self.num_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.k_proj1 = LoRALinear(
            self.dim, self.num_kv_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.k_proj2 = LoRALinear(
            self.dim, self.num_kv_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.q_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )

        # Identity-select projection (frozen) if r*r != head_dim.
        if self._use_qk_proj:
            n = min(self.rank * self.rank, self.head_dim)
            qk = torch.zeros(self.head_dim, self.rank * self.rank,
                             dtype=target_dtype or torch.float32)
            for i in range(n):
                qk[i, i] = 1.0
            self.qk_proj = nn.Parameter(qk, requires_grad=False)
        else:
            self.register_parameter("qk_proj", None)

        dropout_p = (
            getattr(config, "dropout",
                    getattr(config, "attention_probs_dropout_prob", 0.1))
            if config else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        if target_dtype is not None:
            for proj in (self.q_proj1, self.q_proj2,
                         self.k_proj1, self.k_proj2,
                         self.v_proj, self.out_proj):
                proj.to(target_dtype)
            if self.qk_proj is not None:
                self.qk_proj.data = self.qk_proj.data.to(target_dtype)

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        with torch.no_grad():
            w_q = w_k = w_v = w_o = None
            b_q = b_k = b_v = b_o = None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            has_no_out = not any(
                hasattr(old_attn, a)
                for a in ("dense", "out_proj", "o_proj", "c_proj", "out_lin", "proj")
            )
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") \
                    or (hasattr(old_attn, "query") and has_no_out):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # GPT-2
            if hasattr(src, "c_attn"):
                W = src.c_attn.weight.detach()
                B_vec = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if W.shape[0] != 3 * self.dim and W.shape[1] == 3 * self.dim:
                    W = W.t().contiguous()
                if W.shape[0] == 3 * self.dim:
                    w_q, w_k, w_v = W.chunk(3, dim=0)
                    if B_vec is not None:
                        b_q, b_k, b_v = B_vec.chunk(3, dim=0)
                if hasattr(src, "c_proj"):
                    w_o = src.c_proj.weight.detach()
                    if w_o.shape[0] != self.dim and w_o.shape[1] == self.dim:
                        w_o = w_o.t().contiguous()
                    b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # BERT / RoBERTa
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None
                if w_o is None:
                    for a in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, a):
                            proj = getattr(src, a)
                            w_o = proj.weight.detach()
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            # Qwen / LLaMA
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None
                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None

            # DistilBERT
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None
                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None

            # UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    if w_q is None:
                        w_q, b_q = adapter.get_weight_bias(src, "q")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(src, "k")
                    if w_v is None:
                        w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            missing = []
            if w_q is None: missing.append("Q")
            if w_k is None: missing.append("K")
            if w_v is None: missing.append("V")
            if w_o is None and not self.is_nested_encoder: missing.append("O")
            if missing:
                logger.warning("TPAttention: failed to extract %s from %s",
                               missing, type(old_attn).__name__)
                return

            # ---- CP decomposition for Q and K ----
            def _apply_cp(W: torch.Tensor, proj1: LoRALinear, proj2: LoRALinear, num_h: int):
                A = B = None
                if self.rank * self.rank == self.head_dim:
                    try:
                        A, B = _cp_decompose_qk_fast(
                            W, num_h, self.rank, self.head_dim, n_iter=3,
                        )
                    except Exception as e:
                        logger.warning("TPAttention: CP init failed (%s), "
                                       "falling back to orthogonal", e)
                if A is None or B is None or torch.isnan(A).any() or torch.isnan(B).any():
                    # TPA-FIX-06: fallback
                    nn.init.orthogonal_(proj1.weight)
                    nn.init.orthogonal_(proj2.weight)
                    return
                r_rows = num_h * self.rank
                in_dim = min(W.shape[1], proj1.weight.shape[1])
                proj1.weight.data.zero_()
                proj2.weight.data.zero_()
                proj1.weight.data[:r_rows, :in_dim].copy_(
                    A[:r_rows, :in_dim].to(proj1.weight.dtype))
                proj2.weight.data[:r_rows, :in_dim].copy_(
                    B[:r_rows, :in_dim].to(proj2.weight.dtype))
                proj1.weight.requires_grad = False
                proj2.weight.requires_grad = False
                if proj1.lora_A is not None:
                    nn.init.kaiming_uniform_(proj1.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(proj1.lora_B)
                    proj1.lora_A.requires_grad = True
                    proj1.lora_B.requires_grad = True
                if proj2.lora_A is not None:
                    nn.init.kaiming_uniform_(proj2.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(proj2.lora_B)
                    proj2.lora_A.requires_grad = True
                    proj2.lora_B.requires_grad = True

            _apply_cp(w_q, self.q_proj1, self.q_proj2, self.num_heads)
            _apply_cp(w_k, self.k_proj1, self.k_proj2, self.num_kv_heads)

            # V and O direct copy
            def _copy_layer(target: LoRALinear, w, b):
                if w is None:
                    return
                if w.shape == target.weight.shape:
                    target.weight.copy_(w.to(target.weight.dtype))
                elif w.t().contiguous().shape == target.weight.shape:
                    target.weight.copy_(w.t().contiguous().to(target.weight.dtype))
                else:
                    m0 = min(target.weight.shape[0], w.shape[0])
                    m1 = min(target.weight.shape[1], w.shape[1])
                    target.weight.zero_()
                    target.weight[:m0, :m1].copy_(w[:m0, :m1].to(target.weight.dtype))
                if b is not None and target.bias is not None:
                    mb = min(target.bias.shape[0], b.shape[0])
                    target.bias.zero_()
                    target.bias[:mb].copy_(b[:mb].to(target.bias.dtype))
                elif target.bias is not None:
                    target.bias.zero_()

            _copy_layer(self.v_proj, w_v, b_v)
            if w_o is not None:
                _copy_layer(self.out_proj, w_o, b_o)

            if hasattr(old_attn, "is_decoder"):
                self.is_decoder = bool(old_attn.is_decoder)
            elif hasattr(old_attn, "c_attn") or any(
                x in old_attn.__class__.__name__.lower()
                for x in ("gpt", "qwen", "llama", "decoder")
            ):
                self.is_decoder = True

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
        *args: Any,
        **kwargs: Any,
    ):
        B, L, D = hidden_states.shape
        H, H_kv, r, hd = self.num_heads, self.num_kv_heads, self.rank, self.head_dim

        def _compute_q(x: torch.Tensor) -> torch.Tensor:
            q1 = self.q_proj1(x).view(B, L, H, r)
            q2 = self.q_proj2(x).view(B, L, H, r)
            out_q = (q1.unsqueeze(-1) * q2.unsqueeze(-2)).flatten(-2, -1)
            if self.qk_proj is not None:
                out_q = F.linear(out_q, self.qk_proj.to(out_q.dtype))
            return out_q

        def _compute_k(x: torch.Tensor) -> torch.Tensor:
            k1 = self.k_proj1(x).view(B, L, H_kv, r)
            k2 = self.k_proj2(x).view(B, L, H_kv, r)
            out_k = (k1.unsqueeze(-1) * k2.unsqueeze(-2)).flatten(-2, -1)
            if self.qk_proj is not None:
                out_k = F.linear(out_k, self.qk_proj.to(out_k.dtype))
            return out_k

        if self.training and getattr(self, "gradient_checkpointing", True):
            try:
                from torch.utils.checkpoint import checkpoint
                Q = checkpoint(_compute_q, hidden_states, use_reentrant=False)
                K = checkpoint(_compute_k, hidden_states, use_reentrant=False)
            except Exception:
                Q = _compute_q(hidden_states)
                K = _compute_k(hidden_states)
        else:
            Q = _compute_q(hidden_states)
            K = _compute_k(hidden_states)

        v = self.v_proj(hidden_states).view(B, L, H_kv, hd)

        Q = Q.transpose(1, 2).contiguous()
        K = K.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # RoPE
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
            cos, sin = pos_emb
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # GQA
        if H_kv != H:
            K = repeat_kv(K, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # Mask
        is_causal = self.is_decoder and L > 1
        custom_mask = attention_mask if attention_mask is not None else kwargs.get("attn_mask", None)

        if custom_mask is not None:
            if custom_mask.dtype == torch.bool:
                if custom_mask.all():
                    custom_mask = None
                elif custom_mask.dim() == 2:
                    custom_mask = custom_mask.unsqueeze(1).unsqueeze(2)
                elif custom_mask.dim() == 3:
                    custom_mask = custom_mask.unsqueeze(1)
            elif torch.is_floating_point(custom_mask):
                if (custom_mask == 0.0).all():
                    custom_mask = None
                elif (custom_mask <= -1.0).any() and \
                        ((custom_mask == 0.0) | (custom_mask <= -1.0)).all():
                    custom_mask = (custom_mask > -1.0)
                    if custom_mask.dim() == 2:
                        custom_mask = custom_mask.unsqueeze(1).unsqueeze(2)
                    elif custom_mask.dim() == 3:
                        custom_mask = custom_mask.unsqueeze(1)
                else:
                    if custom_mask.dim() == 2:
                        custom_mask = custom_mask.unsqueeze(1).unsqueeze(2)
                    elif custom_mask.dim() == 3:
                        custom_mask = custom_mask.unsqueeze(1)
                    if custom_mask.dtype != Q.dtype:
                        custom_mask = custom_mask.to(dtype=Q.dtype)

        # SDPA fast path
        if not output_attentions and head_mask is None:
            try:
                if _HAS_SDPA_KERNEL:
                    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                        context = F.scaled_dot_product_attention(
                            query=Q, key=K, value=v,
                            attn_mask=custom_mask,
                            dropout_p=self.dropout.p if self.training else 0.0,
                            is_causal=is_causal if custom_mask is None else False,
                        )
                else:
                    context = F.scaled_dot_product_attention(
                        query=Q, key=K, value=v,
                        attn_mask=custom_mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=is_causal if custom_mask is None else False,
                    )
            except (RuntimeError, TypeError):
                context = F.scaled_dot_product_attention(
                    query=Q, key=K, value=v,
                    attn_mask=custom_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=is_causal if custom_mask is None else False,
                )
            attn_weights = None
        else:
            scale = 1.0 / math.sqrt(hd)
            scores = torch.matmul(Q, K.transpose(-1, -2).contiguous()) * scale
            if is_causal and custom_mask is None:
                causal_mask = torch.triu(
                    torch.full((L, L), float("-inf"),
                               device=scores.device, dtype=scores.dtype),
                    diagonal=1,
                )
                scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            elif custom_mask is not None:
                scores = apply_attention_mask(scores, custom_mask)

            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                attn_weights = attn_weights * head_mask.to(attn_weights.dtype)
            context = torch.matmul(attn_weights, v)

        if self.is_nested_encoder:
            output = context.transpose(1, 2).contiguous().view(B, L, H * hd)
        else:
            output = self.out_proj(context.transpose(1, 2).contiguous().view(B, L, H * hd))

        if output.shape[-1] != hidden_states.shape[-1]:
            if output.shape[-1] < hidden_states.shape[-1]:
                output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1]))
            else:
                output = output[..., : hidden_states.shape[-1]]

        valid_m = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
        if valid_m is not None:
            output = output * valid_m.unsqueeze(-1).to(output.dtype)

        return output, (attn_weights if output_attentions else None)


# ============================================================================
# Injection (TPA-FIX-03: recursion guard)
# ============================================================================

@ATTENTION_REGISTRY.register("tpa_inject")
@ATTENTION_REGISTRY.register("tp_attention_inject")
@ATTENTION_REGISTRY.register("inject_tpa")
def inject_tpa(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Optional[Union[TPAttention, nn.Module]]:
    if _depth > 8:
        logger.warning("inject_tpa: max recursion depth exceeded")
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
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_tpa(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot,
                _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if cand is None:
                        continue
                    src = getattr(cand, "self", cand)
                    break
        if src is None and hasattr(target, "forward"):
            src = target
        if src is None:
            return None

        # Double-injection guard
        if isinstance(src, TPAttention):
            return None
        if "TPAttention" in src.__class__.__name__:
            return None
        for sub_attr in ("inner", "_wrapped_attention"):
            sub = getattr(src, sub_attr, None)
            if isinstance(sub, TPAttention):
                return None

        if "DeltaNet" in src.__class__.__name__ or "LinearAttention" in src.__class__.__name__:
            return None

        if cfg is None and hasattr(src, "config"):
            cfg = src.config

        rank = kwargs.pop("rank",
                 _cfg_get(comp_config, "tpa_rank",
                 _cfg_get(comp_config, "rank", 8)))
        lora_rank = kwargs.pop("lora_rank",
                     _cfg_get(comp_config, "lora_rank", 16))

        new_attn = TPAttention(
            config=cfg, rank=rank, lora_rank=lora_rank,
            pretrained_attn=src, **kwargs,
        )

        p = next(src.parameters(), None)
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
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn
        return new_attn
    except Exception as e:
        logger.exception("inject_tpa failed at layer %d: %s", layer_idx, e)
        return None


tp_inject = inject_tpa
TPA = TPAttention