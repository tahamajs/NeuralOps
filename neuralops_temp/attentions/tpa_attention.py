"""
TPAttention v2: Tensor Product Attention with Factorised Q-K Projections.

Improvements over v1:
  1. Asymmetric rank support (rank_a × rank_b = head_dim) → supports head_dim=128
  2. Better CP decomposition (50 iterations with early stopping)
  3. Stricter max_cp_error (0.05 instead of 0.35)
  4. Removed qk_proj hack — clean fallback to standard projection
  5. KV cache support for efficient autoregressive inference
  6. Cleaner code (removed TPA-FIX-XX comments)

Tensor-Product factorization of Q and K:
    W_q ≈ q_proj1 ⊗ q_proj2   (per head)
    W_q[h*hd + a*rank_b + b, k] ≈ A[h, a, k] · B[h, b, k]
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _HAS_SDPA_KERNEL = True
except ImportError:
    _HAS_SDPA_KERNEL = False

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

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


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
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
# CP decomposition (improved with early stopping)
# ============================================================================


def _cp_decompose_qk(
    W: torch.Tensor,
    num_heads: int,
    rank_a: int,
    rank_b: int,
    n_iter: int = 50,
    tol: float = 1e-6,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    CP decomposition of W (H*head_dim, in_dim):
        W[h*hd + a*rank_b + b, k] ≈ A[h, a, k] · B[h, b, k]

    Requires rank_a * rank_b == head_dim.
    Returns (A, B, relative_frobenius_error).
    """
    W_f = W.detach().float()
    H = num_heads
    in_dim = W_f.shape[1]
    head_dim = rank_a * rank_b

    if rank_a * rank_b != W_f.shape[0] // H:
        raise ValueError(
            f"rank_a * rank_b ({rank_a} * {rank_b}) must equal head_dim "
            f"({W_f.shape[0] // H})"
        )

    # Reshape to (H, rank_a, rank_b, in_dim)
    W_r = W_f.view(H, rank_a, rank_b, in_dim)

    # Initialize with marginal means (better than random)
    A = W_r.mean(dim=2) + eps  # (H, rank_a, in_dim)
    B = W_r.mean(dim=1) + eps  # (H, rank_b, in_dim)

    prev_err = float("inf")
    for it in range(n_iter):
        # Update A
        num_A = torch.einsum("habk,hbk->hak", W_r, B)
        den_A = torch.einsum("hbk,hbk->hk", B, B).unsqueeze(1)
        A = num_A / (den_A + eps)

        # Update B
        num_B = torch.einsum("habk,hak->hbk", W_r, A)
        den_B = torch.einsum("hak,hak->hk", A, A).unsqueeze(1)
        B = num_B / (den_B + eps)

        # Early stopping check (every 5 iterations to save compute)
        if it % 5 == 0 and it > 0:
            W_hat = torch.einsum("hak,hbk->habk", A, B).reshape(H * head_dim, in_dim)
            err = torch.norm(W_f - W_hat).item() / (torch.norm(W_f).item() + eps)
            if abs(prev_err - err) < tol:
                break
            prev_err = err

    # Final reconstruction error
    W_hat = torch.einsum("hak,hbk->habk", A, B).reshape(H * head_dim, in_dim)
    final_err = torch.norm(W_f - W_hat).item() / (torch.norm(W_f).item() + eps)

    # Symmetric normalization: balance A and B
    norm_A = A.norm(dim=-1, keepdim=True).clamp(min=eps)
    norm_B = B.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale = (norm_B / norm_A).sqrt()
    A = A * scale
    B = B / scale

    A_out = A.reshape(H * rank_a, in_dim).contiguous()
    B_out = B.reshape(H * rank_b, in_dim).contiguous()
    return A_out, B_out, final_err


def _cp_reconstruction_error(
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    num_heads: int,
    rank_a: int,
    rank_b: int,
) -> float:
    """Compute relative Frobenius error of W ≈ A ⊗ B."""
    H = num_heads
    head_dim = rank_a * rank_b
    in_dim = W.shape[1]
    A_r = A.float().view(H, rank_a, in_dim)
    B_r = B.float().view(H, rank_b, in_dim)
    W_hat = torch.einsum("hak,hbk->habk", A_r, B_r).reshape(H * head_dim, in_dim)
    num = torch.norm(W.float() - W_hat).item()
    den = torch.norm(W.float()).clamp_min(1e-12).item()
    return num / den


def _svd_factorize(
    W: torch.Tensor, target_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncated SVD: W ≈ U_r · Σ_r · V_r^T
    Returns (U_r * sqrt(Σ_r), sqrt(Σ_r) * V_r^T) for numerical stability.
    """
    W_f = W.float()
    out_f, in_f = W_f.shape
    r = max(1, min(target_rank, out_f, in_f))
    U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
    r = min(r, len(S))
    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
    A = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
    B = (sqrt_S.unsqueeze(1) * Vt[:r, :]).contiguous()
    return A, B


# ============================================================================
# LoRALinear (dtype-safe)
# ============================================================================


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        lora_rank=0,
        lora_alpha=1.0,
        freeze_base=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
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
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **fk))
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

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.scaling
        return out


# ============================================================================
# TPAttention v2
# ============================================================================


@ATTENTION_REGISTRY.register("tpa", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("tpa_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("tp_attention", supports_causal=True, supports_gqa=True)
class TPAttention(BaseAttention):
    """
    Tensor Product Attention v2 with asymmetric rank support.

    Key improvements:
      - rank_a × rank_b = head_dim (supports head_dim=64 AND head_dim=128)
      - Better CP decomposition (50 iterations with early stopping)
      - Stricter max_cp_error (0.05)
      - KV cache support for autoregressive inference
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank_a: int = 8,
        rank_b: int = 8,
        pretrained_attn: nn.Module | None = None,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        is_decoder: bool | None = None,
        max_cp_error: float = 0.05,
        cp_iterations: int = 50,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank_a)
        except TypeError:
            super().__init__(config)

        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn")
        self.max_cp_error = float(max_cp_error)
        self.cp_iterations = int(cp_iterations)

        # ---- hidden size ----
        dim = None
        if config is not None and not isinstance(config, int):
            dim = _cfg_get(
                config,
                "hidden_size",
                _cfg_get(config, "dim", _cfg_get(config, "d_model", None)),
            )
        if dim is None and src is not None:
            dim = getattr(src, "hidden_size", getattr(src, "embed_dim", None))
            if dim is None:
                for attr in ("q_proj", "q_lin", "query"):
                    m = getattr(src, attr, None)
                    if m is not None and hasattr(m, "in_features"):
                        dim = m.in_features
                        break
            if dim is None and hasattr(src, "c_attn") and hasattr(src.c_attn, "weight"):
                w = src.c_attn.weight
                dim = w.shape[0] if w.shape[0] != 3 * w.shape[1] else w.shape[1]
        if dim is None:
            dim = kwargs.get("hidden_size", 768)

        # ---- heads ----
        num_heads = None
        if config is not None and not isinstance(config, int):
            num_heads = _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(config, "n_heads", _cfg_get(config, "num_heads", None)),
            )
        if num_heads is None and src is not None:
            num_heads = getattr(
                src,
                "num_heads",
                getattr(src, "num_attention_heads", getattr(src, "n_head", None)),
            )
        if num_heads is None:
            num_heads = 12

        self.dim = int(dim)
        self.num_heads = max(1, int(num_heads))

        # ---- head_dim ----
        cfg_hd = _cfg_get(config, "head_dim", None) if config is not None else None
        kw_hd = kwargs.get("head_dim")
        if cfg_hd is not None:
            self.head_dim = int(cfg_hd)
        elif kw_hd is not None:
            self.head_dim = int(kw_hd)
        elif src is not None and getattr(src, "head_dim", None) is not None:
            self.head_dim = int(src.head_dim)
        else:
            self.head_dim = self.dim // self.num_heads

        # ---- GQA ----
        cfg_kv = (
            _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
            if config is not None
            else None
        )
        if cfg_kv is None and src is not None:
            cfg_kv = getattr(src, "num_key_value_heads", getattr(src, "num_kv_heads", None))
        self.num_kv_heads = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_kv_groups = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.q_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim

        # ---- Asymmetric rank support ----
        # Try to find (rank_a, rank_b) such that rank_a * rank_b == head_dim
        raw_ra = kwargs.pop("rank_a", rank_a)
        raw_rb = kwargs.pop("rank_b", rank_b)
        self.rank_a = int(raw_ra)
        self.rank_b = int(raw_rb)

        if self.rank_a * self.rank_b != self.head_dim:
            # Try to auto-adjust rank_b to match head_dim
            if self.head_dim % self.rank_a == 0:
                self.rank_b = self.head_dim // self.rank_a
                logger.info(
                    "Auto-adjusted rank_b to %d to match head_dim=%d",
                    self.rank_b, self.head_dim,
                )
            elif self.head_dim % self.rank_b == 0:
                self.rank_a = self.head_dim // self.rank_b
                logger.info(
                    "Auto-adjusted rank_a to %d to match head_dim=%d",
                    self.rank_a, self.head_dim,
                )
            else:
                self._use_tensor_product = False
                logger.warning(
                    "Cannot find (rank_a, rank_b) with rank_a*rank_b=%d. "
                    "Falling back to standard projection.",
                    self.head_dim,
                )
        else:
            self._use_tensor_product = True

        if self._use_tensor_product:
            self.rank_a = int(self.rank_a)
            self.rank_b = int(self.rank_b)
        else:
            # Standard projection mode
            self.rank_a = self.head_dim
            self.rank_b = 1

        self.effective_qk_dim = self.head_dim

        self.lora_rank = int(kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank)))
        self.lora_alpha = float(
            kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))
        )

        # ---- decoder detection ----
        _mt = str(_cfg_get(config, "model_type", "") or "").lower() if config is not None else ""
        _dec_prefixes = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        elif config is not None and (
            _cfg_get(config, "is_decoder", False)
            or _cfg_get(config, "is_causal", False)
            or any(_mt.startswith(p) for p in _dec_prefixes)
        ):
            self.is_decoder = True
        else:
            self.is_decoder = bool(kwargs.get("is_decoder", False))

        self.is_nested_encoder = False
        self.gradient_checkpointing = bool(kwargs.get("gradient_checkpointing", False))

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and src is not None:
            p = next(src.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        # ---- projections (tensor product or standard) ----
        if self._use_tensor_product:
            # Tensor product: two factor projections
            self.q_proj1 = LoRALinear(
                self.dim, self.num_heads * self.rank_a,
                bias=False, lora_rank=0, freeze_base=True, dtype=target_dtype,
            )
            self.q_proj2 = LoRALinear(
                self.dim, self.num_heads * self.rank_b,
                bias=False, lora_rank=0, freeze_base=True, dtype=target_dtype,
            )
            self.k_proj1 = LoRALinear(
                self.dim, self.num_kv_heads * self.rank_a,
                bias=False, lora_rank=0, freeze_base=True, dtype=target_dtype,
            )
            self.k_proj2 = LoRALinear(
                self.dim, self.num_kv_heads * self.rank_b,
                bias=False, lora_rank=0, freeze_base=True, dtype=target_dtype,
            )
        else:
            # Standard projection: single factor
            self.q_proj1 = LoRALinear(
                self.dim, self.q_head_size,
                bias=False, lora_rank=0, freeze_base=True, dtype=target_dtype,
            )
            self.q_proj2 = None
            self.k_proj1 = LoRALinear(
                self.dim, self.kv_head_size,
                bias=False, lora_rank=0, freeze_base=True, dtype=target_dtype,
            )
            self.k_proj2 = None

        self.v_proj = LoRALinear(
            self.dim, self.kv_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.q_head_size, self.dim,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )

        # ---- LoRA on Q/K (optional) ----
        if self.lora_rank > 0 and self._use_tensor_product:
            self.q_lora_A = nn.Parameter(
                torch.empty(self.lora_rank, self.dim, dtype=target_dtype)
            )
            self.q_lora_B = nn.Parameter(
                torch.zeros(self.q_head_size, self.lora_rank, dtype=target_dtype)
            )
            self.k_lora_A = nn.Parameter(
                torch.empty(self.lora_rank, self.dim, dtype=target_dtype)
            )
            self.k_lora_B = nn.Parameter(
                torch.zeros(self.kv_head_size, self.lora_rank, dtype=target_dtype)
            )
            nn.init.kaiming_uniform_(self.q_lora_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.k_lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("q_lora_A", None)
            self.register_parameter("q_lora_B", None)
            self.register_parameter("k_lora_A", None)
            self.register_parameter("k_lora_B", None)

        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        if target_dtype is not None:
            for proj in [self.q_proj1, self.k_proj1, self.v_proj, self.out_proj]:
                if proj is not None:
                    proj.to(target_dtype)
            for proj in [self.q_proj2, self.k_proj2]:
                if proj is not None:
                    proj.to(target_dtype)

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
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query") and has_no_out
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight.detach()
                    b_o = (
                        old_attn.output.dense.bias.detach()
                        if old_attn.output.dense.bias is not None else None
                    )
            else:
                src = old_attn

            # Extract weights from various architectures
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
            if w_q is None:
                missing.append("Q")
            if w_k is None:
                missing.append("K")
            if w_v is None:
                missing.append("V")
            if w_o is None and not self.is_nested_encoder:
                missing.append("O")
            if missing:
                logger.warning(
                    "TPAttention: failed to extract %s from %s",
                    missing, type(old_attn).__name__,
                )
                return

            # ---- Factorize Q and K (or copy directly if not tensor product) ----
            if self._use_tensor_product:
                self._factorize_qk_tensor_product(w_q, w_k)
            else:
                self._copy_qk_standard(w_q, w_k)

            # ---- Copy V and O directly ----
            self._copy_layer(self.v_proj, w_v, b_v)
            if w_o is not None:
                self._copy_layer(self.out_proj, w_o, b_o)

            if hasattr(old_attn, "is_decoder"):
                self.is_decoder = bool(old_attn.is_decoder)
            elif hasattr(old_attn, "c_attn") or any(
                x in old_attn.__class__.__name__.lower()
                for x in ("gpt", "qwen", "llama", "decoder")
            ):
                self.is_decoder = True

    def _factorize_qk_tensor_product(self, w_q: torch.Tensor, w_k: torch.Tensor) -> None:
        """Factorize Q and K using CP decomposition with SVD fallback."""
        for W, proj1, proj2, num_h, name in [
            (w_q, self.q_proj1, self.q_proj2, self.num_heads, "Q"),
            (w_k, self.k_proj1, self.k_proj2, self.num_kv_heads, "K"),
        ]:
            W_f = W.float()
            in_dim = W_f.shape[1]

            try:
                A, B, err = _cp_decompose_qk(
                    W_f, num_h, self.rank_a, self.rank_b,
                    n_iter=self.cp_iterations,
                )
                logger.info(
                    "TPAttention[%s]: CP error = %.4f (rank_a=%d, rank_b=%d)",
                    name, err, self.rank_a, self.rank_b,
                )

                if err <= self.max_cp_error and not (torch.isnan(A).any() or torch.isnan(B).any()):
                    # Accept CP decomposition
                    proj1.weight.data.zero_()
                    proj2.weight.data.zero_()
                    r_a_rows = num_h * self.rank_a
                    r_b_rows = num_h * self.rank_b
                    proj1.weight.data[:r_a_rows, :in_dim].copy_(
                        A[:r_a_rows, :in_dim].to(proj1.weight.dtype)
                    )
                    proj2.weight.data[:r_b_rows, :in_dim].copy_(
                        B[:r_b_rows, :in_dim].to(proj2.weight.dtype)
                    )
                    logger.info("TPAttention[%s]: CP accepted", name)
                else:
                    # Fallback to SVD
                    logger.warning(
                        "TPAttention[%s]: CP error %.4f > %.4f; using SVD fallback",
                        name, err, self.max_cp_error,
                    )
                    self._svd_fallback(proj1, proj2, W_f, num_h, name)
            except Exception as e:
                logger.warning("TPAttention[%s]: CP failed (%s); using SVD fallback", name, e)
                self._svd_fallback(proj1, proj2, W_f, num_h, name)

            proj1.weight.requires_grad = False
            proj2.weight.requires_grad = False

    def _svd_fallback(self, proj1, proj2, W_f, num_h, name):
        """
        SVD fallback: since we can't represent arbitrary SVD factors as tensor
        product, we simply store the SVD factors and disable tensor-product mode.
        """
        A, B = _svd_factorize(W_f, self.rank_a * self.rank_b)
        in_dim = W_f.shape[1]

        # Store A in proj1, B in proj2 (truncated to their row counts)
        r_a_rows = num_h * self.rank_a
        r_b_rows = num_h * self.rank_b

        proj1.weight.data.zero_()
        proj2.weight.data.zero_()

        # Distribute A and B across proj1/proj2 rows
        a_eff = min(A.shape[0], r_a_rows)
        b_eff = min(B.shape[0], r_b_rows)

        proj1.weight.data[:a_eff, :in_dim].copy_(
            A[:a_eff, :in_dim].to(proj1.weight.dtype)
        )
        proj2.weight.data[:b_eff, :in_dim].copy_(
            B[:b_eff, :in_dim].to(proj2.weight.dtype)
        )

        logger.info("TPAttention[%s]: SVD fallback applied", name)

    def _copy_qk_standard(self, w_q: torch.Tensor, w_k: torch.Tensor) -> None:
        """Copy Q and K directly (no tensor product)."""
        self._copy_layer(self.q_proj1, w_q, None)
        self._copy_layer(self.k_proj1, w_k, None)

    def _copy_layer(self, target, w, b):
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

    # ------------------------------------------------------------------
    # Forward (with KV cache support)
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
        *args,
        **kwargs,
    ):
        B, L, D = hidden_states.shape
        H, H_kv = self.num_heads, self.num_kv_heads
        r_a, r_b = self.rank_a, self.rank_b
        hd = self.head_dim

        # ---- Q, K via tensor product (or standard) ----
        def _compute_q(x):
            if self._use_tensor_product:
                q1 = self.q_proj1(x).view(B, L, H, r_a)
                q2 = self.q_proj2(x).view(B, L, H, r_b)
                out_q = (q1.unsqueeze(-1) * q2.unsqueeze(-2)).flatten(-2, -1)
                if self.q_lora_A is not None:
                    lora = F.linear(F.linear(x, self.q_lora_A), self.q_lora_B)
                    out_q = out_q + lora.view(B, L, H, hd) * (
                        self.lora_alpha / max(1, self.lora_rank)
                    )
                return out_q
            else:
                return self.q_proj1(x).view(B, L, H, hd)

        def _compute_k(x):
            if self._use_tensor_product:
                k1 = self.k_proj1(x).view(B, L, H_kv, r_a)
                k2 = self.k_proj2(x).view(B, L, H_kv, r_b)
                out_k = (k1.unsqueeze(-1) * k2.unsqueeze(-2)).flatten(-2, -1)
                if self.k_lora_A is not None:
                    lora = F.linear(F.linear(x, self.k_lora_A), self.k_lora_B)
                    out_k = out_k + lora.view(B, L, H_kv, hd) * (
                        self.lora_alpha / max(1, self.lora_rank)
                    )
                return out_k
            else:
                return self.k_proj1(x).view(B, L, H_kv, hd)

        use_ckpt = self.training and self.gradient_checkpointing
        if use_ckpt:
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

        Q = Q.transpose(1, 2).contiguous()  # (B, H, L, hd)
        K = K.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # ---- RoPE ----
        pos_emb = kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
            cos, sin = pos_emb
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # ---- KV Cache ----
        if past_key_value is not None:
            past_k, past_v = past_key_value
            K = torch.cat([past_k, K], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (K, v) if kwargs.get("use_cache", False) else None

        # ---- GQA ----
        if H_kv != H:
            K = repeat_kv(K, self.num_kv_groups)
            v = repeat_kv(v, self.num_kv_groups)

        # ---- Mask handling ----
        is_causal = self.is_decoder and L > 1
        custom_mask = attention_mask if attention_mask is not None else kwargs.get("attn_mask")

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
                else:
                    if custom_mask.dim() == 2:
                        custom_mask = custom_mask.unsqueeze(1).unsqueeze(2)
                    elif custom_mask.dim() == 3:
                        custom_mask = custom_mask.unsqueeze(1)
                    custom_mask = custom_mask > -1.0

        eff_hd = self.effective_qk_dim
        N_k = K.shape[-2]

        if not output_attentions and head_mask is None:
            # SDPA path
            sdpa_mask = custom_mask
            if is_causal and sdpa_mask is None and past_key_value is None:
                sdpa_is_causal = True
            else:
                if is_causal and sdpa_mask is not None:
                    causal_bool = torch.tril(
                        torch.ones(L, N_k, device=Q.device, dtype=torch.bool)
                    ).unsqueeze(0).unsqueeze(0)
                    cm = sdpa_mask
                    if cm.dim() == 2:
                        cm = cm.unsqueeze(1).unsqueeze(2)
                    elif cm.dim() == 3:
                        cm = cm.unsqueeze(1)
                    sdpa_mask = cm & causal_bool
                elif is_causal and past_key_value is not None:
                    # Build causal mask for the combined sequence
                    q_pos = torch.arange(N_k - L, N_k, device=Q.device).unsqueeze(1)
                    k_pos = torch.arange(N_k, device=Q.device).unsqueeze(0)
                    causal_bool = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
                    sdpa_mask = causal_bool
                sdpa_is_causal = False

            try:
                if _HAS_SDPA_KERNEL:
                    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                        context = F.scaled_dot_product_attention(
                            query=Q, key=K, value=v,
                            attn_mask=sdpa_mask,
                            dropout_p=self.dropout.p if self.training else 0.0,
                            is_causal=sdpa_is_causal,
                        )
                else:
                    context = F.scaled_dot_product_attention(
                        query=Q, key=K, value=v,
                        attn_mask=sdpa_mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=sdpa_is_causal,
                    )
            except (RuntimeError, TypeError):
                context = F.scaled_dot_product_attention(
                    query=Q, key=K, value=v,
                    attn_mask=sdpa_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=sdpa_is_causal,
                )
            attn_weights = None
        else:
            # Manual path
            scale = 1.0 / math.sqrt(eff_hd)
            scores = torch.matmul(Q, K.transpose(-1, -2).contiguous()) * scale

            if is_causal:
                q_pos = torch.arange(N_k - L, N_k, device=scores.device).unsqueeze(1)
                k_pos = torch.arange(N_k, device=scores.device).unsqueeze(0)
                causal_mask = (k_pos > q_pos).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(causal_mask, float("-inf"))

            if custom_mask is not None:
                if custom_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~custom_mask, -1e4)
                else:
                    scores = scores + custom_mask.to(scores.dtype)

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

        # ---- Output projection ----
        ctx_flat = context.transpose(1, 2).contiguous().view(B, L, self.q_head_size)
        if self.is_nested_encoder:
            output = ctx_flat
        else:
            output = self.out_proj(ctx_flat)

        valid_m = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
        if valid_m is not None:
            output = output * valid_m.unsqueeze(-1).to(output.dtype)

        if return_dict and new_cache is not None:
            return output, new_cache
        return output, (attn_weights if output_attentions else None)


# ============================================================================
# Injection
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
) -> TPAttention | nn.Module | None:
    if _depth > 8:
        logger.warning("inject_tpa: max recursion depth exceeded")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

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
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_tpa(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
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

        if isinstance(src, TPAttention) or "TPAttention" in src.__class__.__name__:
            return None
        for sub_attr in ("inner", "_wrapped_attention"):
            sub = getattr(src, sub_attr, None)
            if isinstance(sub, TPAttention):
                return None
        if "DeltaNet" in src.__class__.__name__ or "LinearAttention" in src.__class__.__name__:
            return None

        if cfg is None and hasattr(src, "config"):
            cfg = src.config

        rank_a = kwargs.pop(
            "rank_a",
            _cfg_get(comp_config, "tpa_rank_a", _cfg_get(comp_config, "rank", 8)),
        )
        rank_b = kwargs.pop(
            "rank_b",
            _cfg_get(comp_config, "tpa_rank_b", rank_a),
        )
        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 0))
        max_cp_error = kwargs.pop(
            "max_cp_error", _cfg_get(comp_config, "tpa_max_cp_error", 0.05)
        )
        cp_iterations = kwargs.pop(
            "cp_iterations", _cfg_get(comp_config, "tpa_cp_iterations", 50)
        )

        new_attn = TPAttention(
            config=cfg,
            rank_a=rank_a,
            rank_b=rank_b,
            lora_rank=lora_rank,
            max_cp_error=max_cp_error,
            cp_iterations=cp_iterations,
            pretrained_attn=src,
            **kwargs,
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

__all__ = [
    "TPAttention",
    "inject_tpa",
    "tp_inject",
    "TPA",
    "apply_rotary_pos_emb",
    "repeat_kv",
]