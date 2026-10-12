"""
LoLA: Low-Rank Linear Attention with Sparse Caching.

Combines factorized low-rank projections with selective multi-system cached
retrieval, trainable LoRA residual adapters, frozen SVD base weights, GQA,
RoPE, and causal decoding.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import apply_rotary_pos_emb
from neuralops.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================
def _safe_mask_value(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


def _safe_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """[LOLA-FIX-06] Numerically stable softmax."""
    x = scores.float()
    m = x.amax(dim=dim, keepdim=True)
    m_safe = torch.where(m < -1e3, torch.zeros_like(m), m)
    exp_x = torch.exp((x - m_safe).clamp(max=0.0))
    return exp_x / exp_x.sum(dim=dim, keepdim=True).clamp(min=1e-9)


# ===========================================================================
# LowRankCache (ring-buffer implementation)
# ===========================================================================
class LowRankCache(nn.Module):
    """
    [LOLA-FIX-02] Ring-buffer streaming cache.

    Three subsystems:
        * recent_k/v:     sliding window of last `max_len` tokens
        * sparse_k/v:     every-4th-token subsample of last `max_len//2`
        * low_rank_k/v:   projection onto a low-rank subspace, refreshed every 50 steps

    All buffers are pre-allocated with fixed shape; `update()` uses index-copy,
    never `torch.cat` + `clone`.
    """

    def __init__(self, max_len: int = 128, low_rank_k: int = 16) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.low_rank_k = int(low_rank_k)
        # Real buffers are lazily initialized on first update (need B*H, D)
        self.register_buffer("_init_flag", torch.zeros(1, dtype=torch.bool), persistent=False)
        self._B_H: int = 0
        self._D: int = 0
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

        # Pre-allocated buffers (created lazily)
        self._recent_k: torch.Tensor | None = None
        self._recent_v: torch.Tensor | None = None
        self._recent_pos: int = 0  # write cursor
        self._recent_filled: int = 0

        self._sparse_k: torch.Tensor | None = None
        self._sparse_v: torch.Tensor | None = None
        self._sparse_pos: int = 0
        self._sparse_filled: int = 0

        self._low_rank_k_buf: torch.Tensor | None = None
        self._low_rank_v_buf: torch.Tensor | None = None

        self._cache_proj: torch.Tensor | None = None
        self._update_step: int = 0

    # ------------------------------------------------------------------
    def _ensure_buffers(self, B_H: int, D: int, device: torch.device, dtype: torch.dtype) -> None:
        """Allocate buffers on first call or when shape changes."""
        if (
            self._B_H == B_H
            and self._D == D
            and self._device == device
            and self._dtype == dtype
            and self._recent_k is not None
        ):
            return

        self._B_H = B_H
        self._D = D
        self._device = device
        self._dtype = dtype

        max_recent = self.max_len
        max_sparse = self.max_len // 2

        self._recent_k = torch.zeros(B_H, max_recent, D, device=device, dtype=dtype)
        self._recent_v = torch.zeros(B_H, max_recent, D, device=device, dtype=dtype)
        self._recent_pos = 0
        self._recent_filled = 0

        self._sparse_k = torch.zeros(B_H, max_sparse, D, device=device, dtype=dtype)
        self._sparse_v = torch.zeros(B_H, max_sparse, D, device=device, dtype=dtype)
        self._sparse_pos = 0
        self._sparse_filled = 0

        self._low_rank_k_buf = None
        self._low_rank_v_buf = None
        self._update_step = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all state, keep buffer capacities."""
        if self._recent_k is not None:
            self._recent_k.zero_()
            self._recent_v.zero_()
            self._sparse_k.zero_()
            self._sparse_v.zero_()
        self._recent_pos = 0
        self._recent_filled = 0
        self._sparse_pos = 0
        self._sparse_filled = 0
        self._low_rank_k_buf = None
        self._low_rank_v_buf = None
        self._update_step = 0

    # ------------------------------------------------------------------
    def update(self, K: torch.Tensor, V: torch.Tensor) -> None:
        """
        [LOLA-FIX-02] Incremental update via index-copy (no cat, no clone).

        Args:
            K, V: (B, H, N, D)
        """
        B, H, N, D = K.shape
        k_flat = K.detach().reshape(B * H, N, D)
        v_flat = V.detach().reshape(B * H, N, D)

        self._ensure_buffers(B * H, D, k_flat.device, k_flat.dtype)

        assert self._recent_k is not None
        assert self._sparse_k is not None

        max_recent = self._recent_k.size(1)
        max_sparse = self._sparse_k.size(1)

        # ---- Recent cache: append with wraparound ----
        take = min(N, max_recent)
        src = k_flat[:, -take:, :]           # (B*H, take, D)
        src_v = v_flat[:, -take:, :]

        end = self._recent_pos + take
        if end <= max_recent:
            self._recent_k[:, self._recent_pos:end, :].copy_(src)
            self._recent_v[:, self._recent_pos:end, :].copy_(src_v)
        else:
            first = max_recent - self._recent_pos
            self._recent_k[:, self._recent_pos:, :].copy_(src[:, :first, :])
            self._recent_v[:, self._recent_pos:, :].copy_(src_v[:, :first, :])
            rest = take - first
            self._recent_k[:, :rest, :].copy_(src[:, first:, :])
            self._recent_v[:, :rest, :].copy_(src_v[:, first:, :])

        self._recent_pos = end % max_recent
        self._recent_filled = min(max_recent, self._recent_filled + take)

        # ---- Sparse cache: every 4th token from the new block ----
        sparse_idx = torch.arange(0, N, 4, device=K.device)
        sparse_k = k_flat[:, sparse_idx, :]
        sparse_v = v_flat[:, sparse_idx, :]
        take_s = min(sparse_k.size(1), max_sparse)

        if take_s > 0:
            src_s = sparse_k[:, -take_s:, :]
            src_sv = sparse_v[:, -take_s:, :]
            end_s = self._sparse_pos + take_s
            if end_s <= max_sparse:
                self._sparse_k[:, self._sparse_pos:end_s, :].copy_(src_s)
                self._sparse_v[:, self._sparse_pos:end_s, :].copy_(src_sv)
            else:
                first = max_sparse - self._sparse_pos
                self._sparse_k[:, self._sparse_pos:, :].copy_(src_s[:, :first, :])
                self._sparse_v[:, self._sparse_pos:, :].copy_(src_sv[:, :first, :])
                rest = take_s - first
                self._sparse_k[:, :rest, :].copy_(src_s[:, first:, :])
                self._sparse_v[:, :rest, :].copy_(src_sv[:, first:, :])

            self._sparse_pos = end_s % max_sparse
            self._sparse_filled = min(max_sparse, self._sparse_filled + take_s)

        # ---- Low-rank subspace refresh (every 50 steps) ----
        self._update_step += 1
        if self._recent_filled >= self.low_rank_k and (
            self._update_step % 50 == 1 or self._low_rank_k_buf is None
        ):
            self._refresh_low_rank()

    # ------------------------------------------------------------------
    def _refresh_low_rank(self) -> None:
        """Recompute low-rank subspace projection from the recent window."""
        if self._recent_k is None or self._recent_filled < self.low_rank_k:
            return

        with torch.no_grad():
            # Gather valid portion of ring buffer
            valid = self._recent_filled
            if self._recent_pos == 0:
                rk = self._recent_k[:, :valid, :]
                rv = self._recent_v[:, :valid, :]
            else:
                if valid <= self._recent_k.size(1):
                    rk = torch.cat(
                        [
                            self._recent_k[:, self._recent_pos :, :],
                            self._recent_k[:, : self._recent_pos, :],
                        ],
                        dim=1,
                    )
                    rv = torch.cat(
                        [
                            self._recent_v[:, self._recent_pos :, :],
                            self._recent_v[:, : self._recent_pos, :],
                        ],
                        dim=1,
                    )
                else:
                    rk = self._recent_k
                    rv = self._recent_v

            rk_f = rk.float()
            rv_f = rv.float()

            # Random orthogonal projection (fixed per cache instance)
            D = self._D
            r = self.low_rank_k
            if (
                self._cache_proj is None
                or self._cache_proj.size(0) != D
                or self._cache_proj.size(1) != r
                or self._cache_proj.device != rk.device
            ):
                proj = torch.empty(D, r, device=rk.device, dtype=torch.float32)
                nn.init.orthogonal_(proj)
                self._cache_proj = proj

            # QR-based projection onto r-dim subspace
            try:
                y_k = torch.matmul(rk_f, self._cache_proj)  # (B*H, valid, r)
                q_k, _ = torch.linalg.qr(y_k)
                self._low_rank_k_buf = torch.matmul(
                    q_k,
                    torch.matmul(q_k.transpose(-2, -1).contiguous(), rk_f),
                ).to(rk.dtype)

                y_v = torch.matmul(rv_f, self._cache_proj)
                q_v, _ = torch.linalg.qr(y_v)
                self._low_rank_v_buf = torch.matmul(
                    q_v,
                    torch.matmul(q_v.transpose(-2, -1).contiguous(), rv_f),
                ).to(rv.dtype)
            except Exception as e:
                logger.debug("LoLA low-rank cache refresh failed: %s", e)

    # ------------------------------------------------------------------
    def get_cached(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return concatenated cache tensors (B*H, C, D)."""
        if self._recent_k is None or self._recent_filled == 0:
            return None, None

        valid = self._recent_filled
        if self._recent_pos == 0:
            rk = self._recent_k[:, :valid, :]
            rv = self._recent_v[:, :valid, :]
        else:
            if valid <= self._recent_k.size(1):
                rk = torch.cat(
                    [
                        self._recent_k[:, self._recent_pos :, :],
                        self._recent_k[:, : self._recent_pos, :],
                    ],
                    dim=1,
                )
                rv = torch.cat(
                    [
                        self._recent_v[:, self._recent_pos :, :],
                        self._recent_v[:, : self._recent_pos, :],
                    ],
                    dim=1,
                )
            else:
                rk = self._recent_k
                rv = self._recent_v

        tensors_k = [rk]
        tensors_v = [rv]

        if self._sparse_filled > 0 and self._sparse_k is not None:
            sv = self._sparse_filled
            if self._sparse_pos == 0:
                sk = self._sparse_k[:, :sv, :]
                svv = self._sparse_v[:, :sv, :]
            else:
                if sv <= self._sparse_k.size(1):
                    sk = torch.cat(
                        [
                            self._sparse_k[:, self._sparse_pos :, :],
                            self._sparse_k[:, : self._sparse_pos, :],
                        ],
                        dim=1,
                    )
                    svv = torch.cat(
                        [
                            self._sparse_v[:, self._sparse_pos :, :],
                            self._sparse_v[:, : self._sparse_pos, :],
                        ],
                        dim=1,
                    )
                else:
                    sk = self._sparse_k
                    svv = self._sparse_v
            tensors_k.append(sk)
            tensors_v.append(svv)

        if self._low_rank_k_buf is not None:
            tensors_k.append(self._low_rank_k_buf)
            tensors_v.append(self._low_rank_v_buf)

        return torch.cat(tensors_k, dim=1), torch.cat(tensors_v, dim=1)


# ===========================================================================
# LowRankLinear
# ===========================================================================
class LowRankLinear(nn.Module):
    """Factorized linear layer: y = U V x + LoRA(x) + bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        lora_rank: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.lora_rank = min(int(lora_rank), min(in_features, out_features))
        self.lora_scaling = (
            (32.0 / self.lora_rank) if self.lora_rank > 0 else 1.0
        )

        self.U = nn.Parameter(
            torch.empty(out_features, rank), requires_grad=False
        )
        self.V = nn.Parameter(
            torch.empty(rank, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, in_features)
            )
            self.lora_B = nn.Parameter(
                torch.empty(out_features, self.lora_rank)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            u_fp32 = torch.empty_like(self.U, dtype=torch.float32)
            v_fp32 = torch.empty_like(self.V, dtype=torch.float32)
            nn.init.orthogonal_(u_fp32)
            nn.init.orthogonal_(v_fp32)
            self.U.data.copy_(u_fp32.to(self.U.dtype))
            self.V.data.copy_(v_fp32.to(self.V.dtype))
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
            self.lora_B.requires_grad = True

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.U, self.V)
        if (
            self.lora_rank > 0
            and self.lora_A is not None
            and self.lora_B is not None
        ):
            w = w + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if x.dtype != self.V.dtype:
            x = x.to(self.V.dtype)

        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U)

        if (
            self.lora_rank > 0
            and self.lora_A is not None
            and self.lora_B is not None
        ):
            lora_out = (
                F.linear(F.linear(x, self.lora_A), self.lora_B)
                * self.lora_scaling
            )
            out = out + lora_out

        if self.bias is not None:
            out = out + self.bias

        return out.to(orig_dtype)


# ===========================================================================
# LoLAAttention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "lola_attention",
    supports_causal=True,
    supports_gqa=True,
    supports_rope=True,
)
class LoLAAttention(BaseAttention):
    """
    Low-Rank Linear Attention with Sparse Caching.

    Cache subsystems (recent / sparse / low-rank) are only active during
    autoregressive decode (N_q == 1). During prefill and training, the cache
    is bypassed to preserve causal masking and avoid dimension mismatches.
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        model_config: Any | None = None,
        rank: int | None = None,
        lora_rank: int = 16,
        dropout: float = 0.1,
        comp_config: Any | None = None,
        pretrained_attn: nn.Module | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = config if config is not None else model_config

        def _cfg_get(name: str, fallback: Any) -> Any:
            if cfg is not None:
                val = getattr(cfg, name, None)
                if val is not None:
                    return val
            return fallback

        actual_rank = rank if rank is not None else _cfg_get("rank", 32)
        try:
            super().__init__(cfg, actual_rank)
        except TypeError:
            super().__init__(cfg)

        self.layer_idx = int(
            layer_idx if layer_idx is not None else kwargs.get("layer_idx", 0)
        )

        cfg_head_dim = getattr(cfg, "head_dim", None) if cfg is not None else None
        if cfg is not None and not isinstance(cfg, int):
            self.adapter = ModelAdapter(cfg)
            self.num_heads = self.adapter.num_heads or getattr(
                cfg, "num_attention_heads", 12
            )
            self.num_kv_heads = self.adapter.num_key_value_heads or getattr(
                cfg, "num_key_value_heads", self.num_heads
            )
            self.hidden_size = self.adapter.hidden_size or getattr(
                cfg, "hidden_size", 768
            )
            self.head_dim = int(
                cfg_head_dim
                or kwargs.get("head_dim")
                or self.adapter.head_dim
                or (self.hidden_size // max(1, self.num_heads))
            )
            self.kv_repeat = self.adapter.kv_repeat or 1
        else:
            self.adapter = None
            self.num_heads = int(
                kwargs.get("num_heads", kwargs.get("num_attention_heads", 12))
            )
            self.num_kv_heads = int(
                kwargs.get("num_key_value_heads", self.num_heads)
            )
            self.hidden_size = int(kwargs.get("hidden_size", 768))
            self.head_dim = int(
                kwargs.get("head_dim")
                or (self.hidden_size // max(1, self.num_heads))
            )
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # ---- Decoder detection ----
        _mt = str(getattr(cfg, "model_type", "") or "").lower() if cfg else ""
        _decoder_prefixes = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )
        self.is_decoder = bool(
            (cfg is not None and (
                getattr(cfg, "is_decoder", False)
                or getattr(cfg, "causal", False)
                or any(_mt.startswith(p) for p in _decoder_prefixes)
            ))
            or kwargs.get("is_decoder", False)
            or kwargs.get("is_causal", False)
        )

        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.is_nested_encoder: bool = False

        self.rank = int(actual_rank)
        self.lora_rank = int(_cfg_get("lora_rank", lora_rank))

        # ---- Projections ----
        self.q_proj = LowRankLinear(
            self.hidden_size, self.all_head_size, self.rank,
            lora_rank=self.lora_rank, bias=True,
        )
        self.k_proj = LowRankLinear(
            self.hidden_size, self.kv_head_size, self.rank,
            lora_rank=self.lora_rank, bias=True,
        )
        self.v_proj = LowRankLinear(
            self.hidden_size, self.kv_head_size, self.rank,
            lora_rank=self.lora_rank, bias=True,
        )
        self.out_proj = LowRankLinear(
            self.all_head_size, self.hidden_size, self.rank,
            lora_rank=self.lora_rank, bias=True,
        )

        self.dropout = nn.Dropout(dropout)

        # ---- Cache config ----
        max_seq_len = (
            getattr(comp_config, "max_seq_len", 128)
            if comp_config is not None
            else 128
        )
        low_rank_cache_size = (
            getattr(comp_config, "low_rank_cache_size", 16)
            if comp_config is not None
            else 16
        )
        self.cache = LowRankCache(
            max_len=max_seq_len, low_rank_k=low_rank_cache_size
        )
        self.LR_scale = nn.Parameter(torch.tensor(4.0))
        self.sparse_gate = nn.Parameter(torch.tensor(-3.0))

        self.rotary_emb = (
            getattr(pretrained_attn, "rotary_emb", None)
            if pretrained_attn is not None
            else None
        )

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    def reset_cache(self) -> None:
        if hasattr(self, "cache") and hasattr(self.cache, "reset"):
            self.cache.reset()

    # ------------------------------------------------------------------
    # [LOLA-FIX-03] SVD on CPU
    # ------------------------------------------------------------------
    def _set_lowrank_factors(
        self,
        lowrank: LowRankLinear,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        with torch.no_grad():
            # [LOLA-FIX-03] SVD on CPU for numerical stability
            W_cpu = weight.detach().float().cpu()
            try:
                U, S, Vt = torch.linalg.svd(W_cpu, full_matrices=False)
            except Exception as e:
                logger.warning("LoLA SVD failed: %s", e)
                return

            r = min(self.rank, U.size(1), Vt.size(0))
            if r == 0:
                return

            sqrt_s = torch.sqrt(torch.clamp(S[:r], min=1e-12))

            U_r = (U[:, :r] * sqrt_s.unsqueeze(0)).to(
                device=lowrank.U.device, dtype=lowrank.U.dtype
            )
            V_r = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(
                device=lowrank.V.device, dtype=lowrank.V.dtype
            )

            lowrank.U.data.zero_()
            lowrank.V.data.zero_()

            out_d = min(lowrank.U.size(0), U_r.size(0))
            rank_d = min(lowrank.U.size(1), U_r.size(1))
            lowrank.U.data[:out_d, :rank_d].copy_(U_r[:out_d, :rank_d])

            in_d = min(lowrank.V.size(1), V_r.size(1))
            rank_d2 = min(lowrank.V.size(0), V_r.size(0))
            lowrank.V.data[:rank_d2, :in_d].copy_(V_r[:rank_d2, :in_d])

            if lowrank.bias is not None and bias is not None:
                b_val = bias.data if isinstance(bias, nn.Parameter) else bias
                b_val = b_val.to(lowrank.bias.device, lowrank.bias.dtype)
                n = min(lowrank.bias.numel(), b_val.numel())
                lowrank.bias.data[:n].copy_(b_val[:n])

            lowrank.U.requires_grad = False
            lowrank.V.requires_grad = False
            if lowrank.bias is not None:
                lowrank.bias.requires_grad = False

            if lowrank.lora_rank > 0 and lowrank.lora_A is not None:
                nn.init.kaiming_uniform_(lowrank.lora_A, a=math.sqrt(5))
                lowrank.lora_A.requires_grad = True
            if lowrank.lora_rank > 0 and lowrank.lora_B is not None:
                nn.init.normal_(lowrank.lora_B, mean=0.0, std=1e-4)
                lowrank.lora_B.requires_grad = True

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        # 2. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None
            or extracted["k"][0] is None
            or extracted["v"][0] is None
        ):
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                if (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                ):
                    weight = weight.t().contiguous()
                split_size = weight.shape[0] // 3
                extracted["q"] = (
                    weight[:split_size, :],
                    bias_data[:split_size] if bias_data is not None else None,
                )
                extracted["k"] = (
                    weight[split_size : 2 * split_size, :],
                    bias_data[split_size : 2 * split_size]
                    if bias_data is not None
                    else None,
                )
                extracted["v"] = (
                    weight[2 * split_size :, :],
                    bias_data[2 * split_size :] if bias_data is not None else None,
                )

        # 3. GPT-2 c_proj
        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj_mod = src.c_proj
            base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                p_cls = c_proj_mod.__class__.__name__
                b_cls = base_o.__class__.__name__
                if (
                    "Conv1D" in p_cls
                    or "Conv1D" in b_cls
                    or hasattr(c_proj_mod, "nf")
                    or hasattr(base_o, "nf")
                    or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                ):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (
                    w_o,
                    b_o.data.clone() if b_o is not None else None,
                )

        # 4. BERT / RoBERTa nested
        if hasattr(src, "self") and (
            extracted["q"][0] is None
            or extracted["k"][0] is None
            or extracted["v"][0] is None
        ):
            s = src.self
            for proj_key, sub_name in (("q", "query"), ("k", "key"), ("v", "value")):
                mod = getattr(s, sub_name, None)
                if mod is not None:
                    base_mod = getattr(mod, "base_layer", mod)
                    if hasattr(base_mod, "weight"):
                        b = getattr(base_mod, "bias", None)
                        extracted[proj_key] = (
                            base_mod.weight.data.clone(),
                            b.data.clone() if b is not None else None,
                        )

        if (
            hasattr(src, "output")
            and hasattr(src.output, "dense")
            and extracted["o"][0] is None
        ):
            dense = src.output.dense
            base_dense = getattr(dense, "base_layer", dense)
            if hasattr(base_dense, "weight"):
                b = getattr(base_dense, "bias", None)
                extracted["o"] = (
                    base_dense.weight.data.clone(),
                    b.data.clone() if b is not None else None,
                )

        # 5. Flat projections
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        base_l = getattr(layer, "base_layer", layer)
                        if hasattr(base_l, "weight"):
                            b = getattr(base_l, "bias", None)
                            extracted[proj] = (
                                base_l.weight.data.clone(),
                                b.data.clone() if b is not None else None,
                            )
                            break

        # ---- Nested encoder detection ----
        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in (
                "dense", "out_proj", "c_proj", "out_lin",
                "o_proj", "wo", "o", "proj",
            )
        )
        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or (
                cls_name
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                )
                or (cls_name.endswith("SelfAttention") and not has_out_proj)
            )
        ) and not has_out_proj:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning(
                "LoLA copy_weights: missing %s; using random init.", missing
            )

        # Frobenius verification
        for proj_key in ("q", "k", "v", "o"):
            w = extracted[proj_key][0]
            if w is not None:
                norm = float(torch.linalg.norm(w.float()).item())
                if norm < 1e-4:
                    logger.warning(
                        "LoLA: extracted '%s' near-zero norm (%.2e)", proj_key, norm
                    )

        # ---- SVD factorization ----
        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    self._set_lowrank_factors(target, w_oriented, b)
                elif proj_key == "o" and self.is_nested_encoder:
                    # Identity initialization for nested encoder out_proj
                    min_d = min(target.U.size(0), target.V.size(1), self.rank)
                    target.U.data.zero_()
                    target.V.data.zero_()
                    idx = torch.arange(min_d, device=target.U.device)
                    target.U.data[idx, idx] = 1.0
                    target.V.data[idx, idx] = 1.0
                    if target.bias is not None:
                        target.bias.data.zero_()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    # ------------------------------------------------------------------
    # Forward
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
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # ---- Projections ----
        Q = (
            self.q_proj(hidden_states)
            .view(B, N_q, H, d)
            .transpose(1, 2)
            .contiguous()
        )
        K = (
            self.k_proj(kv_input)
            .view(B, N_k, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            self.v_proj(kv_input)
            .view(B, N_k, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE ----
        position_embeddings = kwargs.get("position_embeddings")
        if (
            position_embeddings is not None
            and isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) >= 2
        ):
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))

        # ---- GQA ----
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = d**-0.5

        # ---- Attention scores (no cache yet) ----
        scores_lr = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale

        # ---- Padding mask ----
        mask = attention_mask if attention_mask is not None else kwargs.get("mask")
        mask_val = -1e4 if dtype in (torch.float16, torch.bfloat16) or scores_lr.dtype in (torch.float16, torch.bfloat16) else -1e9
        if mask is not None:
            if mask.dim() == 4:
                if mask.dtype == torch.bool:
                    scores_lr = scores_lr.masked_fill(~mask, mask_val)
                else:
                    scores_lr = scores_lr + mask.to(scores_lr.dtype)
            elif mask.dim() == 3:
                scores_lr = scores_lr + mask.unsqueeze(1).to(scores_lr.dtype)
            else:
                valid_m = get_token_valid_mask(mask, B, N_k, device)
                if valid_m is not None:
                    scores_lr = scores_lr.masked_fill(
                        (~valid_m).unsqueeze(1).unsqueeze(2), mask_val
                    )

        # ---- Causal mask ----
        if self.is_decoder and not is_cross_attention and N_k >= N_q:
            causal_tril = torch.tril(
                torch.ones((N_q, N_k), device=device, dtype=torch.bool),
                diagonal=N_k - N_q,
            ).unsqueeze(0).unsqueeze(0)
            scores_lr = scores_lr.masked_fill(~causal_tril, mask_val)

        # =============================================================
        # [LOLA-FIX-01] Cache is only used during single-token decode.
        # During prefill (N_q > 1) or training, the cache would either
        # cause dimension mismatch or violate causal masking.
        # =============================================================
        is_single_token_decode = (
            self.is_decoder
            and not is_cross_attention
            and N_q == 1
            and not self.training
        )

        use_cache = is_single_token_decode

        if use_cache:
            self.cache.update(K, V)
            cached_k, cached_v = self.cache.get_cached()
        else:
            # Reset stale cache for batch isolation
            self.cache.reset()
            cached_k, cached_v = None, None

        if (
            cached_k is not None
            and cached_v is not None
            and cached_k.size(0) == (B * H)
        ):
            C = cached_k.size(1)
            K_cache = cached_k.view(B, H, C, d)
            V_cache = cached_v.view(B, H, C, d)

            scores_cache = (
                torch.matmul(Q, K_cache.transpose(-2, -1).contiguous())
                * scale
                * torch.sigmoid(
                    torch.clamp(self.sparse_gate, min=-10.0, max=10.0)
                )
            )
            # N_q == 1 here, so cat along dim=-1 is safe
            scores = torch.cat(
                [
                    scores_lr
                    * torch.sigmoid(
                        torch.clamp(self.LR_scale, min=-10.0, max=10.0)
                    ),
                    scores_cache,
                ],
                dim=-1,
            )
            V_comb = torch.cat([V, V_cache], dim=2)
        else:
            scores = scores_lr
            V_comb = V

        # ---- [LOLA-FIX-05, 06] Safe softmax ----
        attn = _safe_softmax(scores, dim=-1).to(dtype)
        attn = self.dropout(attn)

        # ---- head_mask ----
        if head_mask is not None:
            hm = head_mask.to(attn.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(-1)
            attn = attn * hm

        context = torch.matmul(attn, V_comb)
        context = (
            context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        )
        context = torch.nan_to_num(context, nan=0.0)

        # ---- Output projection ----
        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        return output, (attn if output_attentions else None)


# ===========================================================================
# Injection
# ===========================================================================
def inject_lola_attention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    # [LOLA-FIX-08] Recursion guard
    if _depth > 8:
        logger.warning("inject_lola_attention: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = (
        model_config
        if model_config is not None
        else getattr(target, "config", None)
    )

    # ---- Whole-model traversal ----
    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
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
            res = inject_lola_attention(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    # ---- Single-layer injection ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attn", "attention", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if cand is not None and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    if src_attn is None:
        logger.warning(
            "inject_lola_attention: no source attention on %s",
            type(target).__name__,
        )
        return None

    # [LOLA-FIX-08] Idempotency guard
    if isinstance(src_attn, LoLAAttention) or "LoLA" in src_attn.__class__.__name__:
        return None

    if model_config is None and src_attn is not None:
        model_config = getattr(src_attn, "config", None)

    try:
        h_size = (
            getattr(
                model_config,
                "hidden_size",
                getattr(model_config, "dim", 768),
            )
            if model_config
            else 768
        )
        rank = (
            getattr(comp_config, "rank", kwargs.pop("rank", h_size))
            if comp_config is not None
            else kwargs.pop("rank", h_size)
        )
        kwargs.pop("rank", None)

        new_attn = LoLAAttention(
            config=model_config,
            rank=rank,
            comp_config=comp_config,
            pretrained_attn=src_attn,
            layer_idx=layer_idx,
            **kwargs,
        )

        # ---- Move to source device/dtype ----
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        # ---- Splice ----
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
            spliced = True
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
            spliced = True
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
            spliced = True
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
            spliced = True

        if not spliced:
            logger.error(
                "inject_lola_attention: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

        # ---- Preserve training mode ----
        if src_attn is not None:
            new_attn.train(src_attn.training)
        elif target is not None and hasattr(target, "training"):
            new_attn.train(target.training)

        return new_attn
    except Exception as e:
        logger.exception(
            "LoLAAttention injection failed at layer %d: %s", layer_idx, e
        )
        return None


# ===========================================================================
# Aliases
# ===========================================================================
lola_inject = inject_lola_attention
LoLA = LoLAAttention


__all__ = [
    "LoLAAttention",
    "LoLA",
    "LowRankCache",
    "LowRankLinear",
    "inject_lola_attention",
    "lola_inject",
]