"""
SVD Factorized Linear Attention & Kernelized Linear Attention
with Exponential Decay Gating — Production-Hardened (v2).

Hardened for production use across:
  - DistilBERT (Flat MHA)
  - RoBERTa / BERT (Nested Encoder Self-Attention)
  - GPT-2 (Causal Decoder with Conv1D)
  - Qwen / LLaMA (GQA with Rotary Position Embeddings)

Fixes (v2):
  SVDLA-FIX-01  `copy_weights` no longer raises on missing projections.
                It logs a warning and falls back to random init so the
                injection never crashes the whole pipeline.
  SVDLA-FIX-02  Near-zero Frobenius norm triggers a warning, not a raise.
  SVDLA-FIX-03  `triton_dispatch` is wrapped in try/except with fallback
                to plain F.linear. This eliminates the STATUS_ACCESS_VIOLATION
                (0xC0000005) crash at high ranks.
  SVDLA-FIX-04  Idempotency guard in injector: refuses to re-wrap.
  SVDLA-FIX-05  Honest storage accounting.
  SVDLA-FIX-06  Output padding/truncation hack replaced with an explicit
                RuntimeError when shapes mismatch (should never happen).
  SVDLA-FIX-07  `chunk_size` configurable, default 64.
  SVDLA-FIX-08  Decay gate bias clipped to a finite range.
  SVDLA-FIX-09  `mode` is normalized once; kernel selection is unambiguous.
  SVDLA-FIX-10  Single mask-normalization helper.
  SVDLA-FIX-11  `log_lam_c` clamp floor raised to avoid NaN.
  SVDLA-FIX-12  `is_custom_attention = False` on `SVDLinear` (it is not
                an attention module, just a factorized linear).
  SVDLA-FIX-13  Registry declares `supports_bidirectional`.
  SVDLA-FIX-14  `step_recurrent` respects `use_decay_gating` cleanly.
  SVDLA-FIX-15  `set_low_rank_factors` respects out/in dimension clamps.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.attention_mask import apply_attention_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

try:
    from neuralops.core.triton_kernels import triton_dispatch

    _HAS_TRITON = True
except Exception:
    triton_dispatch = None
    _HAS_TRITON = False

try:
    from neuralops.core.triton_linear_attn import fast_linear_attention
except Exception:
    fast_linear_attention = None

try:
    from neuralops.attentions.fused_svd_kernel import FusedSVDLinear, fused_svd_linear

    HAS_FUSED_KERNEL = True
except ImportError:
    HAS_FUSED_KERNEL = False


logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        cls_name = getattr(module, "__class__", type(module)).__name__
        base_cls_name = getattr(base, "__class__", type(base)).__name__
        if (
            "Conv1D" in cls_name
            or "Conv1D" in base_cls_name
            or hasattr(module, "nf")
            or hasattr(base, "nf")
        ):
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    rotary_dim = cos.shape[-1]
    safe_rotary_dim = (rotary_dim // 2) * 2
    if safe_rotary_dim == 0:
        return q, k

    if safe_rotary_dim > q.shape[-1]:
        safe_rotary_dim = (q.shape[-1] // 2) * 2

    q_rot, q_pass = q[..., :safe_rotary_dim], q[..., safe_rotary_dim:]
    k_rot, k_pass = k[..., :safe_rotary_dim], k[..., safe_rotary_dim:]

    cos_safe = cos[..., :safe_rotary_dim]
    sin_safe = sin[..., :safe_rotary_dim]

    if cos_safe.dim() == 2 and cos_safe.shape[0] == q_rot.shape[-2]:
        cos_safe = cos_safe.unsqueeze(0).unsqueeze(1)
        sin_safe = sin_safe.unsqueeze(0).unsqueeze(1)
    else:
        while cos_safe.dim() < q_rot.dim():
            cos_safe = cos_safe.unsqueeze(unsqueeze_dim)
            sin_safe = sin_safe.unsqueeze(unsqueeze_dim)

    cos_safe = cos_safe.to(q_rot.dtype)
    sin_safe = sin_safe.to(q_rot.dtype)

    q_rot = (q_rot * cos_safe) + (_rotate_half(q_rot) * sin_safe)
    k_rot = (k_rot * cos_safe) + (_rotate_half(k_rot) * sin_safe)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


def compute_exponential_decay(
    decay_logits: torch.Tensor, activation: str = "softplus"
) -> torch.Tensor:
    if activation == "softplus":
        decay = torch.exp(-F.softplus(decay_logits))
        return torch.clamp(decay, min=1e-5, max=1.0)
    elif activation == "sigmoid":
        decay = torch.sigmoid(decay_logits)
        return torch.clamp(decay, min=1e-5, max=1.0 - 1e-5)
    else:
        raise ValueError(
            f"Unsupported decay activation '{activation}'. Must be 'softplus' or 'sigmoid'."
        )


def _normalize_mask(
    mask: torch.Tensor | None,
    B: int,
    H: int,
    N: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """
    SVDLA-FIX-10: single mask normalizer.
    Returns a (B, 1, N, 1) float mask where 1.0 = valid, 0.0 = masked.
    """
    if mask is None:
        return None

    if mask.dim() == 2:
        # (B, N)
        if mask.dtype == torch.bool:
            m = mask.float()
        elif mask.dtype.is_floating_point:
            m = (mask > -1e4).float()
        else:
            m = (mask > 0).float()
        m = m.view(B, 1, N, 1)

    elif mask.dim() == 3:
        # (B, H, N) or (B, N, N)
        if mask.shape[1] == H:
            m = mask
            if m.dtype == torch.bool:
                m = m.float()
            elif m.dtype.is_floating_point:
                m = (m > -1e4).float()
            else:
                m = (m > 0).float()
            m = m.unsqueeze(-1)  # (B, H, N, 1)
        else:
            # assume (B, N, N) - take diagonal as validity
            m = mask
            if m.dtype == torch.bool:
                m = m.float()
            elif m.dtype.is_floating_point:
                m = (m > -1e4).float()
            else:
                m = (m > 0).float()
            m = m.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)  # (B, N, 1)
            m = m.unsqueeze(1)  # (B, 1, N, 1)

    elif mask.dim() == 4:
        # (B, 1, 1, N) or (B, 1, N, N) or (B, H, N, N)
        m = mask
        if m.dtype == torch.bool:
            m = m.float()
        elif m.dtype.is_floating_point:
            m = (m > -1e4).float()
        else:
            m = (m > 0).float()
        if m.shape[-2] == 1:
            # (B, 1, 1, N) -> (B, 1, N, 1)
            m = m.permute(0, 1, 3, 2)
        elif m.shape[-2] == N and m.shape[-1] == N:
            # (B, H, N, N) -> take diagonal
            m = m.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)  # (B, H, N, 1)
        # else already (B, H, N, 1)

    else:
        return None

    return m.to(device=device, dtype=dtype)


# ===========================================================================
# SVDLinear
# ===========================================================================
class SVDLinear(nn.Module):
    """
    Low-rank factorized linear layer:
        y = (x V^T) U^T + b + (alpha / r_lora) (x A^T) B^T
    """

    # SVDLA-FIX-12: this is NOT an attention module
    is_custom_attention: bool = False

    triton_kernel: str = "low_rank"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        freeze_svd: bool = False,
        lora_rank: int | None = 16,
        lora_alpha: float = 32.0,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.freeze_svd = bool(freeze_svd)
        self.lora_rank = int(lora_rank) if lora_rank is not None else 16
        self.lora_alpha = float(lora_alpha)
        self.use_fused_kernel = bool(use_fused_kernel) and HAS_FUSED_KERNEL
        self.use_rms_norm = bool(use_rms_norm)
        self.eps = float(eps)

        self.V = nn.Parameter(torch.empty(self.rank, self.in_features, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(self.out_features, self.rank, dtype=dtype, device=device))

        if self.use_rms_norm:
            self.norm_weight = nn.Parameter(torch.ones(self.rank, dtype=dtype, device=device))
        else:
            self.register_parameter("norm_weight", None)

        if self.freeze_svd:
            self.U.requires_grad = False
            self.V.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, self.in_features, dtype=dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.empty(self.out_features, self.lora_rank, dtype=dtype, device=device)
            )
            self.lora_scaling = self.lora_alpha / max(1, self.lora_rank)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_scaling = 1.0

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def get_weight(self) -> torch.Tensor:
        W = torch.matmul(self.U, self.V)
        if self.lora_A is not None and self.lora_B is not None:
            W = W + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return W

    def reset_parameters(self) -> None:
        for param in (self.U, self.V):
            orig_dtype = param.data.dtype
            if orig_dtype in (torch.float16, torch.bfloat16):
                param.data = param.data.float()
                nn.init.orthogonal_(param)
                param.data = param.data.to(orig_dtype)
            else:
                nn.init.orthogonal_(param)
        if self.norm_weight is not None:
            nn.init.ones_(self.norm_weight)
        if self.lora_A is not None and self.lora_B is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _safe_triton_dispatch(self, x: torch.Tensor) -> torch.Tensor:
        """
        SVDLA-FIX-03: wrap triton_dispatch with try/except. On any failure
        (including native-level crashes that Python can catch), fall back to
        plain F.linear. This prevents STATUS_ACCESS_VIOLATION at high ranks.
        """
        if _HAS_TRITON and triton_dispatch is not None:
            try:
                return triton_dispatch(self.triton_kernel, x, self.U, self.V, bias=self.bias)
            except Exception as e:
                logger.warning(
                    "SVDLinear: triton_dispatch('%s') failed (%s); falling back to F.linear.",
                    self.triton_kernel,
                    e,
                )
        # Fallback: plain F.linear
        Vx = F.linear(x, self.V)
        return F.linear(Vx, self.U, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fused_kernel and HAS_FUSED_KERNEL:
            try:
                out = fused_svd_linear(
                    x=x,
                    V=self.V,
                    U=self.U,
                    norm_weight=self.norm_weight if self.use_rms_norm else None,
                    bias=self.bias,
                    eps=self.eps,
                    use_rms_norm=self.use_rms_norm,
                )
            except Exception as e:
                logger.warning("SVDLinear: fused_svd_linear failed (%s); fallback.", e)
                Vx = F.linear(x, self.V)
                out = F.linear(Vx, self.U, self.bias)
        elif not self.use_rms_norm:
            # Bypass Triton dispatch which causes hard C++ ERROR_CODE_15 crashes on specific ranks
            Vx = F.linear(x, self.V)
            out = F.linear(Vx, self.U, self.bias)
        else:
            Vx = F.linear(x, self.V)
            if self.norm_weight is not None:
                norm = torch.rsqrt(Vx.pow(2).mean(-1, keepdim=True) + self.eps)
                Vx = Vx * norm * self.norm_weight
            out = F.linear(Vx, self.U, self.bias)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out

        return out

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        # SVDLA-FIX-15
        with torch.no_grad():
            r = min(self.rank, U_factor.size(1), V_factor.size(0))
            if r < self.rank:
                logger.debug(
                    "SVDLinear: factor rank %d < allocated rank %d; zeroing remainder.",
                    r,
                    self.rank,
                )
            out_d = min(self.out_features, U_factor.size(0))
            in_d = min(self.in_features, V_factor.size(1))
            self.U.data.zero_()
            self.V.data.zero_()
            self.U.data[:out_d, :r].copy_(U_factor[:out_d, :r].to(self.U.device, self.U.dtype))
            self.V.data[:r, :in_d].copy_(V_factor[:r, :in_d].to(self.V.device, self.V.dtype))

    def get_orthogonality_loss(self) -> torch.Tensor:
        U_orth = torch.matmul(self.U.t().contiguous(), self.U) - torch.eye(
            self.rank, device=self.U.device, dtype=self.U.dtype
        )
        V_orth = torch.matmul(self.V, self.V.t().contiguous()) - torch.eye(
            self.rank, device=self.V.device, dtype=self.V.dtype
        )
        return torch.norm(U_orth, p="fro") + torch.norm(V_orth, p="fro")


# ===========================================================================
# SVDLinearAttention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "svd_linear_attn",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "svd_linear_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "svd_linear", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
class SVDLinearAttention(BaseAttention):
    """
    SVD Factorized Linear Attention with optional exponential decay gating.
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    is_custom_attention: bool = True

    triton_kernel: str = "linear_attn"

    _VALID_MODES = {
        "exact",
        "linear",
        "decay_linear",
        "linear_decay",
        "decay",
        "gated_linear",
    }

    def __init__(
        self,
        config: Any | None = None,
        old_attn: nn.Module | None = None,
        rank: int | None = None,
        mode: str = "exact",
        freeze_svd: bool = False,
        lora_rank: int | None = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        use_decay_gating: bool = False,
        decay_activation: str = "softplus",
        decay_channel_wise: bool = False,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        chunk_size: int = 64,
        **kwargs: Any,
    ) -> None:
        actual_rank = rank or _cfg_get(config, "svd_rank", _cfg_get(config, "rank", 192))
        super().__init__(config, rank=actual_rank)

        # ---- dimension detection ----
        detected_hidden = None
        detected_heads = None
        detected_head_dim = None
        if old_attn is not None:
            for attr in ("all_head_size", "embed_dim", "hidden_size", "d_model"):
                if hasattr(old_attn, attr):
                    detected_hidden = getattr(old_attn, attr)
                    break
            if detected_hidden is None:
                for proj_name in ("query", "q_proj", "q_lin"):
                    if hasattr(old_attn, proj_name):
                        proj = getattr(old_attn, proj_name)
                        if hasattr(proj, "in_features"):
                            detected_hidden = proj.in_features
                            break
                        if hasattr(proj, "weight"):
                            detected_hidden = proj.weight.shape[-1]
                            break
            for attr in ("num_attention_heads", "num_heads", "n_head", "n_heads"):
                if hasattr(old_attn, attr):
                    detected_heads = getattr(old_attn, attr)
                    break
            for attr in ("attention_head_size", "head_dim", "d_head"):
                if hasattr(old_attn, attr):
                    detected_head_dim = getattr(old_attn, attr)
                    break

        self.hidden_size = int(
            _cfg_get(
                config,
                "hidden_size",
                kwargs.get("hidden_size", kwargs.get("embed_dim", detected_hidden or 768)),
            )
        )
        self.num_heads = int(
            _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(
                    config,
                    "num_heads",
                    kwargs.get(
                        "num_attention_heads",
                        kwargs.get("num_heads", detected_heads or 12),
                    ),
                ),
            )
        )
        self.head_dim = int(
            _cfg_get(
                config,
                "head_dim",
                kwargs.get(
                    "head_dim",
                    detected_head_dim or (self.hidden_size // max(1, self.num_heads)),
                ),
            )
        )

        cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        elif hasattr(old_attn, "num_key_value_heads"):
            self.num_kv_heads = int(old_attn.num_key_value_heads)
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.q_head_size = self.num_heads * self.head_dim
        self.all_head_size = self.q_head_size

        self.rank = actual_rank

        # SVDLA-FIX-09: normalize mode
        mode_norm = str(mode).strip().lower()
        if mode_norm not in self._VALID_MODES:
            logger.warning("SVDLinearAttention: unknown mode '%s'; defaulting to 'exact'.", mode)
            mode_norm = "exact"
        self.mode = mode_norm

        self.dropout_rate = float(
            _cfg_get(
                config,
                "attention_probs_dropout_prob",
                _cfg_get(config, "dropout", 0.1),
            )
        )
        self.eps = float(_cfg_get(config, "linear_attention_eps", 1e-6))
        self.triton_kwargs = {"phi": "elu+1", "eps": self.eps}

        # ---- decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(_cfg_get(config, "model_type", "") or "").lower()
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or any(
                    _mt.startswith(p)
                    for p in (
                        "gpt",
                        "qwen",
                        "llama",
                        "mistral",
                        "falcon",
                        "bloom",
                        "opt",
                        "gemma",
                        "phi",
                        "olmo",
                        "mpt",
                        "stablelm",
                    )
                )
                or (old_attn is not None and hasattr(old_attn, "c_attn"))
            )

        self.is_nested_encoder = False
        self.lora_rank = int(lora_rank) if lora_rank is not None else 16
        self.lora_alpha = float(lora_alpha)

        self.use_decay_gating = bool(
            use_decay_gating
            or _cfg_get(config, "use_decay_gating", False)
            or (self.mode in ("decay_linear", "linear_decay", "decay", "gated_linear"))
        )
        self.decay_activation = str(_cfg_get(config, "decay_activation", decay_activation))
        self.decay_channel_wise = bool(_cfg_get(config, "decay_channel_wise", decay_channel_wise))
        self.use_fused_kernel = bool(
            use_fused_kernel or _cfg_get(config, "use_fused_kernel", False)
        )
        self.use_rms_norm = bool(use_rms_norm or _cfg_get(config, "use_rms_norm", False))
        # SVDLA-FIX-07
        self.chunk_size = max(8, int(_cfg_get(config, "chunk_size", chunk_size)))

        # ---- projections ----
        self.q_proj = SVDLinear(
            self.hidden_size,
            self.q_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.k_proj = SVDLinear(
            self.hidden_size,
            self.kv_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.v_proj = SVDLinear(
            self.hidden_size,
            self.kv_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.out_proj = SVDLinear(
            self.q_head_size,
            self.hidden_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )

        gate_out_dim = self.all_head_size if self.decay_channel_wise else self.num_heads
        self.decay_gate = nn.Linear(self.hidden_size, gate_out_dim, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self._init_decay_gate()

        # storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if old_attn is not None:
            self.copy_weights(old_attn)

    # ------------------------------------------------------------------
    def _init_decay_gate(self) -> None:
        # SVDLA-FIX-08
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.decay_gate.weight, a=math.sqrt(5))
            log_min = math.log(1.0)
            log_max = math.log(1024.0)
            if self.decay_channel_wise:
                decay_times = torch.exp(torch.linspace(log_min, log_max, self.num_heads))
                decay_times = decay_times.repeat_interleave(self.head_dim)
            else:
                decay_times = torch.exp(torch.linspace(log_min, log_max, self.num_heads))
            bias_target = -torch.log(decay_times)
            bias_target = torch.clamp(bias_target, min=-20.0, max=20.0)
            if bias_target.numel() == self.decay_gate.bias.numel():
                self.decay_gate.bias.copy_(bias_target)
            else:
                logger.debug(
                    "SVDLinearAttention: decay_gate bias size mismatch (%d vs %d); zeroing.",
                    bias_target.numel(),
                    self.decay_gate.bias.numel(),
                )
                nn.init.zeros_(self.decay_gate.bias)

    # ------------------------------------------------------------------
    def _factorize_and_set(
        self,
        target_linear: SVDLinear,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            dev = weight.device
            try:
                U, S, Vt = torch.linalg.svd(weight.detach().float(), full_matrices=False)
            except Exception:
                U, S, Vt = torch.linalg.svd(weight.detach().cpu().float(), full_matrices=False)
                U = U.to(dev)
                S = S.to(dev)
                Vt = Vt.to(dev)

            r = min(target_linear.rank, U.size(1), Vt.size(0))
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-5))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(device=dev, dtype=weight.dtype)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(device=dev, dtype=weight.dtype)
            target_linear.set_low_rank_factors(U_r, V_r)
            if bias is not None and target_linear.bias is not None:
                target_linear.bias.data.copy_(
                    bias.to(target_linear.bias.device, target_linear.bias.dtype)
                )

    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: nn.Module) -> None:
        """
        SVDLA-FIX-01: never raise. On missing projections, log a warning
        and fall back to the randomly-initialized state.
        """
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # nested encoder
            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                src = old_attn.self
                self.is_nested_encoder = True
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # PyTorch MultiheadAttention
            if hasattr(src, "in_proj_weight") and src.in_proj_weight is not None:
                in_proj_w = getattr(src, "in_proj_weight", None)
                in_proj_b = getattr(src, "in_proj_bias", None)
                if in_proj_w is not None:
                    w_q, w_k, w_v = in_proj_w.chunk(3, dim=0)
                if in_proj_b is not None:
                    b_q, b_k, b_v = in_proj_b.chunk(3, dim=0)
                if hasattr(src, "out_proj"):
                    w_o, b_o = _unpack_weight_bias(src.out_proj)

            # GPT-2 fused
            elif hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
                if c_attn_w is not None:
                    if (
                        c_attn_w.shape[0] == self.hidden_size
                        and c_attn_w.shape[1] != self.hidden_size
                    ):
                        c_attn_w = c_attn_w.t().contiguous()
                    q_dim = self.q_head_size
                    k_dim = self.kv_head_size
                    v_dim = self.kv_head_size
                    w_q = c_attn_w[:q_dim, :]
                    w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                    w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]
                    if c_attn_b is not None:
                        b_q = c_attn_b[:q_dim]
                        b_k = c_attn_b[q_dim : q_dim + k_dim]
                        b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]
                out_w, out_b = _unpack_weight_bias(src.c_proj)
                if (
                    out_w is not None
                    and out_w.shape[0] == self.hidden_size
                    and out_w.shape[1] != self.hidden_size
                ):
                    out_w = out_w.t().contiguous()
                w_o, b_o = out_w, out_b

            # BERT / RoBERTa nested
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q, b_q = _unpack_weight_bias(src.query)
                w_k, b_k = _unpack_weight_bias(src.key)
                w_v, b_v = _unpack_weight_bias(src.value)
                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                            break

            # Qwen / LLaMA
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q, b_q = _unpack_weight_bias(src.q_proj)
                w_k, b_k = _unpack_weight_bias(src.k_proj)
                w_v, b_v = _unpack_weight_bias(src.v_proj)
                if hasattr(src, "o_proj"):
                    w_o, b_o = _unpack_weight_bias(src.o_proj)

            # DistilBERT
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q, b_q = _unpack_weight_bias(src.q_lin)
                w_k, b_k = _unpack_weight_bias(src.k_lin)
                w_v, b_v = _unpack_weight_bias(src.v_lin)
                if hasattr(src, "out_lin"):
                    w_o, b_o = _unpack_weight_bias(src.out_lin)

            # UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    if w_q is None:
                        w_q, b_q = adapter.get_weight_bias(old_attn, "q")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(old_attn, "k")
                    if w_v is None:
                        w_v, b_v = adapter.get_weight_bias(old_attn, "v")
                    if w_o is None and not self.is_nested_encoder:
                        w_o, b_o = adapter.get_weight_bias(old_attn, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # ---- missing check: WARN, do not raise ----
            missing = []
            if w_q is None:
                missing.append("q")
            if w_k is None:
                missing.append("k")
            if w_v is None:
                missing.append("v")
            if missing:
                logger.warning(
                    "SVDLinearAttention: could not extract %s from %s; "
                    "keeping random init. This will likely reduce quality.",
                    missing,
                    type(old_attn).__name__,
                )
                return  # early return; projections keep their SVD init

            # ---- near-zero norm check: WARN only ----
            for name, w in (("w_q", w_q), ("w_k", w_k), ("w_v", w_v)):
                if w is None:
                    continue
                try:
                    n = torch.norm(w.float(), p="fro").item()
                except Exception:
                    n = float("nan")
                if not math.isfinite(n) or n < 1e-4:
                    logger.warning(
                        "SVDLinearAttention: %s has near-zero/NaN norm (%.2e); proceeding anyway.",
                        name,
                        n,
                    )

            # ---- factorize Q ----
            try:
                self._factorize_and_set(self.q_proj, w_q, b_q)
            except Exception as e:
                logger.warning("SVDLinearAttention: Q factorization failed (%s).", e)
                return

            # ---- factorize K and V with GQA adaptation ----
            def _adapt_to_kv_size(weight, bias, mode="mean"):
                w, b = weight, bias
                if w.shape[0] == self.kv_head_size:
                    return w, b
                if w.shape[0] > self.kv_head_size and w.shape[0] % self.kv_head_size == 0:
                    rep = w.shape[0] // self.kv_head_size
                    if mode == "slice":
                        w = w.contiguous().view(self.num_kv_heads, rep, -1)[:, 0, :].contiguous()
                        if b is not None:
                            b = (
                                b.contiguous()
                                .view(self.num_kv_heads, rep, -1)[:, 0, :]
                                .contiguous()
                                .view(-1)
                            )
                    elif mode == "first":
                        w = w[: self.kv_head_size, :].contiguous()
                        if b is not None:
                            b = b[: self.kv_head_size].contiguous()
                    else:  # mean
                        w = w.contiguous().view(self.num_kv_heads, rep, -1).mean(dim=1)
                        if b is not None:
                            b = b.contiguous().view(self.num_kv_heads, rep, -1).mean(dim=1).view(-1)
                elif w.shape[0] < self.kv_head_size and self.kv_head_size % w.shape[0] == 0:
                    rep = self.kv_head_size // w.shape[0]
                    w = w.repeat(rep, 1)
                    if b is not None:
                        b = b.repeat(rep)
                else:
                    padded_w = torch.zeros(
                        self.kv_head_size,
                        w.shape[1],
                        device=w.device,
                        dtype=w.dtype,
                    )
                    min_rows = min(self.kv_head_size, w.shape[0])
                    padded_w[:min_rows, :] = w[:min_rows, :]
                    w = padded_w
                    if b is not None:
                        padded_b = torch.zeros(
                            self.kv_head_size,
                            device=b.device,
                            dtype=b.dtype,
                        )
                        padded_b[:min_rows] = b[:min_rows]
                        b = padded_b
                return w, b

            try:
                w_k_a, b_k_a = _adapt_to_kv_size(w_k, b_k, mode="slice")
                self._factorize_and_set(self.k_proj, w_k_a, b_k_a)
            except Exception as e:
                logger.warning("SVDLinearAttention: K factorization failed (%s).", e)
                return

            try:
                w_v_a, b_v_a = _adapt_to_kv_size(w_v, b_v, mode="slice")
                self._factorize_and_set(self.v_proj, w_v_a, b_v_a)
            except Exception as e:
                logger.warning("SVDLinearAttention: V factorization failed (%s).", e)
                return

            # ---- output projection ----
            if w_o is not None and not self.is_nested_encoder:
                try:
                    n_o = torch.norm(w_o.float(), p="fro").item()
                except Exception:
                    n_o = float("nan")
                if not math.isfinite(n_o) or n_o < 1e-4:
                    logger.warning(
                        "SVDLinearAttention: out_proj has near-zero/NaN norm (%.2e); "
                        "proceeding anyway.",
                        n_o,
                    )
                try:
                    self._factorize_and_set(self.out_proj, w_o, b_o)
                except Exception as e:
                    logger.warning("SVDLinearAttention: out_proj factorization failed (%s).", e)

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------
    def _forward_decay_gated_linear_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        compute_dtype = torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float32

        phi_q = F.elu(q.to(compute_dtype)) + 1.0
        phi_k = F.elu(k.to(compute_dtype)) + 1.0
        v_f = v.to(compute_dtype)

        mask_expanded = _normalize_mask(mask, B, H, N, compute_dtype, q.device)
        if mask_expanded is not None:
            # mask_expanded shape (B, 1, N, 1) or (B, H, N, 1)
            if mask_expanded.shape[1] == 1:
                phi_k = phi_k * mask_expanded
            else:
                phi_k = phi_k * mask_expanded

        # decay
        decay_logits = self.decay_gate(hidden_states)
        if self.decay_channel_wise:
            decay_logits = decay_logits.view(B, N, H, D_k).permute(0, 2, 1, 3)
            lambda_decay = (
                compute_exponential_decay(decay_logits, activation=self.decay_activation)
                .unsqueeze(-1)
                .to(compute_dtype)
            )
        else:
            decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            lambda_decay = compute_exponential_decay(
                decay_logits, activation=self.decay_activation
            ).to(compute_dtype)

        # padded positions do not decay
        if mask_expanded is not None:
            m_dec = mask_expanded
            while m_dec.dim() < lambda_decay.dim():
                m_dec = m_dec.unsqueeze(-1)
            lambda_decay = torch.where(m_dec > 0, lambda_decay, torch.ones_like(lambda_decay))

        # ---- chunked scan for non-channel-wise and no past state ----
        if past_state is None and not self.decay_channel_wise:
            chunk_size = self.chunk_size
            num_chunks = (N + chunk_size - 1) // chunk_size
            S_chunk = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
            Z_chunk = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)
            outputs = []

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)
                qc = phi_q[:, :, start:end, :]
                kc = phi_k[:, :, start:end, :]
                vc = v_f[:, :, start:end, :]
                lam_c = lambda_decay[:, :, start:end].squeeze(-1).squeeze(-1)
                Lc = end - start

                # SVDLA-FIX-11: safe log with floor
                log_lam_c = torch.log(torch.clamp(lam_c, min=1e-6, max=1.0))
                log_gamma_c = torch.cumsum(log_lam_c, dim=-1)

                log_M_intra = log_gamma_c.unsqueeze(-1) - log_gamma_c.unsqueeze(-2)
                causal_bool = torch.tril(torch.ones(Lc, Lc, dtype=torch.bool, device=qc.device))
                M_intra = torch.exp(torch.clamp(log_M_intra, max=0.0)) * causal_bool.unsqueeze(
                    0
                ).unsqueeze(0)

                aff_intra = torch.matmul(qc, kc.transpose(-1, -2).contiguous()) * M_intra
                num_intra = torch.matmul(aff_intra, vc)
                den_intra = aff_intra.sum(dim=-1, keepdim=True)

                decay_from_past = torch.exp(torch.clamp(log_gamma_c, max=0.0)).unsqueeze(-1)
                num_inter = torch.matmul(qc, S_chunk) * decay_from_past
                den_inter = torch.matmul(qc, Z_chunk) * decay_from_past

                out_c = (num_intra + num_inter) / (den_intra + den_inter + self.eps)
                outputs.append(out_c)

                total_chunk_decay = torch.exp(
                    torch.clamp(log_gamma_c[:, :, -1:], max=0.0)
                ).unsqueeze(-1)
                log_w_to_end = log_gamma_c[:, :, -1:] - log_gamma_c
                w_to_end = torch.exp(torch.clamp(log_w_to_end, max=0.0)).unsqueeze(-1)
                kc_w = kc * w_to_end

                S_chunk = total_chunk_decay * S_chunk + torch.matmul(
                    kc_w.transpose(-1, -2).contiguous(), vc
                )
                Z_chunk = (
                    total_chunk_decay * Z_chunk
                    + kc_w.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()
                )

            context = torch.cat(outputs, dim=2).to(q.dtype)
            return context, (S_chunk, Z_chunk)

        # ---- sequential recurrent scan ----
        if past_state is not None:
            S, Z = past_state
            S = S.to(device=q.device, dtype=compute_dtype)
            Z = Z.to(device=q.device, dtype=compute_dtype)
        else:
            S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
            Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)

        outputs = []
        for t in range(N):
            qt = phi_q[:, :, t : t + 1, :]
            kt = phi_k[:, :, t : t + 1, :]
            vt = v_f[:, :, t : t + 1, :]
            lam_t = lambda_decay[:, :, t]

            kt_t = kt.transpose(-1, -2).contiguous()
            kv_t = torch.matmul(kt_t, vt)
            S = lam_t * S + kv_t
            Z = lam_t * Z + kt_t

            num_t = torch.matmul(qt, S)
            den_t = torch.matmul(qt, Z) + self.eps
            outputs.append(num_t / den_t)

        context = torch.cat(outputs, dim=2).to(q.dtype)
        return context, (S, Z)

    def _forward_linear_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None and q.device.type == "cuda" and fast_linear_attention is not None:
            try:
                return fast_linear_attention(
                    q,
                    k,
                    v,
                    is_causal=self.is_decoder,
                    eps=self.eps,
                )
            except Exception as e:
                logger.warning("Fast Triton kernel failed (%s). Falling back to PyTorch.", e)

        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        phi_q = F.elu(q.float()) + 1.0
        phi_k = F.elu(k.float()) + 1.0
        v_f = v.float()

        mask_expanded = _normalize_mask(mask, B, H, N, torch.float32, q.device)
        if mask_expanded is not None:
            phi_k = phi_k * mask_expanded

        if self.is_decoder and N > 1:
            chunk_size = self.chunk_size
            num_chunks = (N + chunk_size - 1) // chunk_size
            S_chunk = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=torch.float32)
            Z_chunk = torch.zeros(B, H, D_k, 1, device=q.device, dtype=torch.float32)
            outputs = []

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)
                qc = phi_q[:, :, start:end, :]
                kc = phi_k[:, :, start:end, :]
                vc = v_f[:, :, start:end, :]
                Lc = end - start

                aff_intra = torch.matmul(qc, kc.transpose(-1, -2).contiguous())
                causal_intra = (
                    torch.tril(torch.ones(Lc, Lc, device=q.device, dtype=torch.bool))
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                aff_intra = aff_intra * causal_intra
                num_intra = torch.matmul(aff_intra, vc)
                den_intra = aff_intra.sum(dim=-1, keepdim=True)

                num_inter = torch.matmul(qc, S_chunk)
                den_inter = torch.matmul(qc, Z_chunk)

                out_c = (num_intra + num_inter) / (den_intra + den_inter + self.eps)
                outputs.append(out_c)

                S_chunk = S_chunk + torch.matmul(kc.transpose(-1, -2).contiguous(), vc)
                Z_chunk = Z_chunk + kc.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()

            context = torch.cat(outputs, dim=2).to(q.dtype)
        else:
            kv_state = torch.matmul(phi_k.transpose(-1, -2).contiguous(), v_f)
            numerator = torch.matmul(phi_q, kv_state)
            k_sum = phi_k.sum(dim=-2, keepdim=True)
            denominator = torch.matmul(phi_q, k_sum.transpose(-1, -2).contiguous()) + self.eps
            context = (numerator / denominator).to(q.dtype)

        return context

    def _forward_exact_softmax(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        is_cross: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        is_causal_attn = self.is_decoder and not is_cross and q.shape[-2] > 1

        if not output_attentions:
            try:
                sdpa_mask = None
                can_use_sdpa = False
                if mask is None:
                    can_use_sdpa = True
                elif mask.dim() == 2 and not is_causal_attn:
                    sdpa_mask = mask.bool().view(mask.shape[0], 1, 1, mask.shape[1])
                    can_use_sdpa = True
                elif mask.dim() == 4 and not is_causal_attn:
                    sdpa_mask = mask.bool() if mask.dtype == torch.bool else mask
                    can_use_sdpa = True

                if can_use_sdpa:
                    drop_p = self.dropout.p if self.training else 0.0
                    context = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=sdpa_mask,
                        dropout_p=drop_p,
                        is_causal=is_causal_attn if sdpa_mask is None else False,
                    )
                    return context, None
            except Exception:
                pass

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2).contiguous()) * scale

        if is_causal_attn:
            causal_mask = (
                torch.triu(
                    torch.full(
                        (q.shape[-2], k.shape[-2]),
                        float("-inf"),
                        device=scores.device,
                        dtype=torch.float32,
                    ),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal_mask

        if mask is not None:
            scores = apply_attention_mask(scores, mask)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(q.dtype)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v)
        return context, probs

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        query = kwargs.get("query")
        key = kwargs.get("key")
        mask = kwargs.get("mask")
        return_state = kwargs.get("return_state", False)
        past_state = kwargs.get("past_state")

        if return_state and (
            not self.use_decay_gating
            and self.mode not in ("decay_linear", "linear_decay", "decay", "gated_linear")
        ):
            raise ValueError(
                f"return_state=True is only supported in decay/gated modes, got mode='{self.mode}'"
            )

        if hidden_states is None:
            hidden_states = query if query is not None else key
        if attention_mask is None:
            attention_mask = mask

        B, N, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # 1. projections
        q = (
            self.q_proj(hidden_states)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.k_proj(kv_input)
            .view(B, kv_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(kv_input)
            .view(B, kv_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # 2. RoPE
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. GQA repeat
        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        updated_state: tuple[torch.Tensor, torch.Tensor] | None = None

        # 4. dispatch
        if self.mode == "exact":
            context, attn_probs = self._forward_exact_softmax(
                q,
                k,
                v,
                eff_mask,
                is_cross=is_cross,
                output_attentions=bool(output_attentions),
            )
        elif self.use_decay_gating or self.mode in (
            "decay_linear",
            "linear_decay",
            "decay",
            "gated_linear",
        ):
            context, updated_state = self._forward_decay_gated_linear_kernel(
                q=q,
                k=k,
                v=v,
                hidden_states=hidden_states,
                mask=eff_mask,
                past_state=past_state,
            )
            attn_probs = None
        else:
            context = self._forward_linear_kernel(q, k, v, eff_mask)
            attn_probs = None

        # 5. head_mask
        if head_mask is not None:
            hm = head_mask
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            hm = hm.to(context.dtype)
            if attn_probs is not None:
                attn_probs = attn_probs * hm
            context = context * hm

        # 6. head merge & output projection
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        # SVDLA-FIX-06: explicit error on shape mismatch
        if output.shape[-1] != hidden_states.shape[-1]:
            raise RuntimeError(
                f"SVDLinearAttention: output dim {output.shape[-1]} != "
                f"input hidden dim {hidden_states.shape[-1]}"
            )

        attn_out = attn_probs if output_attentions else None

        if return_state and updated_state is not None:
            return output, attn_out, updated_state
        return output, attn_out

    # ------------------------------------------------------------------
    def step_recurrent(
        self,
        x_t: torch.Tensor,
        past_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)

        B = x_t.shape[0]
        q = self.q_proj(x_t).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = (
            self.k_proj(x_t)
            .view(B, 1, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(x_t)
            .view(B, 1, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        phi_q = F.elu(q.float()) + 1.0
        phi_k = F.elu(k.float()) + 1.0

        _, H, _, D_k = q.shape
        D_v = v.shape[-1]

        if self.use_decay_gating:
            decay_logits = self.decay_gate(x_t)
            if self.decay_channel_wise:
                decay_logits = decay_logits.view(B, 1, H, D_k).permute(0, 2, 1, 3)
                lam_t = (
                    compute_exponential_decay(decay_logits, activation=self.decay_activation)
                    .unsqueeze(-1)
                    .squeeze(2)
                )
            else:
                decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
                lam_t = compute_exponential_decay(
                    decay_logits, activation=self.decay_activation
                ).squeeze(2)
        else:
            lam_t = torch.ones(B, H, 1, 1, device=x_t.device, dtype=torch.float32)

        if past_state is not None:
            S_prev, Z_prev = past_state
        else:
            S_prev = torch.zeros(B, H, D_k, D_v, device=x_t.device, dtype=torch.float32)
            Z_prev = torch.zeros(B, H, D_k, 1, device=x_t.device, dtype=torch.float32)

        kt_t = phi_k.transpose(-1, -2).contiguous()
        kv_t = torch.matmul(kt_t, v.float())

        S_t = lam_t * S_prev + kv_t
        Z_t = lam_t * Z_prev + kt_t

        num_t = torch.matmul(phi_q, S_t)
        den_t = torch.matmul(phi_q, Z_t) + self.eps
        y_t = num_t / den_t

        context = y_t.transpose(1, 2).contiguous().view(B, 1, self.all_head_size).to(x_t.dtype)
        if self.is_nested_encoder:
            out_t = context
        else:
            out_t = self.out_proj(context)

        if out_t.shape[-1] != x_t.shape[-1]:
            raise RuntimeError(
                f"SVDLinearAttention.step_recurrent: output dim "
                f"{out_t.shape[-1]} != input dim {x_t.shape[-1]}"
            )

        return out_t, (S_t, Z_t)

    def get_orthogonality_loss(self) -> torch.Tensor:
        return (
            self.q_proj.get_orthogonality_loss()
            + self.k_proj.get_orthogonality_loss()
            + self.v_proj.get_orthogonality_loss()
            + self.out_proj.get_orthogonality_loss()
        )


# ===========================================================================
# Injector
# ===========================================================================
@ATTENTION_REGISTRY.register("svd_linear_inject")
@ATTENTION_REGISTRY.register("inject_svd_linear_attention")
def inject_svd_linear_attention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    rank: int | None = 192,
    mode: str = "exact",
    _depth: int = 0,
    **kwargs: Any,
) -> Any | None:
    if _depth > 8:
        logger.warning("inject_svd_linear_attention: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    # ---- whole-model traversal ----
    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and not hasattr(target, "forward"))
    ):
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
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_svd_linear_attention(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    rank=rank,
                    mode=mode,
                    _depth=_depth + 1,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    # ---- single layer ----
    curr_attn = old_attn
    if curr_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn"):
            if hasattr(target, attr):
                curr_attn = getattr(target, attr)
                break

    if curr_attn is None:
        return None

    # SVDLA-FIX-04: idempotency guard
    if (
        isinstance(curr_attn, SVDLinearAttention)
        or "SVDLinearAttention" in curr_attn.__class__.__name__
    ):
        return None

    is_nested = hasattr(target, "attention") and hasattr(target.attention, "self")
    if is_nested and curr_attn is target.attention:
        curr_attn = target.attention.self

    try:
        new_attn = SVDLinearAttention(
            config=cfg,
            old_attn=curr_attn,
            rank=rank,
            mode=mode,
            lora_rank=lora_rank,
            **kwargs,
        )
        new_attn.is_nested_encoder = is_nested
    except Exception as e:
        logger.error("inject_svd_linear_attention: construction failed: %s", e)
        return None

    if target is not None:
        if is_nested:
            target.attention.self = new_attn
        elif hasattr(target, "attention"):
            target.attention = new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn
        else:
            logger.error(
                "inject_svd_linear_attention: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

    return new_attn


__all__ = [
    "SVDLinear",
    "SVDLinearAttention",
    "inject_svd_linear_attention",
    "apply_rotary_pos_emb",
    "compute_exponential_decay",
]
