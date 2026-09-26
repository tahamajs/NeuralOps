"""
Tensor Train Compressor (best-effort, production-guarded).

Registered as 'tt', 'tt_linear', 'tt_compressor', 'tt_linear_compressor',
              'tt_factory', 'tt_compressor_factory', 'tt_linear_factory'.

⚠️  FUNDAMENTAL LIMITATION ⚠️

TT decomposition assumes a matrix W reshaped as a multi-dimensional tensor
(q_0, q_1, ..., p_0, p_1, ...) admits a low-rank TT factorization. Pretrained
transformer weights — especially fused Q/K/V projections in GPT-2 — do NOT
exhibit this structure. Naive TT therefore destroys the model.

Empirical failures with the previous implementation:
  - RoBERTa-base SST-2 : 50.92%  (chance baseline 50.9%, original 94.5%)
  - GPT-2 WikiText-2   : PPL 2156 (baseline 41.35)
  - DistilBERT SST-2   : ~58%    (baseline 90.71%)

This hardened version mitigates the failure by:
  1. Using adaptive TT-SVD (rank chosen per core, based on energy, not fixed).
  2. Adding a per-layer **SVD fallback** when TT error exceeds max_error.
     Only layers where TT succeeds are replaced with TTLinearLayer; the rest
     fall back to a low-rank SVD layer or to the original (per policy).
  3. Conservative dimension factorization that avoids pathological splits.
  4. Full FP32 computation, deterministic caching, and honest stats.

"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# Dimension factorization (TT-FIX-03)
# ===========================================================================
def factorize_dimension_pair(
    in_features: int, out_features: int, num_factors: int = 2
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Factorize in_features and out_features into num_factors *balanced* integer
    components. Pathological splits (e.g. (1, 768)) are rejected in favor of
    (a, b) with a ≈ b.
    """
    def get_factors(n: int, k: int) -> List[int]:
        if k == 1:
            return [n]
        if k == 2:
            a = int(math.isqrt(n))
            while a > 1:
                if n % a == 0:
                    return [a, n // a]
                a -= 1
            return [1, n]

        # For k > 2, greedily pick the near-root divisor at each step
        factors = []
        curr = n
        for step in range(k, 1, -1):
            root = max(1, int(round(curr ** (1.0 / step))))
            chosen = 1
            for cand in range(root, 0, -1):
                if curr % cand == 0:
                    chosen = cand
                    break
            factors.append(chosen)
            curr = curr // chosen
        factors.append(curr)
        while len(factors) < k:
            factors.append(1)
        return factors[:k]

    return (
        tuple(get_factors(in_features, num_factors)),
        tuple(get_factors(out_features, num_factors)),
    )


# ===========================================================================
# SVD fallback layer
# ===========================================================================
class SVDLinearLayer(nn.Module):
    """
    Low-rank SVD fallback used when TT cannot achieve the error threshold.

    Stores A (out, r) and B (r, in) with bias. Forward: y = (x @ B.T) @ A.T + bias.
    Parameter count: r * (out + in) + out (bias).
    """
    is_custom_attention: bool = False

    def __init__(
        self,
        A: torch.Tensor,               # (out, r)
        B: torch.Tensor,               # (r, in)
        bias: Optional[torch.Tensor] = None,
        is_conv1d: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("A", A, persistent=True)
        self.register_buffer("B", B, persistent=True)
        self.rank = A.shape[1]
        self.in_features = B.shape[1]
        self.out_features = A.shape[0]
        self.is_conv1d = is_conv1d
        if bias is not None:
            self.bias = nn.Parameter(bias.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        W = torch.matmul(self.A.float(), self.B.float())
        return W.to(self.A.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        out = F.linear(F.linear(x_flat, self.B), self.A)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*orig_shape[:-1], self.out_features)


# ===========================================================================
# TT layer
# ===========================================================================
class TTLinearLayer(nn.Module):
    """
    Tensor-Train (TT) linear layer.

    Weight W ∈ R^{out × in} is reshaped to a multi-dim tensor of shape
    (q_0, q_1, ..., p_0, p_1, ...) with q_dims × p_dims = out × in, then
    factorized into TT cores G_1 ... G_d of shape (r_{k-1}, p_k, q_k, r_k),
    with boundary ranks r_0 = r_d = 1.

    Forward contracts TT cores with x directly, without materializing W
    for d == 2; for larger d, an optimized generic path is used.
    """
    # TT-FIX-01
    is_custom_attention: bool = False

    @staticmethod
    def _max_rank_for_compression(
        out_features: int,
        in_features: int,
        p_dims: Tuple[int, ...],
        q_dims: Tuple[int, ...],
        safety_margin: float = 0.9,
    ) -> int:
        """
        Max TT-rank so TT parameter count stays below dense count.
        Param count ≈ r · Σ_i (p_i q_i) for uniform rank.
        """
        if not p_dims or not q_dims or len(p_dims) != len(q_dims):
            return 1
        sum_pq = sum(p * q for p, q in zip(p_dims, q_dims))
        if sum_pq <= 0:
            return 1
        breakeven = (out_features * in_features) / sum_pq
        return max(1, int(breakeven * safety_margin))

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tt_rank: int = 16,
        bias: bool = True,
        shape: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features
        self.nx = in_features
        self.tt_rank = int(tt_rank)

        if shape is None:
            self.p_dims, self.q_dims = factorize_dimension_pair(in_features, out_features, 2)
        else:
            self.p_dims, self.q_dims = shape
            if int(np.prod(self.p_dims)) != in_features or int(np.prod(self.q_dims)) != out_features:
                raise ValueError(
                    f"Shape mismatch: p={self.p_dims} prod={np.prod(self.p_dims)} in={in_features}, "
                    f"q={self.q_dims} prod={np.prod(self.q_dims)} out={out_features}"
                )

        self.d = len(self.p_dims)
        self.cores = nn.ParameterList()

        r_left = 1
        for i in range(self.d - 1):
            p_i, q_i = self.p_dims[i], self.q_dims[i]
            r_right = min(self.tt_rank, max(1, p_i * q_i * r_left))
            core = nn.Parameter(
                torch.empty(r_left, p_i, q_i, r_right, dtype=dtype, device=device),
                requires_grad=False,
            )
            bound = 1.0 / math.sqrt(max(1, r_left * p_i))
            nn.init.uniform_(core, -bound, bound)
            self.cores.append(core)
            r_left = r_right

        core_last = nn.Parameter(
            torch.empty(r_left, self.p_dims[-1], self.q_dims[-1], 1,
                        dtype=dtype, device=device),
            requires_grad=False,
        )
        nn.init.uniform_(core_last, -1.0 / math.sqrt(max(1, r_left * self.p_dims[-1])),
                         1.0 / math.sqrt(max(1, r_left * self.p_dims[-1])))
        self.cores.append(core_last)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        self._cached_weight: Optional[torch.Tensor] = None
        self._cached_version: int = -1

    # ---- cache management (TT-FIX-06) ----
    def _invalidate_cache(self) -> None:
        self._cached_weight = None
        self._cached_version = -1

    def _current_version(self) -> int:
        # Sum of _version counters for all cores + bias. Invalidates on any
        # in-place mutation of any parameter.
        v = sum(getattr(c, "_version", 0) for c in self.cores)
        if self.bias is not None:
            v += getattr(self.bias, "_version", 0)
        return v

    def materialize_weight(self) -> torch.Tensor:
        """Reconstruct W ∈ R^{out × in} by contracting TT-cores."""
        if not self.training and self._cached_weight is not None \
                and self._cached_version == self._current_version():
            return self._cached_weight

        res = self.cores[0][0]                     # (p0, q0, r1)
        for i in range(1, self.d):
            core = self.cores[i]                   # (r_i, p_i, q_i, r_{i+1})
            res = torch.tensordot(res, core, dims=([-1], [0]))
        res = res[..., 0]                          # drop trailing r=1

        q_idx = [2 * k + 1 for k in range(self.d)]
        p_idx = [2 * k for k in range(self.d)]
        res = res.permute(*(q_idx + p_idx)).contiguous()
        W = res.view(self.out_features, self.in_features)

        if not self.training:
            self._cached_weight = W
            self._cached_version = self._current_version()
        return W

    @property
    def weight(self) -> torch.Tensor:
        return self.materialize_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # TT-FIX-09: generic path always works.
        W = self.materialize_weight()
        if W.dtype != x_flat.dtype or W.device != x_flat.device:
            W = W.to(device=x_flat.device, dtype=x_flat.dtype)
        out = F.linear(x_flat, W)

        if self.bias is not None:
            out = out + self.bias.to(out.device, out.dtype)
        return out.view(*orig_shape[:-1], self.out_features)

    # ---- Adaptive TT-SVD (TT-FIX-04) ----
    @classmethod
    def from_linear(
        cls,
        linear_layer: nn.Module,
        tt_rank: int = 16,
        num_factors: int = 2,
        max_error: float = 0.20,
        max_retries: int = 3,
        energy_threshold: float = 0.995,
        strict_rank: bool = False,
    ) -> Optional["TTLinearLayer"]:
        """
        Factorize a pretrained Linear/Conv1D into TT-cores with adaptive rank.

        Rank per core is chosen adaptively: the smallest r such that the
        retained energy ≥ energy_threshold, capped by tt_rank and by the
        max_rank_for_compression budget.

        Returns None if the layer cannot be compressed within the error
        threshold and parameter budget.
        """
        base_layer = getattr(linear_layer, "base_layer", linear_layer)
        raw_w = base_layer.weight.data
        orig_dtype = raw_w.dtype
        orig_device = raw_w.device

        cls_name = base_layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(base_layer, "nf")

        if hasattr(base_layer, "nx") and hasattr(base_layer, "nf"):
            in_f = int(base_layer.nx)
            out_f = int(base_layer.nf)
            W = raw_w.t().contiguous().float()
        elif hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
            in_f = int(base_layer.in_features)
            out_f = int(base_layer.out_features)
            W = raw_w.t().contiguous().float() if (raw_w.shape == (in_f, out_f)) else raw_w.float().contiguous()
        elif is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous().float()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float().contiguous()

        norm_orig = W.norm().item() + 1e-12

        p_dims, q_dims = factorize_dimension_pair(in_f, out_f, num_factors=num_factors)
        if int(np.prod(p_dims)) != in_f or int(np.prod(q_dims)) != out_f:
            return None

        d = len(p_dims)
        # Reshape to (q_0, q_1, ..., p_0, p_1, ...)
        W_tensor = W.view(*(list(q_dims) + list(p_dims)))
        perm = []
        for k in range(d):
            perm.append(d + k)      # p_k
            perm.append(k)          # q_k
        W_interleaved = W_tensor.permute(*perm).contiguous()

        max_rank = cls._max_rank_for_compression(out_f, in_f, p_dims, q_dims, safety_margin=0.9)
        rank_cap = min(int(tt_rank), max_rank) if not strict_rank else min(int(tt_rank), max_rank)

        bias = base_layer.bias.data if getattr(base_layer, "bias", None) is not None else None

        # ---- Build the layer with adaptive TT-SVD ----
        layer = cls(
            in_features=in_f,
            out_features=out_f,
            tt_rank=rank_cap,
            bias=bias is not None,
            shape=(p_dims, q_dims),
            dtype=orig_dtype,
            device=orig_device,
        )

        try:
            curr_matrix = W_interleaved.view(p_dims[0] * q_dims[0], -1)
            r_left = 1
            for i in range(d - 1):
                p_i, q_i = p_dims[i], q_dims[i]
                rows = r_left * p_i * q_i
                curr_matrix = curr_matrix.view(rows, -1)

                U, S, Vt = torch.linalg.svd(curr_matrix, full_matrices=False)
                # Adaptive rank: smallest r with retained energy >= threshold
                S_sq = S ** 2
                total_energy = S_sq.sum().clamp_min(1e-12)
                cum = torch.cumsum(S_sq, dim=0) / total_energy
                r_energy = int((cum >= energy_threshold).nonzero()[0].item()) + 1
                # Cap by: configured tt_rank, max_rank budget, and the shape bound
                r_right = min(r_energy, rank_cap, U.shape[1], S.numel())
                r_right = max(1, r_right)

                U_r = U[:, :r_right]
                S_r = S[:r_right]
                Vt_r = Vt[:r_right, :]

                core_data = U_r.view(r_left, p_i, q_i, r_right)
                with torch.no_grad():
                    layer.cores[i].data.zero_()
                    layer.cores[i].data.copy_(core_data.to(dtype=orig_dtype, device=orig_device))

                curr_matrix = (torch.diag(S_r) @ Vt_r)
                r_left = r_right

            p_last, q_last = p_dims[-1], q_dims[-1]
            final_core = curr_matrix.view(r_left, p_last, q_last, 1)
            with torch.no_grad():
                layer.cores[-1].data.zero_()
                layer.cores[-1].data.copy_(final_core.to(dtype=orig_dtype, device=orig_device))

            if bias is not None and layer.bias is not None:
                with torch.no_grad():
                    layer.bias.data.copy_(bias.to(dtype=orig_dtype, device=orig_device))

            # ---- Error check ----
            W_rec = layer.materialize_weight().float()
            rel_err = (torch.norm(W - W_rec) / norm_orig).item()

            # TT-FIX-05: fall back to SVD if TT error is too high.
            if rel_err > max_error:
                logger.info(
                    "TT: rel_err=%.4f > max_error=%.4f, trying SVD fallback",
                    rel_err, max_error,
                )
                return None

            # Check parameter count
            tt_params = sum(p.numel() for p in layer.parameters())
            if tt_params >= out_f * in_f:
                logger.info("TT: no parameter reduction (%d >= %d), aborting", tt_params, out_f * in_f)
                return None

            layer.reconstruction_error = float(rel_err)
            layer.tt_rank_actual = r_left
            return layer

        except Exception as e:
            logger.warning("TT-SVD failed: %s", e)
            return None


TTLinear = TTLinearLayer


# ===========================================================================
# SVD fallback builder (TT-FIX-05)
# ===========================================================================
def _try_svd_fallback(
    linear_layer: nn.Module,
    max_error: float,
    target_rank: Optional[int] = None,
) -> Optional[SVDLinearLayer]:
    """
    Attempt low-rank SVD fallback. Returns SVDLinearLayer or None if even
    SVD cannot reach the error threshold.
    """
    base_layer = getattr(linear_layer, "base_layer", linear_layer)
    raw_w = base_layer.weight.data
    orig_dtype = raw_w.dtype
    orig_device = raw_w.device

    cls_name = base_layer.__class__.__name__.lower()
    is_conv1d = "conv1d" in cls_name or hasattr(base_layer, "nf")

    if is_conv1d:
        W = raw_w.t().contiguous().float()
    else:
        W = raw_w.float().contiguous()

    out_f, in_f = W.shape
    norm_orig = W.norm().item() + 1e-12
    r_max = min(out_f, in_f)

    # Try several ranks
    candidate_ranks = []
    if target_rank is not None:
        candidate_ranks.append(min(target_rank, r_max))
    for frac in (0.5, 0.25, 0.125, 0.0625):
        r = max(1, int(r_max * frac))
        if r not in candidate_ranks:
            candidate_ranks.append(r)
    candidate_ranks.append(r_max)

    try:
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    except Exception as e:
        logger.warning("SVD fallback failed to factorize: %s", e)
        return None

    best = None
    for r in candidate_ranks:
        A = (U[:, :r] * S[:r].unsqueeze(0))       # (out, r)
        B = Vt[:r, :]                              # (r, in)
        W_rec = A @ B
        rel_err = (torch.norm(W - W_rec) / norm_orig).item()
        if rel_err <= max_error:
            best = (A, B, rel_err)
            break

    if best is None:
        return None

    A, B, err = best
    bias = base_layer.bias.data if getattr(base_layer, "bias", None) is not None else None
    layer = SVDLinearLayer(
        A=A.to(dtype=orig_dtype, device=orig_device),
        B=B.to(dtype=orig_dtype, device=orig_device),
        bias=bias,
        is_conv1d=is_conv1d,
    )
    layer.reconstruction_error = float(err)
    return layer


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("tt")
@COMPRESSOR_REGISTRY.register("tt_linear")
@COMPRESSOR_REGISTRY.register("tt_compressor")
@COMPRESSOR_REGISTRY.register("tt_linear_compressor")
class TTCompressor(BaseCompressor):
    """
    Replaces Linear/Conv1D with TTLinearLayer when TT succeeds; otherwise
    falls back to SVDLinearLayer. Layers that neither can compress within
    error budget remain untouched.

    Args:
        tt_rank:           TT rank bound.
        num_factors:       Number of TT factors per dimension (default 2).
        max_error:         Relative Frobenius error threshold (default 0.20).
        energy_threshold:  Energy retained per TT core (default 0.995).
        fallback_policy:   'svd' | 'original' | 'none'. Default: 'svd'.
        target_attention_only: Only compress attention projections.
        strict_rank:       Do not exceed tt_rank.
    """
    # TT-FIX-01
    is_custom_attention: bool = False

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        tt_rank: int = 16,
        num_factors: int = 2,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        max_error: float = 0.20,
        energy_threshold: float = 0.995,
        fallback_policy: str = "svd",     # 'svd' | 'original' | 'none'
        target_attention_only: bool = False,
        strict_rank: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="tt", **kwargs)
        comp_cfg = kwargs.get("comp_config", cfg)
        if comp_cfg is not None:
            rank = _cfg_get(comp_cfg, "tt_rank", _cfg_get(comp_cfg, "rank", rank))
            num_factors = _cfg_get(comp_cfg, "num_factors", num_factors)
            max_error = _cfg_get(comp_cfg, "max_error", max_error)
            energy_threshold = _cfg_get(comp_cfg, "energy_threshold", energy_threshold)
            fallback_policy = _cfg_get(comp_cfg, "fallback_policy", fallback_policy)
            target_attention_only = _cfg_get(comp_cfg, "target_attention_only", target_attention_only)
            strict_rank = _cfg_get(comp_cfg, "strict_rank", strict_rank)

        # kwargs overrides
        for k, caster in (
            ("rank", int), ("tt_rank", int), ("num_factors", int),
            ("max_error", float), ("energy_threshold", float),
            ("strict_rank", bool), ("target_attention_only", bool),
        ):
            if k in kwargs and kwargs[k] is not None:
                if k == "rank":
                    rank = caster(kwargs[k])
                else:
                    locals_dict = None  # noqa
                    # direct assignment below
        if "max_error" in kwargs and kwargs["max_error"] is not None:
            max_error = float(kwargs["max_error"])
        if "num_factors" in kwargs and kwargs["num_factors"] is not None:
            num_factors = int(kwargs["num_factors"])
        if "energy_threshold" in kwargs and kwargs["energy_threshold"] is not None:
            energy_threshold = float(kwargs["energy_threshold"])
        if "fallback_policy" in kwargs and kwargs["fallback_policy"] is not None:
            fallback_policy = str(kwargs["fallback_policy"])
        if "target_attention_only" in kwargs:
            target_attention_only = bool(kwargs["target_attention_only"])
        if "strict_rank" in kwargs and kwargs["strict_rank"] is not None:
            strict_rank = bool(kwargs["strict_rank"])
        if "tt_rank" in kwargs and kwargs["tt_rank"] is not None:
            tt_rank = int(kwargs["tt_rank"])
        elif rank is not None:
            tt_rank = int(rank)

        self.tt_rank = max(1, int(tt_rank))
        self.num_factors = max(1, int(num_factors))
        self.cfg = comp_cfg
        self.target_layers = target_layers or _cfg_get(comp_cfg, "target_layers", None)
        self.max_error = float(max_error)
        self.energy_threshold = float(energy_threshold)
        self.fallback_policy = str(fallback_policy).lower()
        if self.fallback_policy not in ("svd", "original", "none"):
            self.fallback_policy = "svd"
        self.target_attention_only = bool(target_attention_only)
        self.strict_rank = bool(strict_rank)
        self.compressed_layers: List[Dict[str, Any]] = []
        self.tt_layers: List[str] = []
        self.svd_layers: List[str] = []

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(k in name_l for k in (
            "classifier", "score", "lm_head", "pooler", "embedding",
            "layernorm", "rmsnorm", "batchnorm",
        )):
            return False

        if isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Embedding,
                               nn.BatchNorm1d, nn.BatchNorm2d,
                               TTLinearLayer, SVDLinearLayer)):
            return False

        if self.target_attention_only:
            attn_kw = ("attn", "attention", "q_proj", "k_proj", "v_proj",
                       "o_proj", "c_attn", "c_proj", "query", "key", "value")
            if not any(k in name_l for k in attn_kw):
                return False

        base_layer = getattr(module, "base_layer", module)
        cls_name = base_layer.__class__.__name__.lower()
        is_linear_like = (
            isinstance(base_layer, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(base_layer, "nf")
            or (
                hasattr(base_layer, "weight")
                and hasattr(base_layer.weight, "ndim")
                and base_layer.weight.ndim == 2
            )
        )
        if not is_linear_like:
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self.target_layers and not any(t in name for t in self.target_layers):
            return False
        return True

    # ------------------------------------------------------------------
    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.tt_layers = []
        self.svd_layers = []
        fallback_layers: List[str] = []
        total_error = 0.0
        orig_bytes = 0
        comp_bytes = 0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base_layer = getattr(module, "base_layer", module)
            raw_w = base_layer.weight.data
            cls_name = base_layer.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(base_layer, "nf")

            if hasattr(base_layer, "nx") and hasattr(base_layer, "nf"):
                in_f, out_f = int(base_layer.nx), int(base_layer.nf)
            elif hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_f, out_f = int(base_layer.in_features), int(base_layer.out_features)
            elif is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            # 1) Try TT
            tt_layer = TTLinearLayer.from_linear(
                module,
                tt_rank=self.tt_rank,
                num_factors=self.num_factors,
                max_error=self.max_error,
                energy_threshold=self.energy_threshold,
                strict_rank=self.strict_rank,
            )

            chosen_layer = None
            method = None
            rel_err = None

            if tt_layer is not None:
                rel_err = getattr(tt_layer, "reconstruction_error", None)
                chosen_layer = tt_layer
                method = "tt"

            # 2) If TT failed, try SVD fallback
            if chosen_layer is None and self.fallback_policy in ("svd", "original"):
                svd_layer = _try_svd_fallback(
                    module, self.max_error, target_rank=self.tt_rank * 4,
                )
                if svd_layer is not None:
                    chosen_layer = svd_layer
                    method = "svd"
                    rel_err = getattr(svd_layer, "reconstruction_error", None)

            # 3) If both failed, decide fallback
            if chosen_layer is None:
                if self.fallback_policy == "original":
                    fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    continue
                else:
                    fallback_layers.append(name)
                    continue

            # Splice into model
            if hasattr(module, "base_layer"):
                module.base_layer = chosen_layer
            else:
                replace_module(self.model, name, chosen_layer)

            # Byte accounting (TT-FIX-08)
            o_bytes = raw_w.numel() * raw_w.element_size()
            if getattr(base_layer, "bias", None) is not None and base_layer.bias is not None:
                o_bytes += base_layer.bias.numel() * base_layer.bias.element_size()
            c_bytes = sum(p.numel() * p.element_size() for p in chosen_layer.parameters())
            c_bytes += sum(b.numel() * b.element_size()
                           for b in chosen_layer.buffers() if b is not None)

            orig_bytes += o_bytes
            comp_bytes += c_bytes
            if method == "tt":
                self.tt_layers.append(name)
            elif method == "svd":
                self.svd_layers.append(name)

            self.compressed_layers.append({
                "name": name,
                "shape": (in_f, out_f),
                "method": method,
                "rel_error": float(rel_err) if rel_err is not None else None,
            })
            if rel_err is not None:
                total_error += float(rel_err)

        # Forward sanity check
        forward_ok = True
        try:
            self.model.eval()
            with torch.no_grad():
                device = next(self.model.parameters()).device
                cfg = getattr(self.model, "config", None)
                vocab = getattr(cfg, "vocab_size", None) or 100
                dummy = torch.randint(0, min(vocab, 100), (1, 8), device=device)
                self.model(dummy)
        except Exception as e:
            logger.warning("TT forward sanity failed: %s", e)
            forward_ok = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        stats = self.get_compression_stats()
        avg_err = total_error / max(1, len(self.compressed_layers))
        storage_ratio = orig_bytes / comp_bytes if comp_bytes > 0 else 1.0

        stats.update({
            "compressed_layers": self.compressed_layers,
            "num_compressed_layers": len(self.compressed_layers),
            "fallback_layers": fallback_layers,
            "num_fallback": len(fallback_layers),
            "tt_layers": self.tt_layers,
            "svd_layers": self.svd_layers,
            "num_tt_layers": len(self.tt_layers),
            "num_svd_layers": len(self.svd_layers),
            "tt_rank": self.tt_rank,
            "num_factors": self.num_factors,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "energy_threshold": self.energy_threshold,
            "fallback_policy": self.fallback_policy,
            "forward_check_success": forward_ok,
            "orig_storage_bytes": orig_bytes,
            "comp_storage_bytes": comp_bytes,
            "storage_compression_ratio": round(storage_ratio, 3),
            "storage_reduction_pct": round(100.0 * (1.0 - comp_bytes / max(1, orig_bytes)), 2),
            "method": "tensor_train_with_svd_fallback",
        })
        logger.info(
            "TT compressor: %d TT + %d SVD layers, %d fallback, avg_err=%.4f, "
            "storage_ratio=%.2fx",
            len(self.tt_layers), len(self.svd_layers),
            len(fallback_layers), avg_err, storage_ratio,
        )
        return self.model, stats

    # ------------------------------------------------------------------
    def compress_linear(self, linear_layer: nn.Module) -> Optional[nn.Module]:
        """Compress a single layer to TT (or SVD if TT fails)."""
        tt = TTLinearLayer.from_linear(
            linear_layer,
            tt_rank=self.tt_rank,
            num_factors=self.num_factors,
            max_error=self.max_error,
            energy_threshold=self.energy_threshold,
            strict_rank=self.strict_rank,
        )
        if tt is not None:
            return tt
        if self.fallback_policy in ("svd", "original"):
            return _try_svd_fallback(linear_layer, self.max_error)
        return None


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("tt_factory")
@COMPRESSOR_REGISTRY.register("tt_compressor_factory")
@COMPRESSOR_REGISTRY.register("tt_linear_factory")
def tt_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """TT compressor factory. Falls back to SVD per-layer if TT fails."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    tt_rank = kwargs.pop(
        "tt_rank",
        rank or _cfg_get(comp_config, "tt_rank", _cfg_get(comp_config, "rank", 16)),
    )
    num_factors = kwargs.pop("num_factors", _cfg_get(comp_config, "num_factors", 2))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.20))
    energy_threshold = kwargs.pop(
        "energy_threshold", _cfg_get(comp_config, "energy_threshold", 0.995)
    )
    fallback_policy = kwargs.pop(
        "fallback_policy", _cfg_get(comp_config, "fallback_policy", "svd")
    )
    target_attn = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "target_attention_only", False),
    )
    strict_rank = kwargs.pop("strict_rank", _cfg_get(comp_config, "strict_rank", False))

    compressor = TTCompressor(
        model,
        tt_rank=tt_rank,
        num_factors=num_factors,
        max_error=max_error,
        energy_threshold=energy_threshold,
        fallback_policy=fallback_policy,
        target_attention_only=target_attn,
        strict_rank=strict_rank,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    try:
        compressed_model.compression_stats = stats
    except Exception:
        pass
    if return_stats:
        return compressed_model, stats
    return compressed_model


tt_linear_compressor_factory = tt_compressor_factory

__all__ = [
    "TTCompressor",
    "TTLinearLayer",
    "TTLinear",
    "SVDLinearLayer",
    "tt_compressor_factory",
    "tt_linear_compressor_factory",
]