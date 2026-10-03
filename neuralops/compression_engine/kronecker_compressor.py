"""
Kronecker Product Compressor: replaces nn.Linear and Conv1D layers with KroneckerLinear.
Registered as 'kronecker_class', 'kronecker_compressor_class', and (factory)
'kronecker', 'kronecker_compressor', 'kronecker_factory'.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear


def factorize_dim(dim: int) -> tuple[int, int]:
    """
    Factorize a dimension into two balanced factors (a, b) with a * b = dim,
    minimizing |a - b|. Returns (1, dim) for prime dimensions.
    KL-FIX-13: for dim=1 returns (1, 1) explicitly.
    """
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    if dim == 1:
        return (1, 1)
    a = int(math.isqrt(dim))
    while a > 1:
        if dim % a == 0:
            return (a, dim // a)
        a -= 1
    return (1, dim)


# ===========================================================================
# KroneckerLinear
# ===========================================================================


class KroneckerLinear(nn.Module):
    # KL-FIX-01: this is a linear layer, not an attention layer.
    is_custom_attention: bool = False
    is_custom_linear: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int | None = None,
        m1: int | None = None,
        n1: int | None = None,
        m2: int | None = None,
        n2: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.nf = self.out_features  # Conv1D compatibility
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        # ---- Factor dimensions ----
        if m1 is not None and m2 is not None and n1 is not None and n2 is not None:
            if m1 * m2 != self.out_features or n1 * n2 != self.in_features:
                raise ValueError(
                    f"Factor dims mismatch: {m1}*{m2}!={self.out_features} "
                    f"or {n1}*{n2}!={self.in_features}"
                )
            self.m1, self.m2 = int(m1), int(m2)
            self.n1, self.n2 = int(n1), int(n2)
        else:
            self.n1, self.n2 = factorize_dim(self.in_features)
            self.m1, self.m2 = factorize_dim(self.out_features)

        # Decomposable when both factorizations are non-trivial OR dims are tiny.
        self.can_decompose = (
            (self.n1 * self.n2 == self.in_features)
            and (self.m1 * self.m2 == self.out_features)
            and (self.n1 > 1 or self.in_features <= 3)
            and (self.m1 > 1 or self.out_features <= 3)
        )

        # ---- Parameter allocation ----
        requires_grad = not (self.lora_r > 0)
        self._init_base_parameters(
            bias=bias,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            rank=rank,
        )

        # ---- LoRA parameters ----
        if self.lora_r > 0:
            # KL-FIX-12: keep LoRA in the same dtype as base parameters.
            param_dtype = (
                next(self.parameters()).dtype
                if any(True for _ in self.parameters())
                else torch.float32
            )
            param_device = (
                next(self.parameters()).device
                if any(True for _ in self.parameters())
                else torch.device("cpu")
            )
            self.lora_A = nn.Parameter(
                torch.empty(
                    (self.lora_r, self.in_features),
                    dtype=param_dtype,
                    device=param_device,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    (self.out_features, self.lora_r),
                    dtype=param_dtype,
                    device=param_device,
                )
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    # ------------------------------------------------------------------
    # KL-FIX-09: clean parameter initialization
    # ------------------------------------------------------------------
    def _init_base_parameters(
        self,
        bias: bool,
        dtype: torch.dtype | None,
        device: torch.device | None,
        requires_grad: bool,
        rank: int | None,
    ) -> None:
        if self.can_decompose:
            # Kronecker factors A (m1, n1), B (m2, n2)
            self.A = nn.Parameter(
                torch.empty(self.m1, self.n1, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
            self.B = nn.Parameter(
                torch.empty(self.m2, self.n2, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
            fan_in = max(self.in_features, 1)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.A, -bound, bound)
            nn.init.uniform_(self.B, -bound, bound)

            # Ensure U / V don't exist.
            self.register_parameter("U", None)
            self.register_parameter("V", None)

            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.out_features, dtype=dtype, device=device),
                    requires_grad=requires_grad,
                )
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter("bias", None)
        else:
            effective_rank = rank or max(1, min(self.in_features, self.out_features) // 4)
            self.rank = min(effective_rank, self.in_features, self.out_features)
            self.U = nn.Parameter(
                torch.empty(self.out_features, self.rank, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
            self.V = nn.Parameter(
                torch.empty(self.rank, self.in_features, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)

            # Ensure A / B don't exist.
            self.register_parameter("A", None)
            self.register_parameter("B", None)

            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features, dtype=dtype, device=device),
                    requires_grad=requires_grad,
                )
            else:
                self.register_parameter("bias", None)

    # ------------------------------------------------------------------
    # KL-FIX-02 / KL-FIX-07: cache lifecycle
    # ------------------------------------------------------------------
    def _load_from_state_dict(self, *args, **kwargs):
        # KL-FIX-07: state_dict load invalidates weight cache.
        self._cached_weight = None
        super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, fn, *args, **kwargs):
        # _apply is called for .to(), .cuda(), .float(), .half(), etc.
        # Invalidate cache whenever parameters move.
        self._cached_weight = None
        return super()._apply(fn, *args, **kwargs)

    def _invalidate_cache(self) -> None:
        self._cached_weight = None

    # ------------------------------------------------------------------
    # Weight accessors (KL-FIX-10: no stale caching)
    # ------------------------------------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return self.get_weight()

    def get_weight(self) -> torch.Tensor:
        if self.can_decompose and self.A is not None and self.B is not None:
            return torch.kron(self.A, self.B)
        if self.U is not None and self.V is not None:
            return torch.matmul(self.U, self.V)
        # Fallback: should never happen for a properly initialized layer.
        raise RuntimeError("KroneckerLinear has no valid parameter set.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.can_decompose and self.A is not None and self.B is not None:
            out = self._forward_kron(x)
        else:
            out = self._forward_svd(x)

        # KL-FIX-12: LoRA in native dtype (no explicit fp32 cast).
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            out = out + self.scaling * lora_out.to(out.dtype)
        return out

    def _forward_kron(self, x: torch.Tensor) -> torch.Tensor:
        """(A ⊗ B) x computed without materializing the full weight."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # (K, n1 * n2)
        K = x_flat.shape[0]

        # Step 1: contract with B over n2  ->  (K * n1, m2)
        x_b = torch.matmul(
            x_flat.reshape(K * self.n1, self.n2),
            self.B.t().contiguous(),
        )
        # Step 2: (K, m2, n1) -> (K * m2, n1) @ A^T  ->  (K * m2, m1)
        x_b = (
            x_b.reshape(K, self.n1, self.m2)
            .transpose(1, 2)
            .contiguous()
            .reshape(K * self.m2, self.n1)
        )
        x_a = torch.matmul(x_b, self.A.t().contiguous())
        # Step 3: back to (K, m1 * m2)
        out = (
            x_a.reshape(K, self.m2, self.m1)
            .transpose(1, 2)
            .contiguous()
            .reshape(K, self.out_features)
        )

        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    def _forward_svd(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.V), self.U, self.bias)

    # ------------------------------------------------------------------
    # SVD fitting (KL-FIX-03, KL-FIX-05, KL-FIX-09, KL-FIX-11)
    # ------------------------------------------------------------------
    def _fit_svd(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        rank: int | None = None,
        max_error: float = 0.20,
    ) -> None:
        """Fit low-rank SVD factors ensuring parameter reduction."""
        in_f, out_f = self.in_features, self.out_features
        target_dtype = weight.dtype
        target_device = weight.device
        W_norm = torch.norm(weight, p="fro").item()

        # Parameter reduction constraint: r * (in + out) < in * out
        max_allowed_rank = max(
            1,
            min(in_f, out_f, int(0.75 * (in_f * out_f) / max(in_f + out_f, 1))),
        )
        base_r = rank if rank is not None and rank > 0 else max(1, min(in_f, out_f) // 4)
        base_r = min(base_r, max_allowed_rank)

        try:
            U, S, Vt = torch.linalg.svd(weight.float(), full_matrices=False)
            candidate_ranks = [base_r]
            if base_r * 2 <= max_allowed_rank:
                candidate_ranks.append(base_r * 2)
            if base_r * 4 <= max_allowed_rank:
                candidate_ranks.append(base_r * 4)

            best_r = base_r
            best_U_r = None
            best_V_r = None
            for cand_r in candidate_ranks:
                U_r = U[:, :cand_r] * torch.sqrt(S[:cand_r].unsqueeze(0))
                V_r = torch.sqrt(S[:cand_r].unsqueeze(1)) * Vt[:cand_r, :]
                W_cand = torch.matmul(U_r, V_r)
                err = float(
                    torch.norm(weight.float() - W_cand, p="fro").item() / max(W_norm, 1e-12)
                )
                best_r = cand_r
                best_U_r = U_r
                best_V_r = V_r
                if err <= max_error:
                    break
        except Exception:
            best_r = base_r
            best_U_r = torch.zeros(out_f, best_r, device=target_device, dtype=torch.float32)
            best_V_r = torch.zeros(best_r, in_f, device=target_device, dtype=torch.float32)

        self.can_decompose = False
        # KL-FIX-05: clean removal of Kronecker parameters.
        self.register_parameter("A", None)
        self.register_parameter("B", None)

        self.rank = best_r
        # KL-FIX-09: clean registration of SVD parameters.
        self.U = nn.Parameter(best_U_r.to(dtype=target_dtype, device=target_device))
        self.V = nn.Parameter(best_V_r.to(dtype=target_dtype, device=target_device))

        # KL-FIX-03 + KL-FIX-11: unconditional, dtype-matched bias.
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().to(dtype=target_dtype, device=target_device))
        else:
            self.register_parameter("bias", None)

        # KL-FIX-10: invalidate cache.
        self._invalidate_cache()

    # ------------------------------------------------------------------
    # Factory (KL-FIX-04, KL-FIX-08)
    # ------------------------------------------------------------------
    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int | None = None,
        max_error: float = 0.20,
        m1: int | None = None,
        n1: int | None = None,
        m2: int | None = None,
        n2: int | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> KroneckerLinear:
        """Build a KroneckerLinear from a pretrained Linear/Conv1D/LoRA wrapper."""
        target_mod = getattr(linear, "base_layer", linear)
        cls_lower = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if is_conv1d:
            in_features = raw_w.shape[0]
            out_features = raw_w.shape[1]
            weight = raw_w.t().contiguous().float()
        else:
            out_features = raw_w.shape[0]
            in_features = raw_w.shape[1]
            weight = raw_w.float()

        bias_holder = getattr(target_mod, "bias", None)
        bias = bias_holder.data.clone() if bias_holder is not None else None
        target_dtype = raw_w.dtype
        target_device = raw_w.device

        # KL-FIX-04: robust LoRA detection.
        existing_lora_A = getattr(linear, "lora_A", None)
        existing_lora_B = getattr(linear, "lora_B", None)
        effective_lora_r = lora_r
        effective_lora_alpha = lora_alpha
        if existing_lora_A is not None and existing_lora_B is not None:
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            if isinstance(w_A, torch.Tensor) and w_A.dim() >= 1:
                effective_lora_r = int(w_A.shape[0])
            scaling = getattr(linear, "scaling", None)
            if isinstance(scaling, (int, float)) and scaling > 0:
                effective_lora_alpha = float(scaling) * effective_lora_r
            elif hasattr(linear, "lora_alpha"):
                effective_lora_alpha = float(linear.lora_alpha)

        kron_layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            rank=rank,
            m1=m1,
            n1=n1,
            m2=m2,
            n2=n2,
            dtype=target_dtype,
            device=target_device,
            lora_r=effective_lora_r,
            lora_alpha=effective_lora_alpha,
        )

        # Transfer existing LoRA if shapes match.
        if (
            existing_lora_A is not None
            and existing_lora_B is not None
            and kron_layer.lora_A is not None
            and kron_layer.lora_B is not None
        ):
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            w_B = getattr(existing_lora_B, "weight", existing_lora_B)
            if isinstance(w_A, torch.Tensor) and w_A.shape == kron_layer.lora_A.shape:
                kron_layer.lora_A.data.copy_(
                    w_A.to(kron_layer.lora_A.device, dtype=kron_layer.lora_A.dtype)
                )
            if isinstance(w_B, torch.Tensor) and w_B.shape == kron_layer.lora_B.shape:
                kron_layer.lora_B.data.copy_(
                    w_B.to(kron_layer.lora_B.device, dtype=kron_layer.lora_B.dtype)
                )

        W_norm = torch.norm(weight, p="fro").item()
        err = 1.0

        if kron_layer.can_decompose:
            m1_, m2_ = kron_layer.m1, kron_layer.m2
            n1_, n2_ = kron_layer.n1, kron_layer.n2
            try:
                # Rearrangement for Nearest Kronecker Product.
                W_tensor = weight.view(m1_, m2_, n1_, n2_).permute(0, 2, 1, 3).contiguous()
                W_reordered = W_tensor.view(m1_ * n1_, m2_ * n2_).float()

                U, S, Vt = torch.linalg.svd(W_reordered, full_matrices=False)
                sigma1 = S[0].item() if S.numel() > 0 else 1.0
                scale = math.sqrt(max(sigma1, 1e-12))

                u1 = U[:, 0] * scale
                v1 = Vt[0, :] * scale

                kron_layer.A.data.copy_(
                    u1.view(m1_, n1_).to(dtype=target_dtype, device=target_device)
                )
                kron_layer.B.data.copy_(
                    v1.view(m2_, n2_).to(dtype=target_dtype, device=target_device)
                )

                W_recon = torch.kron(kron_layer.A, kron_layer.B).float()
                err = float(torch.norm(weight - W_recon, p="fro").item() / max(W_norm, 1e-12))
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(
                    "KroneckerLinear NKP failed: %s; falling back to SVD.", e
                )
                err = 1.0

            if err > max_error:
                kron_layer._fit_svd(weight, bias, rank=rank, max_error=max_error)
        else:
            kron_layer._fit_svd(weight, bias, rank=rank, max_error=max_error)

        if bias is not None and kron_layer.bias is not None:
            kron_layer.bias.data.copy_(
                bias.to(dtype=kron_layer.bias.dtype, device=kron_layer.bias.device)
            )

        # KL-FIX-08: forward sanity check.
        try:
            with torch.no_grad():
                dummy = torch.randn(2, in_features, device=target_device, dtype=target_dtype)
                _ = kron_layer(dummy)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                "KroneckerLinear forward sanity check failed: %s", e
            )

        return kron_layer


__all__ = ["KroneckerLinear", "factorize_dim"]

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_device(model: nn.Module) -> torch.device:
    """KC-FIX-04: safe device resolution."""
    p = next(iter(model.parameters()), None)
    return p.device if p is not None else torch.device("cpu")


_ATTN_NAMES = {
    "attn",
    "attention",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "c_attn",
    "c_proj",
    "query",
    "key",
    "value",
    "q_lin",
    "k_lin",
    "v_lin",
    "out_lin",
}


# ===========================================================================
# Compressor
# ===========================================================================


@COMPRESSOR_REGISTRY.register("kronecker_class")
@COMPRESSOR_REGISTRY.register("kronecker_compressor_class")
class KroneckerCompressor(BaseCompressor):
    # KC-FIX-01
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module | None = None,
        rank: int | None = None,
        max_error: float = 0.80,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        target_attention_only: bool = False,
        skip_attention_qkv: bool = False,
        skip_should_compress_filter: bool = False,
        min_layer_dim: int = 8,
        cfg: Any | None = None,
        target_layers: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model if model is not None else nn.Identity(),
            name="kronecker",
            **kwargs,
        )
        conf = cfg or kwargs.get("comp_config")

        rank = _cfg_get(conf, "kronecker_rank", _cfg_get(conf, "rank", rank))
        max_error = _cfg_get(conf, "kronecker_max_error", _cfg_get(conf, "max_error", max_error))
        fallback_to_original = _cfg_get(
            conf,
            "kronecker_fallback_to_original",
            _cfg_get(conf, "fallback_to_original", fallback_to_original),
        )
        lora_r = _cfg_get(conf, "lora_r", _cfg_get(conf, "lora_rank", lora_r))
        lora_alpha = _cfg_get(conf, "lora_alpha", lora_alpha)
        target_attention_only = _cfg_get(
            conf,
            "kronecker_target_attention_only",
            _cfg_get(conf, "target_attention_only", target_attention_only),
        )
        skip_attention_qkv = _cfg_get(conf, "skip_attention_qkv", skip_attention_qkv)
        skip_should_compress_filter = _cfg_get(
            conf,
            "skip_should_compress_filter",
            skip_should_compress_filter,
        )
        min_layer_dim = _cfg_get(conf, "min_layer_dim", min_layer_dim)
        target_layers = _cfg_get(conf, "target_layers", target_layers)

        self.rank = int(rank) if rank is not None and rank > 0 else None
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.target_attention_only = bool(target_attention_only)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.skip_should_compress_filter = bool(skip_should_compress_filter)
        self.min_layer_dim = max(1, int(min_layer_dim))
        self.target_layers = target_layers
        self.cfg = conf

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int | None] = {}
        self.layer_modes: dict[str, str] = {}
        self.stats: dict[str, Any] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            if any(
                p in name.lower()
                for p in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "query",
                    "key",
                    "value",
                    "c_attn",
                )
            ):
                return True
        return False

    # KC-FIX-12: debug-enriched target check.
    def _is_target_module(self, name: str, module: nn.Module, _debug: bool = False) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ("lora_A", "lora_B", "base_layer")):
            if _debug:
                logger.info(f"  [skip] {name}: lora/base_layer part")
            return False
        if any(
            skip in name_lower
            for skip in (
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "router",
                "gate",
            )
        ):
            if _debug:
                logger.info(f"  [skip] {name}: skip keyword")
            return False

        if self.target_attention_only:
            if not any(k in name_lower for k in _ATTN_NAMES):
                if _debug:
                    logger.info(f"  [skip] {name}: not attention")
                return False

        if self._is_gqa_asymmetric(name, module):
            if _debug:
                logger.info(f"  [skip] {name}: GQA asymmetric")
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_lower = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            if _debug:
                logger.info(f"  [skip] {name}: not linear-like")
            return False
        if isinstance(target_mod, KroneckerLinear):
            if _debug:
                logger.info(f"  [skip] {name}: already KroneckerLinear")
            return False

        w = getattr(target_mod, "weight", None)
        if w is None:
            if _debug:
                logger.info(f"  [skip] {name}: weight None")
            return False

        # KC-FIX-11: filter tiny layers.
        if min(w.shape[0], w.shape[1]) < self.min_layer_dim:
            if _debug:
                logger.info(f"  [skip] {name}: dim too small {tuple(w.shape)}")
            return False

        # KC-FIX-14: guard against framework errors.
        if not self.skip_should_compress_filter:
            try:
                if not should_compress_linear(self.model, name):
                    if _debug:
                        logger.info(f"  [skip] {name}: should_compress_linear=False")
                    return False
            except Exception as e:
                if _debug:
                    logger.info(f"  [skip] {name}: should_compress_linear raised {e}")
                return False

        # KC-FIX-13: empty target_layers means "no targets".
        if self.target_layers is not None:
            ok = any(t in name for t in self.target_layers)
            if not ok and _debug:
                logger.info(f"  [skip] {name}: not in target_layers")
            return ok

        if _debug:
            logger.info(f"  [TARGET] {name}: shape={tuple(w.shape)}")
        return True

    # ------------------------------------------------------------------
    def compress(self, model: nn.Module | None = None) -> tuple[nn.Module, dict[str, Any]]:
        # KC-FIX-09: idempotence.
        if getattr(self.model, "_kronecker_compressed", False):
            logger.warning("KroneckerCompressor: model already compressed; skipping.")
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        if model is not None:
            self.model = model
            try:
                self._original_params = self._count_params(self.model)
            except Exception:
                self._original_params = sum(p.numel() for p in self.model.parameters())

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.layer_modes = {}
        seen_weights: set[int] = set()

        total_orig_params = 0
        total_comp_params = 0

        # Freeze non-targets when attention_only.
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                target_mod = getattr(module, "base_layer", module)
                cls_lower = target_mod.__class__.__name__.lower()
                if (
                    isinstance(target_mod, nn.Linear)
                    or "conv1d" in cls_lower
                    or hasattr(target_mod, "nf")
                ):
                    if not self._is_target_module(name, module):
                        for p in module.parameters():
                            p.requires_grad = False
                        self.skipped_layers.append(name)

        targets = [
            (name, mod)
            for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not targets:
            # KC-FIX-12: diagnostic breakdown.
            logger.warning("KroneckerCompressor: no targets found. Diagnostic...")
            n_scanned = 0
            for name, module in self.model.named_modules():
                n_scanned += 1
                self._is_target_module(name, module, _debug=True)
            logger.warning(
                "KroneckerCompressor diagnostic: %d modules scanned, 0 targets.",
                n_scanned,
            )
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        logger.info("KroneckerCompressor: %d target layers identified.", len(targets))

        for name, module in targets:
            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            cls_lower = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")
            raw_w = target_mod.weight.data

            orig_layer_params = raw_w.numel()
            bias_mod = getattr(target_mod, "bias", None)
            if bias_mod is not None:
                orig_layer_params += bias_mod.numel()
            total_orig_params += orig_layer_params

            W_target = raw_w.t().float() if is_conv1d else raw_w.float()
            W_norm = torch.norm(W_target, p="fro").item()

            new_layer: KroneckerLinear | None = None
            err = 1.0
            try:
                new_layer = KroneckerLinear.from_linear(
                    module,
                    rank=self.rank,
                    max_error=self.max_error,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                W_comp = new_layer.get_weight().float()
                err = float(torch.norm(W_target - W_comp, p="fro").item() / max(W_norm, 1e-12))
            except Exception as e:
                logger.warning("KroneckerLinear factorization failed on %s: %s", name, e)
                new_layer = None
                err = 1.0

            self.layer_errors[name] = err

            # KC-FIX-15: only set mode/rank after decision.
            if new_layer is None or err > self.max_error:
                logger.warning(
                    "Kronecker: error %.4f on %s > %.4f. %s",
                    err,
                    name,
                    self.max_error,
                    ("Falling back." if self.fallback_to_original else "Keeping original weight."),
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    self.layer_ranks[name] = None
                    self.layer_modes[name] = "fallback"
                    for p in module.parameters():
                        p.requires_grad = False
                    # KC-FIX-06: use orig_layer_params (no double-count).
                    total_comp_params += orig_layer_params
                    continue
                else:
                    self.fallback_layers.append(name)
                    self.layer_ranks[name] = None
                    self.layer_modes[name] = "fallback"
                    total_comp_params += orig_layer_params
                    continue

            # Accepted.
            mode = "kronecker" if getattr(new_layer, "can_decompose", False) else "svd"
            self.layer_modes[name] = mode
            self.layer_ranks[name] = getattr(new_layer, "rank", None)

            comp_layer_params = sum(p.numel() for p in new_layer.parameters())
            total_comp_params += comp_layer_params

            try:
                replace_module(self.model, name, new_layer)
                self.compressed_layers.append(name)
            except Exception as e:
                logger.warning("Kronecker: replace_module failed for %s: %s", name, e)
                self.fallback_layers.append(name)

        # KC-FIX-08: robust forward check.
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = _get_device(self.model)
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                try:
                    self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception as e2:
                        logger.warning("Kronecker forward check failed: %s", e2)
                        forward_check_success = False
        except Exception as e:
            logger.warning("Kronecker forward check raised: %s", e)
            forward_check_success = False

        # KC-FIX-03: safe param count.
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        # KC-FIX-05: avg_err only over compressed layers.
        compressed_errs = [
            self.layer_errors[n] for n in self.compressed_layers if n in self.layer_errors
        ]
        avg_err = sum(compressed_errs) / len(compressed_errs) if compressed_errs else 0.0
        param_ratio = (
            total_comp_params / max(total_orig_params, 1) if total_orig_params > 0 else 1.0
        )

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "skipped_layers": self.skipped_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "layer_modes": self.layer_modes,
                "rank": self.rank,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "target_attention_only": self.target_attention_only,
                "forward_check_success": forward_check_success,
                "avg_reconstruction_error": avg_err,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layer_count": len(self.fallback_layers),
                "effective_parameter_ratio": round(param_ratio, 4),
                "method": "kronecker",
            }
        )
        self.stats = stats

        # KC-FIX-09: write stats onto model.
        try:
            if not hasattr(self.model, "compression_stats") or self.model.compression_stats is None:
                self.model.compression_stats = {}
            self.model.compression_stats.update(
                {
                    "num_compressed": len(self.compressed_layers),
                    "num_fallback": len(self.fallback_layers),
                    "compression_method": "kronecker",
                    "effective_parameter_ratio": param_ratio,
                    "avg_reconstruction_error": avg_err,
                }
            )
        except Exception:
            pass

        self.model._kronecker_compressed = True
        logger.info(
            "Kronecker: %d compressed, %d fallback. param_ratio=%.4f, avg_err=%.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            param_ratio,
            avg_err,
        )
        return self.model, stats


# ===========================================================================
# Factory (KC-FIX-02)
# ===========================================================================


@COMPRESSOR_REGISTRY.register("kronecker")
@COMPRESSOR_REGISTRY.register("kronecker_compressor")
@COMPRESSOR_REGISTRY.register("kronecker_factory")
def kronecker_factory(
    model: nn.Module,
    comp_config: Any | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """Kronecker compressor factory."""
    rank = rank or _cfg_get(comp_config, "kronecker_rank", _cfg_get(comp_config, "rank", 128))
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "kronecker_max_error", _cfg_get(comp_config, "max_error", 0.80)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "kronecker_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 16))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "kronecker_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        ),
    )
    skip_filter = kwargs.pop(
        "skip_should_compress_filter",
        _cfg_get(comp_config, "skip_should_compress_filter", False),
    )
    min_layer_dim = kwargs.pop(
        "min_layer_dim",
        _cfg_get(comp_config, "min_layer_dim", 8),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = KroneckerCompressor(
        model,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        skip_should_compress_filter=skip_filter,
        min_layer_dim=min_layer_dim,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "KroneckerCompressor",
    "kronecker_factory",
]
