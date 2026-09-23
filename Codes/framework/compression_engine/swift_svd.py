"""
Swift-SVD: fast closed-form truncated SVD with symmetric square-root singular value distribution.
Registered as 'swift_svd'.

Bug-fix map:
  SV-FIX-01  redundant exception classes in SVD try/except collapsed.
  SV-FIX-02  _compressed_params initialized in __init__.
  SV-FIX-03  raw_shape reporting correct for both Linear and Conv1D.
  SV-FIX-04  LoRA transfer via data-shape check (not attribute-name guessing).
  SV-FIX-05  get_compression_stats + _count_params guarded.
  SV-FIX-06  cross-device handling for CPU-only SVD fallback.
  SV-FIX-07  weight-sharing detection across wrappers.
  SV-FIX-08  max_retries=0 handled without division by zero.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
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


@COMPRESSOR_REGISTRY.register("swift_svd_class")
class SwiftSVDCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    Swift-SVD Compressor: Fast Closed-Form Truncated SVD.

    Decomposes dense weight matrices into rank-r factorized layers with symmetric
    square-root singular value splitting (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T)
    and replaces nn.Linear/Conv1D with factorized SVDLinear modules.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        rank_ratio: float = 0.5,
        target_modules: Optional[List[str]] = None,
        max_error: float = 0.60,
        max_retries: int = 3,
        fallback_to_original: bool = True,
        validation_calibration: bool = False,
        calibration_threshold: float = 0.10,
        target_attention_only: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="swift_svd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = _cfg_get(cfg, "swift_svd_rank", _cfg_get(cfg, "rank", rank))
            rank_ratio = _cfg_get(cfg, "swift_svd_rank_ratio",
                                  _cfg_get(cfg, "rank_ratio", rank_ratio))
            max_error = _cfg_get(cfg, "max_error",
                                 _cfg_get(cfg, "max_reconstruction_error", max_error))
            max_retries = _cfg_get(cfg, "max_retries", max_retries)
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            validation_calibration = _cfg_get(cfg, "validation_calibration", validation_calibration)
            calibration_threshold = _cfg_get(cfg, "calibration_threshold", calibration_threshold)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]
        if "max_error" in kwargs and kwargs["max_error"] is not None:
            max_error = float(kwargs["max_error"])
        if "max_retries" in kwargs and kwargs["max_retries"] is not None:
            max_retries = int(kwargs["max_retries"])
        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])
        if "validation_calibration" in kwargs:
            validation_calibration = bool(kwargs["validation_calibration"])
        if "calibration_threshold" in kwargs:
            calibration_threshold = float(kwargs["calibration_threshold"])
        if "target_attention_only" in kwargs:
            target_attention_only = bool(kwargs["target_attention_only"])
        if "lora_r" in kwargs:
            lora_r = int(kwargs["lora_r"])
        if "lora_alpha" in kwargs:
            lora_alpha = float(kwargs["lora_alpha"])

        self.rank = rank
        self.rank_ratio = float(rank_ratio)
        self.target_modules = target_modules
        self.max_error = float(max_error)
        self.max_retries = max(0, int(max_retries))
        self.fallback_to_original = bool(fallback_to_original)
        self.validation_calibration = bool(validation_calibration)
        self.calibration_threshold = float(calibration_threshold)
        self.target_attention_only = bool(target_attention_only)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)

        self.compressed_layers: List[Dict[str, Any]] = []
        # SV-FIX-02: initialize in __init__.
        self._compressed_params: int = 0
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if self.target_attention_only:
            attn_keywords = ["attn", "attention", "q_proj", "k_proj", "v_proj",
                             "o_proj", "c_attn", "c_proj", "query", "key", "value"]
            if not any(k in name.lower() for k in attn_keywords):
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
                and not isinstance(base_layer, (nn.Embedding, nn.LayerNorm, nn.RMSNorm))
            )
        )
        if not is_linear_like:
            return False
        if isinstance(module, SVDLinear) or isinstance(base_layer, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    # Per-layer compression
    # ------------------------------------------------------------------
    def compress_linear(
        self,
        layer: nn.Module,
        rank: Optional[int] = None,
        name: str = "",
    ) -> Tuple[Optional[SVDLinear], float, int]:
        """
        Decompose a single linear or Conv1D layer via truncated SVD.
        Returns (svd_layer_or_None, relative_frobenius_error, final_rank).
        """
        base_layer = getattr(layer, "base_layer", layer)
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        cls_name = base_layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(base_layer, "nf")

        raw_w = base_layer.weight.data
        # Always move weight to a canonical (out_features, in_features) form.
        if is_conv1d:
            W = raw_w.t().contiguous().float()
            out_f, in_f = W.shape[0], W.shape[1]
        else:
            W = raw_w.float()
            out_f, in_f = W.shape[0], W.shape[1]

        # Param budget: r * (in_f + out_f) < in_f * out_f
        max_param_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = min(out_f, in_f, max_param_r)
        if max_r < 1:
            return None, 1.0, 0

        # Target rank
        if rank is not None:
            target_r = max(1, min(int(rank), max_r))
        else:
            target_r = max(1, min(int(round(min(out_f, in_f) * self.rank_ratio)), max_r))

        # SV-FIX-01: single try/except with explicit CUDA OOM handling.
        U: Optional[torch.Tensor] = None
        S: Optional[torch.Tensor] = None
        Vt: Optional[torch.Tensor] = None
        try:
            if W.is_cuda:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            else:
                W_dev = W.to(device) if device.type == "cuda" else W
                U, S, Vt = torch.linalg.svd(W_dev, full_matrices=False)
        except torch.cuda.OutOfMemoryError:
            try:
                torch.cuda.empty_cache()
                U, S, Vt = torch.linalg.svd(W.cpu(), full_matrices=False)
                U, S, Vt = U.to(device), S.to(device), Vt.to(device)
            except Exception as e2:
                logger.warning("SwiftSVD SVD (OOM fallback) failed for %s: %s", name, e2)
                return None, 1.0, 0
        except Exception as e:
            logger.warning("SwiftSVD SVD failed for %s: %s", name, e)
            return None, 1.0, 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        norm_orig = torch.linalg.norm(W).item() + 1e-8
        cur_r = target_r
        best_factors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        best_err = float("inf")
        best_r = cur_r

        # Iterative rank-step retries
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            sqrt_S = torch.sqrt(torch.clamp(S[:cur_r], min=1e-12))
            U_r = U[:, :cur_r] * sqrt_S.unsqueeze(0)
            V_r = sqrt_S.unsqueeze(1) * Vt[:cur_r, :]

            W_rec = torch.matmul(U_r, V_r)
            rel_err = torch.linalg.norm(W - W_rec).item() / norm_orig

            if rel_err < best_err:
                best_err = rel_err
                best_factors = (U_r, V_r)
                best_r = cur_r

            if rel_err <= self.max_error or cur_r >= max_r:
                break

            remaining = attempts - attempt - 1
            if remaining <= 0:
                break
            step = max(4, (max_r - cur_r) // max(1, remaining))
            cur_r = min(cur_r + step, max_r)

        if best_err > self.max_error:
            logger.warning(
                "SwiftSVD: %s error (%.4f) > max_error (%.4f) at rank %d. %s",
                name, best_err, self.max_error, best_r,
                "Keeping original." if self.fallback_to_original else "Using compressed layer anyway.",
            )
            if self.fallback_to_original:
                return None, best_err, best_r

        if best_factors is None:
            return None, best_err, best_r

        U_final, V_final = best_factors
        U_final = U_final.to(dtype=dtype, device=device)
        V_final = V_final.to(dtype=dtype, device=device)

        has_bias = getattr(base_layer, "bias", None) is not None
        has_base_layer = hasattr(layer, "base_layer")
        has_existing_lora = hasattr(layer, "lora_A") and hasattr(layer, "lora_B")

        if has_base_layer:
            # Wrapped: SVD becomes the new base, existing LoRA stays on outer.
            effective_lora_r = 0
        elif has_existing_lora:
            detected = self._detect_lora_rank(layer)
            effective_lora_r = detected if detected > 0 else self.lora_r
        else:
            effective_lora_r = self.lora_r

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=best_r,
            bias=has_bias,
            dtype=dtype,
            device=device,
            lora_r=effective_lora_r,
            lora_alpha=self.lora_alpha,
        )
        svd_layer.set_low_rank_factors(U_final, V_final)

        if has_bias and svd_layer.bias is not None and base_layer.bias is not None:
            svd_layer.bias.data.copy_(base_layer.bias.data.to(dtype=dtype, device=device))

        # SV-FIX-04: robust LoRA copy via data-shape check.
        if not has_base_layer and has_existing_lora and getattr(svd_layer, "use_lora", False):
            try:
                src_lA = getattr(layer, "lora_A", None)
                src_lB = getattr(layer, "lora_B", None)
                if src_lA is not None and src_lB is not None:
                    src_lA_w = getattr(src_lA, "weight", src_lA)
                    src_lB_w = getattr(src_lB, "weight", src_lB)
                    dst_lA_w = getattr(svd_layer.lora_A, "weight", svd_layer.lora_A)
                    dst_lB_w = getattr(svd_layer.lora_B, "weight", svd_layer.lora_B)
                    if src_lA_w.shape == dst_lA_w.shape:
                        dst_lA_w.data.copy_(src_lA_w.data)
                    if src_lB_w.shape == dst_lB_w.shape:
                        dst_lB_w.data.copy_(src_lB_w.data)
            except Exception as e:
                logger.warning("SwiftSVD LoRA copy warning for %s: %s", name, e)

        # Dimensional integrity check
        if svd_layer.in_features != in_f or svd_layer.out_features != out_f:
            logger.error(
                "SwiftSVD: shape mismatch for %s: expected (in=%d, out=%d), got (in=%d, out=%d)",
                name, in_f, out_f, svd_layer.in_features, svd_layer.out_features,
            )
            return None, best_err, best_r

        # Optional calibration validation
        if self.validation_calibration:
            try:
                with torch.no_grad():
                    dummy_x = torch.randn(1, in_f, device=device, dtype=dtype)
                    orig_out = base_layer(dummy_x)
                    svd_out = svd_layer(dummy_x)
                    rel_diff = (
                        torch.norm(orig_out.float() - svd_out.float())
                        / (torch.norm(orig_out.float()) + 1e-8)
                    ).item()
                    if rel_diff > self.calibration_threshold:
                        logger.warning(
                            "SwiftSVD: %s calibration diff=%.4f > %.4f. Keeping original.",
                            name, rel_diff, self.calibration_threshold,
                        )
                        return None, rel_diff, best_r
            except Exception as ce:
                logger.warning("SwiftSVD: %s calibration error %s. Proceeding.", name, ce)

        return svd_layer, best_err, best_r

    # ------------------------------------------------------------------
    # LoRA rank detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_lora_rank(module: nn.Module) -> int:
        for attr in ("r", "lora_r", "rank"):
            val = getattr(module, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        la = getattr(module, "lora_A", None)
        if la is not None:
            if hasattr(la, "out_features"):
                try:
                    return int(la.out_features)
                except Exception:
                    pass
            if hasattr(la, "weight") and la.weight is not None and la.weight.dim() >= 1:
                return int(la.weight.shape[0])
        return 0

    # ------------------------------------------------------------------
    # Main compress
    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all eligible linear and Conv1D layers."""
        # SV-FIX-07: deduplicate shared weights.
        seen_weights: set = set()
        target_layers: List[Tuple[str, nn.Module]] = []
        for name, mod in self.model.named_modules():
            if not self._is_target_module(name, mod):
                continue
            base = getattr(mod, "base_layer", mod)
            w = getattr(base, "weight", None)
            if w is None:
                continue
            w_id = id(w)
            if w_id in seen_weights:
                continue
            seen_weights.add(w_id)
            target_layers.append((name, mod))

        if not target_layers:
            logger.warning("SwiftSVD: No compressible linear layers found.")
            try:
                stats = self.get_compression_stats()
            except Exception:
                stats = {}
            return self.model, stats

        self.compressed_layers = []
        fallback_layers: List[str] = []
        total_error = 0.0

        for name, module in target_layers:
            svd_layer, rel_err, final_rank = self.compress_linear(module, rank=self.rank, name=name)

            if svd_layer is not None:
                # SV-FIX-03: raw_shape from base_layer, correctly oriented.
                base_mod = getattr(module, "base_layer", module)
                cls_name = base_mod.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(base_mod, "nf")
                if is_conv1d:
                    expected_in = base_mod.weight.shape[0]
                    expected_out = base_mod.weight.shape[1]
                else:
                    expected_in = getattr(base_mod, "in_features", base_mod.weight.shape[1])
                    expected_out = getattr(base_mod, "out_features", base_mod.weight.shape[0])

                if svd_layer.in_features != expected_in or svd_layer.out_features != expected_out:
                    logger.error(
                        "SwiftSVD: skipping %s due to shape mismatch "
                        "(expected in=%d, out=%d; got in=%d, out=%d)",
                        name, expected_in, expected_out,
                        svd_layer.in_features, svd_layer.out_features,
                    )
                    fallback_layers.append(name)
                    continue

                if hasattr(module, "base_layer"):
                    module.base_layer = svd_layer
                    for attr in ("lora_A", "lora_B"):
                        sub = getattr(module, attr, None)
                        if sub is None:
                            continue
                        params = sub.parameters() if hasattr(sub, "parameters") else iter([sub])
                        for p in params:
                            p.requires_grad = True
                else:
                    replace_module(self.model, name, svd_layer)

                self.compressed_layers.append({
                    "name": name,
                    "original_shape": (expected_out, expected_in),
                    "rank": final_rank,
                    "rel_error": float(rel_err),
                })
                total_error += rel_err
            else:
                fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                logger.info("SwiftSVD: kept original layer for %s.", name)

        # SV-FIX-02: safe param count.
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        # SV-FIX-05: guard stats.
        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        avg_err = (total_error / len(self.compressed_layers)) if self.compressed_layers else 0.0
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": fallback_layers,
            "num_compressed_layers": len(self.compressed_layers),
            "num_fallback": len(fallback_layers),
            "avg_reconstruction_error": avg_err,
            "rank": self.rank,
            "rank_ratio": self.rank_ratio,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "param_reduction_pct": stats.get("param_reduction_pct", 0.0),
            "method": "swift_svd",
        })
        self.stats = stats
        logger.info(
            "SwiftSVD: %d layers compressed, avg error %.4f, param reduction %.2f%%",
            len(self.compressed_layers), avg_err, stats.get("param_reduction_pct", 0.0),
        )
        return self.model, stats


# ===========================================================================
# Factories
# ===========================================================================

@COMPRESSOR_REGISTRY.register("swift_svd")
@COMPRESSOR_REGISTRY.register("swift_svd_factory")
@COMPRESSOR_REGISTRY.register("swift_svd_compressor")
def swift_svd_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> Any:
    """Swift-SVD factory function."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    target_attn = kwargs.pop("target_attention_only",
                             _cfg_get(comp_config, "target_attention_only", False))
    compressor = SwiftSVDCompressor(
        model,
        comp_config=comp_config,
        target_attention_only=target_attn,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


@COMPRESSOR_REGISTRY.register("inject_swift_svd")
@COMPRESSOR_REGISTRY.register("swift_svd_inject")
def inject_swift_svd(model: Any = None, **kwargs: Any) -> Any:
    """Safe injection hook for Swift-SVD."""
    if model is None:
        return None
    try:
        return swift_svd_factory(model, **kwargs)
    except Exception as e:
        logger.error("inject_swift_svd failed: %s", e)
        return None