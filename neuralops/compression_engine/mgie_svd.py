"""
MGIE-SVD: Maximum Information Entropy SVD Rank Selection.

Registered as 'mgie', 'mgie_svd', 'mgie_factory', 'mgie_svd_factory'.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from neuralops.compression_engine.aa_svd_compressor import SVDLinear
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.pruning_utils import adaptive_max_error
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


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
    "self_attn",
}


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("mgie")
@COMPRESSOR_REGISTRY.register("mgie_svd")
class MGIECompressor(BaseCompressor):
    """
    MGIE-SVD Compressor (Maximum Information Entropy SVD).
    """

    # MGIE-FIX-01
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        rank: int | None = None,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        target_attention_only: bool = False,
        min_rank: int = 4,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="mgie", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))

        if cfg is not None:
            rank = _cfg_get(cfg, "mgie_rank", _cfg_get(cfg, "rank", rank))
            max_error = _cfg_get(
                cfg, "mgie_max_error", _cfg_get(cfg, "max_error", max_error)
            )
            fallback_to_original = _cfg_get(
                cfg,
                "mgie_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(
                cfg, "target_attention_only", target_attention_only
            )
            min_rank = _cfg_get(cfg, "min_rank", min_rank)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.rank = rank
        if max_error is None or max_error == 0.50:
            max_error = adaptive_max_error(self.rank, 768, 768, "distilbert")

        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = bool(target_attention_only)
        self.min_rank = max(1, int(min_rank))
        self.target_modules = target_modules

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if (
            num_heads is not None
            and num_kv_heads is not None
            and num_kv_heads < num_heads
        ):
            name_lower = name.lower()
            if any(
                p in name_lower
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

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        # MGIE-FIX-07: skip already-compressed SVDLinear
        if isinstance(module, SVDLinear):
            return False
        name_l = name.lower()
        if any(
            k in name_l
            for k in (
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "layernorm",
                "rmsnorm",
                "batchnorm",
            )
        ):
            return False

        cls_name = module.__class__.__name__.lower()
        is_linear_like = (
            isinstance(module, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(module, "nf")
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self._is_gqa_asymmetric(name, module):
            return False

        if self.target_attention_only:
            if not any(k in name_l for k in _ATTN_NAMES):
                return False

        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    def _compute_entropy_rank(self, S: torch.Tensor, max_rank: int) -> int:
        """
        Compute rank from singular entropy of squared singular values
        (energy distribution).
        """
        if max_rank <= 1:
            return 1

        energy = S**2
        total_energy = energy.sum()
        if total_energy <= 1e-12:
            return max(1, min(self.min_rank, max_rank))

        p = torch.clamp(energy / total_energy, min=1e-12, max=1.0)
        entropy = -torch.sum(p * torch.log(p)).item()
        max_entropy = math.log(float(max_rank))

        normalized_h = entropy / max(1e-6, max_entropy)
        allocated_rank = max(
            self.min_rank,
            int(round(normalized_h * max_rank)),
        )

        if self.rank is not None:
            allocated_rank = min(allocated_rank, self.rank)

        return max(1, min(allocated_rank, max_rank))

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}

        seen_weights: set[int] = set()

        # MGIE-FIX-04: compute original parameter count properly.
        total_orig_params = sum(p.numel() for p in self.model.parameters())

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_layer = getattr(module, "base_layer", module)
            if not hasattr(target_layer, "weight") or target_layer.weight is None:
                continue

            w_id = id(target_layer.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = target_layer.weight.data
            cls_name = target_layer.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_layer, "nf")

            W = (
                raw_w.t().contiguous().clone().float()
                if is_conv1d
                else raw_w.clone().float()
            )
            d_out, d_in = W.shape
            max_r = min(d_out, d_in)
            device = raw_w.device
            dtype = raw_w.dtype

            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = self._compute_entropy_rank(S, max_r)

                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
                V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

                W_approx = torch.nan_to_num(U_r @ V_r, nan=0.0)
                w_norm = torch.norm(W, p="fro").clamp_min(1e-12).item()
                err = (torch.norm(W.to(device) - W_approx, p="fro") / w_norm).item()
                self.layer_errors[name] = float(err)
                self.layer_ranks[name] = r

                if not math.isfinite(err) or err > self.max_error:
                    logger.warning(
                        "MGIE: reconstruction error %.4f on %s exceeded %.4f.",
                        err,
                        name,
                        self.max_error,
                    )
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters():
                            p.requires_grad = False
                        continue

                existing_lora_r = getattr(module, "r", getattr(module, "lora_r", 0))
                eff_lora_r = existing_lora_r if existing_lora_r > 0 else self.lora_r
                eff_lora_alpha = getattr(
                    module,
                    "lora_alpha",
                    getattr(module, "alpha", self.lora_alpha),
                )

                svd_layer = SVDLinear(
                    in_features=d_in,
                    out_features=d_out,
                    rank=r,
                    bias=target_layer.bias is not None,
                    dtype=dtype,
                    device=device,
                    lora_r=eff_lora_r,
                    lora_alpha=eff_lora_alpha,
                )

                # MGIE-FIX-08: verify API availability
                if not hasattr(svd_layer, "set_low_rank_factors"):
                    raise AttributeError(
                        "SVDLinear does not implement set_low_rank_factors; "
                        "MGIE requires an updated aa_svd_compressor."
                    )
                svd_layer.set_low_rank_factors(U_r, V_r)

                # MGIE-FIX-05: bias copy under no_grad
                if target_layer.bias is not None and svd_layer.bias is not None:
                    with torch.no_grad():
                        svd_layer.bias.data.copy_(
                            target_layer.bias.data.to(
                                device=svd_layer.bias.device,
                                dtype=svd_layer.bias.dtype,
                            )
                        )

                # Preserve pre-existing LoRA adapter parameters if wrapped
                if (
                    hasattr(module, "lora_A")
                    and hasattr(module, "lora_B")
                    and svd_layer.lora_A is not None
                    and svd_layer.lora_B is not None
                ):
                    with torch.no_grad():
                        src_A = (
                            module.lora_A.weight.data
                            if hasattr(module.lora_A, "weight")
                            else module.lora_A.data
                        )
                        src_B = (
                            module.lora_B.weight.data
                            if hasattr(module.lora_B, "weight")
                            else module.lora_B.data
                        )
                        if (
                            src_A.shape == svd_layer.lora_A.shape
                            and src_B.shape == svd_layer.lora_B.shape
                        ):
                            svd_layer.lora_A.copy_(
                                src_A.to(
                                    svd_layer.lora_A.device,
                                    dtype=svd_layer.lora_A.dtype,
                                )
                            )
                            svd_layer.lora_B.copy_(
                                src_B.to(
                                    svd_layer.lora_B.device,
                                    dtype=svd_layer.lora_B.dtype,
                                )
                            )
                    svd_layer.lora_A.requires_grad = True
                    svd_layer.lora_B.requires_grad = True

                # MGIE-FIX-02: preserve LoRA wrapper.
                if hasattr(module, "base_layer"):
                    module.base_layer = svd_layer
                else:
                    replace_module(self.model, name, svd_layer)

                self.compressed_layers.append(name)

            except Exception as e:
                logger.warning("MGIE: failed on %s: %s", name, e)
                self.fallback_layers.append(name)
                self.layer_errors[name] = 1.0
                for p in module.parameters():
                    p.requires_grad = False

        # --------------------------------------------------------------
        # Forward sanity check (MGIE-FIX-03)
        # --------------------------------------------------------------
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                try:
                    self.model(input_ids=dummy)
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception:
                        cfg = getattr(self.model, "config", None)
                        hidden = getattr(cfg, "hidden_size", None) if cfg else None
                        if hidden is None:
                            first_lin = next(
                                (
                                    m
                                    for m in self.model.modules()
                                    if isinstance(m, nn.Linear)
                                ),
                                None,
                            )
                            hidden = getattr(first_lin, "in_features", 8)
                        dt = next(self.model.parameters()).dtype
                        dummy_feat = torch.randn(1, 8, hidden, device=dev, dtype=dt)
                        self.model(dummy_feat)
        except Exception as e:
            logger.warning("MGIE forward sanity check failed: %s", e)
            forward_check_success = False

        # --------------------------------------------------------------
        # Stats
        # --------------------------------------------------------------
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        # MGIE-FIX-04: accurate ratio
        eff_ratio = self._compressed_params / max(1, total_orig_params)

        avg_err = (
            sum(self.layer_errors.values()) / max(1, len(self.layer_errors))
            if self.layer_errors
            else 0.0
        )

        stats = self.get_compression_stats()
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
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "total_params_before": total_orig_params,
                "total_params_after": self._compressed_params,
                "effective_parameter_ratio": round(eff_ratio, 4),
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "forward_check_success": forward_check_success,
                "method": "mgie_svd",
            }
        )
        self.stats = stats
        logger.info(
            "Compressor mgie finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("mgie_factory")
@COMPRESSOR_REGISTRY.register("mgie_svd_factory")
@COMPRESSOR_REGISTRY.register("mgie_svd_compressor_factory")
def mgie_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """MGIE factory function."""
    rank = kwargs.pop(
        "rank",
        _cfg_get(comp_config, "mgie_rank", _cfg_get(comp_config, "rank", None)),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config, "mgie_max_error", _cfg_get(comp_config, "max_error", 0.50)
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "mgie_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "target_attention_only", False),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = MGIECompressor(
        model,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


mgie_svd_factory = mgie_factory

__all__ = [
    "MGIECompressor",
    "mgie_factory",
    "mgie_svd_factory",
]
