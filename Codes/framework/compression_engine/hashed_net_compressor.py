"""
HashedNet Compressor: replaces nn.Linear and Conv1D layers with HashedLinear.
Registered in COMPRESSOR_REGISTRY as 'hashed_net'. Production hardened with
reconstruction validation, adaptive bucket scaling, Conv1D support, and safe fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

from .hashed_linear import HashedLinear

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("hashed_net")
class HashedNetCompressor(BaseCompressor):
    """
    Compress a model by replacing target linear/Conv1D layers with HashedLinear,
    which shares weights globally through compact hash tables.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Adaptive bucket retries: automatically doubles bucket budget when error exceeds threshold.
    - Safe fallback to original module if error remains high or table contains NaNs/Infs.
    - Conv1D support (GPT-2 c_attn, c_proj) via transposed weight projection.
    - Unified critical-layer protection via should_compress_linear.
    - Optional trainable LoRA adapters and hash-table freezing.
    - Clean rank / bucket budget derivation guaranteeing parameter reduction.
    - Detailed per-layer compression statistics and logging.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_buckets: Optional[int] = None,
        compression_ratio: Optional[float] = None,
        rank: Optional[Union[int, float]] = None,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        use_sign: bool = True,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 1.0,
        freeze_hash_table: bool = False,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="hashed_net", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)

        if conf is not None:
            n_buckets = getattr(conf, "hashed_net_buckets", getattr(conf, "n_buckets", n_buckets))
            compression_ratio = getattr(conf, "compression_ratio", getattr(conf, "target_param_ratio", compression_ratio))
            rank = getattr(conf, "rank", getattr(conf, "hashed_net_rank", rank))
            max_error = getattr(conf, "hashed_net_max_error", getattr(conf, "max_error", max_error))
            fallback_to_original = getattr(
                conf, "hashed_net_fallback_to_original", getattr(conf, "fallback_to_original", fallback_to_original)
            )
            use_sign = getattr(conf, "use_sign", use_sign)
            lora_rank = getattr(conf, "lora_rank", getattr(conf, "hashed_net_lora_rank", lora_rank))
            lora_alpha = getattr(conf, "lora_alpha", getattr(conf, "hashed_net_lora_alpha", lora_alpha))
            freeze_hash_table = getattr(conf, "freeze_hash_table", freeze_hash_table)
            target_layers = getattr(conf, "target_layers", target_layers)

        # Parse bucket budget or ratio
        self.compression_ratio: Optional[float] = None
        if compression_ratio is not None:
            self.compression_ratio = float(compression_ratio)
        elif rank is not None and 0.0 < float(rank) <= 1.0:
            self.compression_ratio = float(rank)

        if n_buckets is not None:
            self.base_buckets: Optional[int] = max(16, int(n_buckets))
        elif rank is not None and float(rank) > 1.0:
            self.base_buckets = max(16, int(rank) * 32)
        else:
            self.base_buckets = 65536 if self.compression_ratio is None else None

        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.max_retries = int(max_retries)
        self.use_sign = bool(use_sign)
        self.lora_rank = int(lora_rank) if lora_rank is not None and lora_rank > 0 else None
        self.lora_alpha = float(lora_alpha)
        self.freeze_hash_table = bool(freeze_hash_table)
        self.target_layers = target_layers

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_buckets: Dict[str, int] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(module, HashedLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_layers is not None:
            return any(t in name for t in self.target_layers)
        return True

    def _determine_layer_buckets(self, in_f: int, out_f: int) -> int:
        """Compute target bucket count guaranteeing parameter reduction."""
        total_weights = in_f * out_f
        max_allowed_buckets = max(16, int(total_weights * 0.75))

        if self.compression_ratio is not None:
            raw_b = int(total_weights * self.compression_ratio)
        elif self.base_buckets is not None:
            raw_b = self.base_buckets
        else:
            raw_b = 65536

        return max(16, min(raw_b, max_allowed_buckets))

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress the target model by substituting linear/Conv1D layers with HashedLinear."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_buckets = {}

        targets = [
            (name, mod)
            for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("HashedNet: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            raw_w = module.weight.data

            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                W_target = raw_w.t().float()
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]
                W_target = raw_w.float()

            W_norm = torch.norm(W_target, p="fro").item()
            total_weights = in_f * out_f
            max_allowed_buckets = max(16, int(total_weights * 0.90))

            init_buckets = self._determine_layer_buckets(in_f, out_f)
            current_buckets = init_buckets
            layer_compressed: Optional[HashedLinear] = None
            final_err = 1.0

            for attempt in range(self.max_retries + 1):
                try:
                    candidate = HashedLinear.from_linear(
                        module,
                        n_buckets=current_buckets,
                        use_sign=self.use_sign,
                        lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                    )
                except Exception as e:
                    logger.warning("HashedLinear instantiation failed on %s: %s", name, e)
                    break

                # Check for numerical validity
                if (
                    torch.isnan(candidate.weight_table).any()
                    or torch.isinf(candidate.weight_table).any()
                ):
                    logger.warning("HashedLinear weight table contains NaN/Inf on %s", name)
                    break

                # Measure relative Frobenius reconstruction error
                W_recon = candidate.get_weight().float()
                if W_norm < 1e-12:
                    err = 0.0
                else:
                    err = float(torch.norm(W_target - W_recon, p="fro").item() / W_norm)

                final_err = err
                layer_compressed = candidate

                if err <= self.max_error:
                    break

                # Adaptive bucket increase on error threshold exceedance
                if attempt < self.max_retries:
                    next_buckets = min(int(current_buckets * 2), max_allowed_buckets)
                    if next_buckets <= current_buckets:
                        break
                    logger.debug(
                        "HashedNet: Error %.4f > %.4f on %s. Retrying with %d buckets.",
                        err,
                        self.max_error,
                        name,
                        next_buckets,
                    )
                    current_buckets = next_buckets

            self.layer_errors[name] = final_err
            self.layer_buckets[name] = current_buckets

            if layer_compressed is None or final_err > self.max_error:
                logger.warning(
                    "HashedNet reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping HashedNet compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            if self.freeze_hash_table and layer_compressed is not None:
                layer_compressed.weight_table.requires_grad = False

            replace_module(self.model, name, layer_compressed)
            self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "layer_errors": self.layer_errors,
                "layer_buckets": self.layer_buckets,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "avg_reconstruction_error": avg_err,
                "use_sign": self.use_sign,
                "lora_rank": self.lora_rank,
            }
        )
        logger.info(
            "HashedNet compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Module) -> Tuple[HashedLinear, float]:
        """Compress a single linear or Conv1D layer into HashedLinear."""
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data
        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W_target = raw_w.t().float()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W_target = raw_w.float()

        n_b = self._determine_layer_buckets(in_f, out_f)
        hashed = HashedLinear.from_linear(
            linear,
            n_buckets=n_b,
            use_sign=self.use_sign,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )
        W_recon = hashed.get_weight().float()
        W_norm = torch.norm(W_target, p="fro").item()
        err = float(torch.norm(W_target - W_recon, p="fro").item() / max(W_norm, 1e-12))
        return hashed, err


@COMPRESSOR_REGISTRY.register("hashed_net")
def hashed_net_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[Union[int, float]] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for HashedNet compressor."""
    compressor = HashedNetCompressor(
        model,
        rank=rank,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = ["HashedNetCompressor", "hashed_net_factory"]
