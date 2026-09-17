"""
ADMM Compressor: Alternating Direction Method of Multipliers for Joint Pruning and Quantization.
Registered in COMPRESSOR_REGISTRY as 'admm' and 'admm_compressor'. Production hardened with
reconstruction validation, Conv1D support, FP32 numerical stability, safe fallback,
LoRALinear preservation, and attention-only targeting.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    apply_block_mask,
)
from framework.utils.compat import should_compress_linear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely extract configuration attribute from None, dict, or object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("admm")
@COMPRESSOR_REGISTRY.register("admm_compressor")
class ADMMCompressor(BaseCompressor):
    """
    ADMM-based Joint Structured Pruning and Non-Uniform / Uniform Quantization Engine.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original module if optimization produces NaN/Inf or error exceeds threshold.
    - Fallback layer freezing: all parameters of fallback layers have requires_grad=False.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition to [out_features, in_features].
    - Robust LoRALinear support: operates on base_layer.weight, preserves LoRA adapters intact,
      retaining requires_grad=True for LoRA adapters during recovery when target_attention_only is True.
    - Attention-only targeting option: selectively compresses attention projections and skips FFN/MLP layers.
    - Strict FP32 computation for ADMM auxiliary and dual variables (W, Z, Q, U1, U2).
    - Monotonic error tracker: preserves the best iterate across ADMM steps against divergence.
    - Shape verification ensuring reconstructed weights match original layer dimensions.
    - Per-layer stats, actual sparsity (fraction of zeros), and effective compression rate reporting.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        quant_bits: int = 4,
        rho1: float = 0.1,
        rho2: float = 0.1,
        max_iter: int = 8,
        structured_24: bool = True,
        block_size: Tuple[int, int] = (4, 4),
        max_error: float = 0.45,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
        target_modules: Optional[List[str]] = None,
        comp_config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="admm", **kwargs)
        cfg = comp_config or kwargs.get("comp_config", kwargs.get("cfg", getattr(model, "config", None)))

        self.sparsity = max(0.0, min(float(_cfg_get(cfg, "admm_sparsity", _cfg_get(cfg, "sparsity", sparsity))), 0.95))
        self.quant_bits = int(_cfg_get(cfg, "admm_quant_bits", _cfg_get(cfg, "quant_bits", quant_bits)))
        self.rho1 = float(_cfg_get(cfg, "admm_rho1", _cfg_get(cfg, "rho1", rho1)))
        self.rho2 = float(_cfg_get(cfg, "admm_rho2", _cfg_get(cfg, "rho2", rho2)))
        self.max_iter = int(_cfg_get(cfg, "admm_max_iter", _cfg_get(cfg, "max_iter", max_iter)))
        self.structured_24 = bool(_cfg_get(cfg, "admm_structured_24", _cfg_get(cfg, "structured_24", structured_24)))
        self.block_size = block_size
        self.max_error = float(_cfg_get(cfg, "admm_max_error", _cfg_get(cfg, "max_error", max_error)))
        self.fallback_to_original = bool(
            _cfg_get(cfg, "admm_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.target_attention_only = bool(
            _cfg_get(cfg, "admm_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.target_modules = target_modules
        self.comp_config = cfg

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_sparsities: Dict[str, float] = {}
        self.layer_actual_sparsities: Dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        # Do not target inner submodules of LoRA or wrappers directly
        if any(name.endswith(sfx) for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout", ".linear"]):
            return False

        target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        # Extract weight and ensure it is strictly a 2D matrix (filters out 3D/4D convs like nn.Conv1d)
        weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
        if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
            return False

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
            or hasattr(module, "nf")
            or hasattr(target_mod, "nf")
            or "loralinear" in cls_name
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False

        name_lower = name.lower()
        # Skip classification / output heads
        if any(skip in name_lower for skip in ["lm_head", "classifier", "head", "output_layer", "score", "pooler", "pre_classifier"]):
            return False

        if self.target_attention_only:
            # Skip FFN / MLP layers
            ffn_patterns = [
                "ffn", "mlp", "lin1", "lin2", "intermediate",
                "gate_proj", "up_proj", "down_proj", "c_fc"
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            # RoBERTa layer.X.output.dense is FFN, while attention.output.dense is attention
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            # Match attention projections
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj", "query", "key", "value",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "out_proj"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _quantize_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Projects continuous tensor onto b-bit uniform quantization grid in FP32."""
        if self.quant_bits >= 32:
            return tensor
        scale = 2.0 ** (self.quant_bits - 1) - 1.0
        max_val = torch.max(torch.abs(tensor)).clamp(min=1e-6)
        normalized = torch.clamp(tensor / max_val, -1.0, 1.0)
        quantized = torch.round(normalized * scale) / scale
        return quantized * max_val

    def _prune_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Projects continuous tensor onto target sparse constraint set S."""
        if self.structured_24:
            pruned, _ = apply_24_structured_mask(tensor, saliency=torch.abs(tensor), dim=1)
            return pruned
        elif self.block_size != (1, 1):
            pruned, _ = apply_block_mask(tensor, block_size=self.block_size, sparsity=self.sparsity)
            return pruned
        else:
            flat = tensor.abs().reshape(-1)
            k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
            thr = torch.topk(flat, k, largest=True).values[-1]
            mask = (tensor.abs() >= thr).float()
            return tensor * mask

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes ADMM joint optimization across all target linear/Conv1D modules."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_sparsities = {}
        self.layer_actual_sparsities = {}

        targets = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("ADMM: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
            target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
            cls_name = module.__class__.__name__.lower()
            target_cls = target_mod.__class__.__name__.lower()
            is_conv1d = (
                (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
                or hasattr(target_mod, "nf")
                or hasattr(module, "nf")
            )

            raw_w = target_mod.weight.data
            orig_shape = raw_w.shape
            device = raw_w.device
            dtype = raw_w.dtype

            if is_conv1d:
                W0 = raw_w.t().float()  # [out_features, in_features]
            else:
                W0 = raw_w.float()      # [out_features, in_features]

            norm_w0 = torch.norm(W0, p="fro").item()

            best_W_final = None
            best_err = float("inf")

            try:
                # Baseline projection candidate
                Z0 = self._prune_projection(W0)
                W_cand = self._quantize_projection(Z0)
                mask0 = (Z0 != 0.0).float()
                W_cand = W_cand * mask0
                err0 = float(torch.norm(W0 - W_cand, p="fro").item() / max(1e-12, norm_w0))
                best_W_final = W_cand
                best_err = err0

                # Initialize ADMM variables in FP32
                W = W0.clone()
                Z = Z0.clone()
                Q = self._quantize_projection(W0)
                U1 = torch.zeros_like(W0)
                U2 = torch.zeros_like(W0)

                for it in range(self.max_iter):
                    denom = 1.0 + self.rho1 + self.rho2
                    W = (W0 + self.rho1 * (Z - U1) + self.rho2 * (Q - U2)) / denom

                    Z = self._prune_projection(W + U1)
                    Q = self._quantize_projection(W + U2)

                    U1 = U1 + (W - Z)
                    U2 = U2 + (W - Q)

                    if (
                        torch.isnan(W).any()
                        or torch.isnan(Z).any()
                        or torch.isnan(Q).any()
                        or torch.isinf(W).any()
                        or torch.isinf(Z).any()
                        or torch.isinf(Q).any()
                    ):
                        logger.warning("ADMM: NaN/Inf detected at iter %d on %s", it, name)
                        break

                    W_curr = self._quantize_projection(Z)
                    mask = (Z != 0.0).float()
                    W_curr = W_curr * mask
                    curr_err = float(torch.norm(W0 - W_curr, p="fro").item() / max(1e-12, norm_w0))
                    if curr_err < best_err:
                        best_err = curr_err
                        best_W_final = W_curr

            except Exception as e:
                logger.warning("ADMM optimization failed on %s: %s", name, e)
                best_W_final = None
                best_err = 1.0

            W_final = best_W_final
            final_err = best_err

            self.layer_errors[name] = final_err
            self.layer_sparsities[name] = 0.5 if self.structured_24 else self.sparsity

            if (
                W_final is None
                or torch.isnan(W_final).any()
                or torch.isinf(W_final).any()
                or final_err > self.max_error
            ):
                logger.warning(
                    "ADMM reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping ADMM compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    # Freeze fallback layer so recovery fine-tuning does not alter it
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            # Assign compressed weight ensuring shape match
            if is_conv1d:
                res_w = W_final.t().to(dtype=dtype, device=device)
            else:
                res_w = W_final.to(dtype=dtype, device=device)

            if res_w.shape != orig_shape:
                logger.warning("Shape mismatch on %s: %s vs %s. Skipping.", name, res_w.shape, orig_shape)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            target_mod.weight.data.copy_(res_w)

            # Record actual zero sparsity fraction
            actual_zero_frac = float((res_w == 0).sum().item() / max(1, res_w.numel()))
            self.layer_actual_sparsities[name] = actual_zero_frac

            # Preserve LoRA adapter parameters
            has_lora = (hasattr(module, "lora_A") and module.lora_A is not None) or hasattr(module, "base_layer")
            if has_lora:
                # Base layer remains frozen
                if hasattr(target_mod, "weight") and target_mod.weight is not None:
                    target_mod.weight.requires_grad = False
                if hasattr(target_mod, "bias") and target_mod.bias is not None:
                    target_mod.bias.requires_grad = False

                # Preserve LoRA adapter parameters: do not freeze unless target_attention_only is False
                if hasattr(module, "lora_A") and module.lora_A is not None:
                    req_grad = bool(self.target_attention_only)
                    for p in module.lora_A.parameters():
                        p.requires_grad = req_grad
                if hasattr(module, "lora_B") and module.lora_B is not None:
                    req_grad = bool(self.target_attention_only)
                    for p in module.lora_B.parameters():
                        p.requires_grad = req_grad

            self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        avg_actual_sparsity = (
            sum(self.layer_actual_sparsities.values()) / max(1, len(self.layer_actual_sparsities))
            if self.layer_actual_sparsities
            else 0.0
        )
        effective_storage_comp_rate = round(16.0 / max(1, self.quant_bits), 2)

        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "layer_errors": self.layer_errors,
            "layer_sparsities": self.layer_sparsities,
            "layer_actual_sparsities": self.layer_actual_sparsities,
            "actual_sparsity": avg_actual_sparsity,
            "sparsity_target": self.sparsity,
            "quant_bits": self.quant_bits,
            "structured_24": self.structured_24,
            "effective_compression_rate": effective_storage_comp_rate,
            "admm_iterations": self.max_iter,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "target_attention_only": self.target_attention_only,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "method": "admm_joint_prune_quant",
        })
        logger.info(
            "ADMM compression complete: %d compressed, %d fallback. Avg error: %.4f, Actual Sparsity: %.2f%%, Effective Comp Rate: %.2fx",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            avg_actual_sparsity * 100.0,
            effective_storage_comp_rate,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("admm")
@COMPRESSOR_REGISTRY.register("admm_compressor")
def admm_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ADMM compressor."""
    cfg = comp_config if comp_config is not None else kwargs.get("comp_config", getattr(model, "config", None))
    sparsity = _cfg_get(cfg, "admm_sparsity", _cfg_get(cfg, "sparsity", kwargs.pop("sparsity", 0.5)))
    quant_bits = _cfg_get(cfg, "admm_quant_bits", _cfg_get(cfg, "quant_bits", kwargs.pop("quant_bits", 4)))
    structured_24 = _cfg_get(cfg, "admm_structured_24", _cfg_get(cfg, "structured_24", kwargs.pop("structured_24", True)))
    max_error = _cfg_get(cfg, "admm_max_error", _cfg_get(cfg, "max_error", kwargs.pop("max_error", 0.45)))
    fallback = _cfg_get(cfg, "admm_fallback_to_original", _cfg_get(cfg, "fallback_to_original", kwargs.pop("fallback_to_original", True)))
    target_attention_only = _cfg_get(cfg, "admm_target_attention_only", _cfg_get(cfg, "target_attention_only", kwargs.pop("target_attention_only", True)))

    compressor = ADMMCompressor(
        model,
        sparsity=sparsity,
        quant_bits=quant_bits,
        structured_24=structured_24,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    setattr(compressed_model, "compression_stats", stats)
    return compressed_model


__all__ = ["ADMMCompressor", "admm_factory"]
