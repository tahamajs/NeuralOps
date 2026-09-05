"""
KVTC: KV Cache Transform Coding - PCA decorrelation + adaptive quantization
      + entropy coding for compact KV cache storage.

Paper: "KV Cache Transform Coding for Compact Storage in LLM Inference"
       Staniszewski & Lancucki, ICLR 2026.
Registered as 'kvtc' and 'kvtc_quantizer'.

Production Hardening:
- Fix 1: Reconstruction error validation with configurable max_error (default 0.20) and safe fallback.
- Fix 2: Support for Conv1D (GPT-2) and fused QKV projections (Qwen/LLaMA).
- Fix 3: FP32 computations with PCA rank bounded to [1, min(d_out, d_in)] and fallback to original on failure.
- Fix 4: Comprehensive per-layer statistics (ranks, errors, entropy bits).
- Fix 5: Unified critical-layer protection via should_compress_linear.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class SimpleEntropyCoder:
    """Lightweight entropy coder using empirical symbol frequency estimation."""

    @staticmethod
    def compute_entropy(tensor: torch.Tensor) -> float:
        """Compute empirical Shannon entropy in bits per symbol."""
        if tensor.numel() == 0:
            return 0.0
        unique_vals, counts = torch.unique(tensor, return_counts=True)
        probs = counts.float() / float(tensor.numel())
        entropy = -torch.sum(probs * torch.log2(torch.clamp(probs, min=1e-12))).item()
        return float(entropy)

    @staticmethod
    def compute_compression_ratio(tensor: torch.Tensor, uncompressed_bits: int = 16) -> float:
        """Estimate theoretical compression ratio relative to FP16/FP32 baseline."""
        entropy = SimpleEntropyCoder.compute_entropy(tensor)
        return float(uncompressed_bits / max(entropy, 1e-4))


class KVTCStore:
    """
    Runtime KV Cache Transform Coding engine.
    Applies PCA decorrelation across head feature dimensions and adaptive quantization,
    supporting dynamic sequence lengths across prefill and decoding phases.
    """

    def __init__(self, bits: int = 4, pca_energy: float = 0.95) -> None:
        self.bits = int(bits)
        self.pca_energy = float(pca_energy)
        self.pca_basis_k: Optional[torch.Tensor] = None
        self.pca_basis_v: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset PCA basis to prevent cross-sequence memory leaks."""
        self.pca_basis_k = None
        self.pca_basis_v = None

    @torch.no_grad()
    def calibrate_basis(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Calibrate PCA projection basis from sample K/V tensor (B, H, N, D)."""
        k_flat = k.detach().float().reshape(-1, k.shape[-1])
        v_flat = v.detach().float().reshape(-1, v.shape[-1])

        # SVD on centered channels in FP32
        _, _, Vt_k = torch.linalg.svd(k_flat - k_flat.mean(dim=0, keepdim=True), full_matrices=False)
        _, _, Vt_v = torch.linalg.svd(v_flat - v_flat.mean(dim=0, keepdim=True), full_matrices=False)

        self.pca_basis_k = Vt_k.t().to(device=k.device, dtype=k.dtype)
        self.pca_basis_v = Vt_v.t().to(device=v.device, dtype=v.dtype)

    @torch.no_grad()
    def compress_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Transform and quantize dynamic KV cache tensors.
        Returns reconstructed (k_hat, v_hat) and coding statistics.
        """
        orig_shape = k.shape
        D = orig_shape[-1]

        if self.pca_basis_k is None or self.pca_basis_v is None:
            self.calibrate_basis(k, v)

        k_flat = k.reshape(-1, D)
        v_flat = v.reshape(-1, D)

        k_trans = torch.matmul(k_flat, self.pca_basis_k)
        v_trans = torch.matmul(v_flat, self.pca_basis_v)

        qmax = (2 ** (self.bits - 1)) - 1
        scale_k = torch.clamp(k_trans.abs().max() / float(qmax), min=1e-6)
        scale_v = torch.clamp(v_trans.abs().max() / float(qmax), min=1e-6)

        q_k = torch.clamp(torch.round(k_trans / scale_k), -qmax, qmax)
        q_v = torch.clamp(torch.round(v_trans / scale_v), -qmax, qmax)

        h_k = SimpleEntropyCoder.compute_entropy(q_k)
        h_v = SimpleEntropyCoder.compute_entropy(q_v)

        k_hat = torch.matmul(q_k * scale_k, self.pca_basis_k.t()).reshape(orig_shape)
        v_hat = torch.matmul(q_v * scale_v, self.pca_basis_v.t()).reshape(orig_shape)

        stats = {
            "entropy_k_bits": h_k,
            "entropy_v_bits": h_v,
            "mean_entropy_bits": (h_k + h_v) / 2.0,
            "theoretical_compression_ratio": 16.0 / max((h_k + h_v) / 2.0, 1e-4),
        }
        return k_hat, v_hat, stats


@COMPRESSOR_REGISTRY.register("kvtc")
@COMPRESSOR_REGISTRY.register("kvtc_quantizer")
class KVTCCompressor(BaseCompressor):
    """
    KVTC: KV Cache Transform Coding compressor.
    Compresses linear projection weights using PCA decorrelation,
    adaptive quantization, and entropy coding estimation.
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        pca_energy: float = 0.95,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="kvtc", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            bits = getattr(cfg, "kvtc_bits", getattr(cfg, "bits", bits))
            pca_energy = getattr(cfg, "kvtc_pca_energy", getattr(cfg, "pca_energy", pca_energy))
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])

        self.bits = int(bits)
        self.pca_energy = float(pca_energy)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
        if not is_linear_like:
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using PCA decorrelation and adaptive quantization."""
        total_entropy = 0.0
        layers_compressed = 0
        self.layer_errors = {}
        layer_details = {}

        for name, module in self.model.named_modules():
            if not self._is_target_module(name, module):
                continue

            raw_w = module.weight.data
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            device = raw_w.device
            dtype = raw_w.dtype

            W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape
            min_dim = min(d_out, d_in)

            try:
                # Step 1: PCA decorrelation via SVD in FP32
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                total_energy = (S ** 2).sum()
                if total_energy > 0:
                    cum_energy = torch.cumsum(S ** 2, dim=0) / total_energy
                    rank = int((cum_energy < self.pca_energy).sum().item()) + 1
                else:
                    rank = min_dim

                # Ensure PCA rank is at least 1 and not greater than original dimension
                rank = max(1, min(rank, min_dim))

                # Reconstruct low-rank approximation
                U_r = U[:, :rank]
                S_r = S[:rank]
                Vt_r = Vt[:rank, :]
                W_lowrank = (U_r * S_r.unsqueeze(0)) @ Vt_r

                # Step 2: Adaptive quantization
                min_val, max_val = W_lowrank.min(), W_lowrank.max()
                if max_val == min_val:
                    max_val = min_val + 1e-6
                scale = (max_val - min_val) / float((2 ** self.bits) - 1)
                q_vals = torch.round((W_lowrank - min_val) / scale)
                W_quant = q_vals * scale + min_val
                W_quant = torch.nan_to_num(W_quant, nan=0.0)

                # Reconstruction error check
                norm_orig = torch.norm(W, p="fro")
                rel_err = (torch.norm(W - W_quant, p="fro") / (norm_orig + 1e-8)).item()
                self.layer_errors[name] = rel_err

                if rel_err > self.max_error:
                    logger.warning(
                        "KVTC reconstruction error %.4f on %s exceeded max_error %.4f%s",
                        rel_err,
                        name,
                        self.max_error,
                        ". Keeping original layer." if self.fallback_to_original else ". Proceeding with quantized weight.",
                    )
                    if self.fallback_to_original:
                        continue

                # Step 3: Empirical entropy calculation
                entropy = SimpleEntropyCoder.compute_entropy(q_vals)
                total_entropy += entropy

                final_w = W_quant.t() if is_conv1d else W_quant
                module.weight.data = final_w.to(dtype=dtype, device=device)
                layers_compressed += 1
                layer_details[name] = {
                    "rank": rank,
                    "rel_error": rel_err,
                    "entropy": entropy,
                    "shape": tuple(final_w.shape),
                }
            except Exception as e:
                logger.warning("KVTC compression exception on %s: %s. Keeping original layer.", name, e)
                continue

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_entropy = total_entropy / max(layers_compressed, 1)
        avg_err = float(torch.tensor(list(self.layer_errors.values())).mean().item()) if self.layer_errors else 0.0
        stats.update({
            "compressed_layers": layers_compressed,
            "bits": self.bits,
            "pca_energy": self.pca_energy,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_entropy_bits": round(avg_entropy, 2),
            "avg_reconstruction_error": avg_err,
            "theoretical_entropy_compression_ratio": round(16.0 / max(avg_entropy, 1e-4), 2),
            "theoretical_vram_saving": f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16",
            "layer_errors": self.layer_errors,
            "layer_details": layer_details,
        })
        self.stats = stats
        logger.info(
            "KVTC compression complete: %d layers compressed, avg error: %.4f, avg entropy: %.2f bits",
            layers_compressed,
            avg_err,
            avg_entropy,
        )
        return self.model, stats


def kvtc_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """KVTC compressor factory function."""
    bits = kwargs.pop("bits", getattr(comp_config, "kvtc_bits", getattr(comp_config, "bits", 4)) if comp_config else 4)
    pca_energy = kwargs.pop("pca_energy", getattr(comp_config, "kvtc_pca_energy", getattr(comp_config, "pca_energy", 0.95)) if comp_config else 0.95)
    max_error = kwargs.pop("max_error", getattr(comp_config, "max_error", 0.20) if comp_config else 0.20)
    fallback_to_original = kwargs.pop("fallback_to_original", getattr(comp_config, "fallback_to_original", True) if comp_config else True)

    compressor = KVTCCompressor(
        model,
        bits=bits,
        pca_energy=pca_energy,
        max_error=max_error,
        fallback_to_original=fallback_to_original,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "KVTCCompressor",
    "KVTCStore",
    "SimpleEntropyCoder",
    "kvtc_factory",
]
