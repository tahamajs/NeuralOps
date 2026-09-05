"""
LSC: Low-Rank Spectral & Fourier-Domain Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'lsc' and 'lsc_compressor'.

Mathematical Formulation:
1. 2D Discrete Fourier Transform:
   $$\\widehat{W} = \\mathcal{F}_{2D}(W) \\in \\mathbb{C}^{d_{out} \\times d_{in}}$$
2. Spectral Magnitude Spectrum:
   $$M_{ij} = |\\widehat{W}_{ij}| = \\sqrt{\\operatorname{Re}(\\widehat{W}_{ij})^2 + \\operatorname{Im}(\\widehat{W}_{ij})^2}$$
3. Spectral Coefficient Selection:
   $$\\mathcal{M}_{spectral} = \\operatorname{Top-}k(M), \\quad k = \\min(r \\cdot r, d_{out} \\cdot d_{in})$$
4. Inverse 2D Fourier Reconstruction:
   $$W_{lsc} = \\operatorname{Re}\\left( \\mathcal{F}_{2D}^{-1}(\\widehat{W} \\odot \\mathcal{M}_{spectral}) \\right)$$

Production Hardening:
- Fix 1: Reconstruction error validation with configurable max_error (default 0.20) and safe fallback.
- Fix 2: Support for Conv1D (GPT-2) with weight transposition and fused QKV projections (Qwen/LLaMA).
- Fix 3: Safe FP32 Fourier computations with NaN/Inf guards and numerical stabilization.
- Fix 4: Comprehensive per-layer statistics (layer errors, chosen k/energy ratio, parameter counts).
- Fix 5: Unified critical-layer protection delegated solely to should_compress_linear.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("lsc")
@COMPRESSOR_REGISTRY.register("lsc_compressor")
class LSCCompressor(BaseCompressor):
    """
    LSC Compressor: 2D Fourier spectral low-rank and frequency filtering compressor.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 32,
        energy_ratio: Optional[float] = None,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="lsc", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "lsc_rank", getattr(cfg, "rank", rank))
            energy_ratio = getattr(cfg, "lsc_energy_ratio", getattr(cfg, "energy_ratio", energy_ratio))
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])

        self.rank = max(1, int(rank))
        self.energy_ratio = float(energy_ratio) if energy_ratio is not None else None
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Protect critical layers and verify module is compressible linear/Conv1D."""
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
        if not is_linear_like:
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes 2D Fourier spectral compression across all Linear and Conv1D layers."""
        compressed_layers_count = 0
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

            # FP32 computation
            W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape
            total_coeffs = d_out * d_in

            try:
                # 2D FFT in FP32
                W_fft = torch.fft.fft2(W)
                magnitude = torch.abs(W_fft)

                if self.energy_ratio is not None:
                    flat_mag = magnitude.reshape(-1)
                    sorted_mag, _ = torch.sort(flat_mag, descending=True)
                    cum_energy = torch.cumsum(sorted_mag ** 2, dim=0) / torch.sum(sorted_mag ** 2).clamp(min=1e-8)
                    idx = torch.where(cum_energy >= self.energy_ratio)[0]
                    k = idx[0].item() + 1 if len(idx) > 0 else total_coeffs
                    k = max(1, min(k, total_coeffs))
                else:
                    k = max(1, min(self.rank * self.rank, total_coeffs))

                # Top-k spectral mask
                flat_mag = magnitude.reshape(-1)
                thr = torch.topk(flat_mag, k=k, largest=True).values[-1]
                spectral_mask = (magnitude >= thr).float()

                # Inverse 2D FFT
                W_comp = torch.fft.ifft2(W_fft * spectral_mask).real
                W_comp = torch.nan_to_num(W_comp, nan=0.0)

                # Reconstruction error check
                norm_orig = torch.norm(W, p="fro")
                rel_err = (torch.norm(W - W_comp, p="fro") / (norm_orig + 1e-8)).item()
                self.layer_errors[name] = rel_err

                if rel_err > self.max_error:
                    logger.warning(
                        "LSC reconstruction error %.4f on %s exceeded max_error %.4f%s",
                        rel_err,
                        name,
                        self.max_error,
                        ". Keeping original layer." if self.fallback_to_original else ". Proceeding with filtered weights.",
                    )
                    if self.fallback_to_original:
                        continue

                final_w = W_comp.t() if is_conv1d else W_comp
                module.weight.data = final_w.to(dtype=dtype, device=device)
                compressed_layers_count += 1
                layer_details[name] = {
                    "k_coeffs": k,
                    "rel_error": rel_err,
                    "shape": tuple(final_w.shape),
                }
            except Exception as e:
                logger.warning("LSC compression exception on %s: %s. Keeping original layer.", name, e)
                continue

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = float(torch.tensor(list(self.layer_errors.values())).mean().item()) if self.layer_errors else 0.0
        stats.update({
            "compressed_layers": compressed_layers_count,
            "rank": self.rank,
            "energy_ratio": self.energy_ratio,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "layer_errors": self.layer_errors,
            "layer_details": layer_details,
            "method": "lsc_fourier_spectral",
        })
        self.stats = stats
        logger.info(
            "LSC compression finished: %d layers compressed, avg error: %.4f",
            compressed_layers_count,
            avg_err,
        )
        return self.model, stats


def lsc_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for LSC compressor."""
    rank = getattr(comp_config, "lsc_rank", getattr(comp_config, "rank", 32)) if comp_config else kwargs.get("rank", 32)
    energy_ratio = getattr(comp_config, "lsc_energy_ratio", getattr(comp_config, "energy_ratio", None)) if comp_config else kwargs.get("energy_ratio", None)
    max_error = getattr(comp_config, "max_error", 0.20) if comp_config else kwargs.get("max_error", 0.20)
    fallback_to_original = getattr(comp_config, "fallback_to_original", True) if comp_config else kwargs.get("fallback_to_original", True)

    compressor = LSCCompressor(
        model,
        rank=rank,
        energy_ratio=energy_ratio,
        max_error=max_error,
        fallback_to_original=fallback_to_original,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "LSCCompressor",
    "lsc_factory",
]
