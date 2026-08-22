"""
LSC: Low-Rank Spectral & Fourier-Domain Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'lsc'.

Mathematical Formulation:
1. 2D Discrete Fourier Transform:
   $$\\widehat{W} = \\mathcal{F}_{2D}(W) \\in \\mathbb{C}^{d_{out} \\times d_{in}}$$
2. Spectral Magnitude Spectrum:
   $$M_{ij} = |\\widehat{W}_{ij}| = \\sqrt{\\operatorname{Re}(\\widehat{W}_{ij})^2 + \\operatorname{Im}(\\widehat{W}_{ij})^2}$$
3. Spectral Coefficient Selection:
   $$\\mathcal{M}_{spectral} = \\operatorname{Top-}k(M), \\quad k = \\min(r \\cdot r, d_{out} \\cdot d_{in})$$
4. Inverse 2D Fourier Reconstruction:
   $$W_{lsc} = \\operatorname{Re}\\left( \\mathcal{F}_{2D}^{-1}(\\widehat{W} \\odot \\mathcal{M}_{spectral}) \\right)$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class LSCCompressor(BaseCompressor):
    """
    LSC Compressor: 2D Fourier spectral low-rank and frequency filtering compressor.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 32,
        energy_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="lsc", **kwargs)
        self.rank = max(1, int(rank))
        self.energy_ratio = energy_ratio
        self.stats: Dict[str, Any] = {}

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes 2D Fourier spectral compression across all Linear layers."""
        compressed_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                d_out, d_in = W.shape
                device = W.device
                dtype = module.weight.dtype

                # 2D FFT
                W_fft = torch.fft.fft2(W)
                magnitude = torch.abs(W_fft)

                total_coeffs = d_out * d_in
                if self.energy_ratio is not None:
                    # Select coefficients by cumulative spectral energy
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
                module.weight.data = W_comp.to(dtype=dtype, device=device)
                compressed_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": compressed_layers_count,
            "rank": self.rank,
            "energy_ratio": self.energy_ratio,
            "method": "lsc_fourier_spectral",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("lsc")
def lsc_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for LSC compressor."""
    rank = getattr(comp_config, "lsc_rank", getattr(comp_config, "rank", 32)) if comp_config else kwargs.get("rank", 32)
    energy_ratio = getattr(comp_config, "lsc_energy_ratio", getattr(comp_config, "energy_ratio", None)) if comp_config else kwargs.get("energy_ratio", None)

    compressor = LSCCompressor(model, rank=rank, energy_ratio=energy_ratio, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
