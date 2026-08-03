"""FRanDI: Feature‑Regression aligned low‑rank compression. Registered as 'frandi'."""
from typing import Any

import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class FRanDICompressor(BaseCompressor):
    """
    Compresses linear layers using truncated SVD.
    If a tokenizer is attached to the model, it can optionally use calibration
    activations to align the low‑rank approximation (future work placeholder).
    """

    def __init__(self, model, rank=8, energy_threshold=None, **kwargs):
        super().__init__(model, name="frandi", **kwargs)
        self.rank = rank
        self.energy_threshold = energy_threshold

    def _select_rank(self, S) -> Any:
        """Choose rank based on energy threshold if provided, else fixed rank."""
        if self.energy_threshold is None:
            return min(self.rank, len(S))
        total_energy = (S**2).sum()
        cum_energy = torch.cumsum(S**2, dim=0) / total_energy
        # find first index where cumulative energy >= threshold
        idx = torch.where(cum_energy >= self.energy_threshold)[0]
        if len(idx) > 0:
            rank = idx[0].item() + 1
        else:
            rank = 1
        return min(rank, len(S))

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = self._select_rank(S)
                W_approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
                module.weight.data = W_approx.to(dtype=module.weight.dtype)

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("frandi")
def frandi_factory(model, comp_config) -> Any:
    """Frandi factory."""
    rank = getattr(comp_config, "frandi_rank", 8)
    energy = getattr(comp_config, "frandi_energy", None)
    compressor = FRanDICompressor(model, rank=rank, energy_threshold=energy)
    compressed_model, _ = compressor.compress()
    return compressed_model
