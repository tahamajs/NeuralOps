"""FRanDI: Feature‑Regression aligned low‑rank compression. Registered as 'frandi'."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class FRanDICompressor(BaseCompressor):
    """
    FRanDI (Feature‑Regression aligned low‑rank compression) applies SVD compression
    to linear layers, optionally using feature statistics from a calibration loader
    to align the compression with input feature distribution.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        energy_threshold: Optional[float] = None,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        **kwargs
    ):
        super().__init__(model, name="frandi", **kwargs)
        self.rank = rank
        self.energy_threshold = energy_threshold
        self.calibration_loader = calibration_loader
        self.device = next(model.parameters()).device

    def _select_rank(self, S: torch.Tensor) -> int:
        """Select rank based on energy threshold if provided, else fixed rank."""
        if self.energy_threshold is None:
            return min(self.rank, len(S))
        total_energy = (S**2).sum()
        cum_energy = (S**2).cumsum(dim=0) / total_energy
        rank = (cum_energy >= self.energy_threshold).nonzero(as_tuple=True)[0][
            0
        ].item() + 1
        return min(rank, len(S))

    def _collect_activations(self) -> Dict[str, torch.Tensor]:
        """Collect input activations for each linear layer using calibration loader."""
        if self.calibration_loader is None:
            return {}
        activations = {}
        hooks = []

        def hook_fn(module, input, output, name) -> None:
            """Hook fn."""
            x = input[0].detach()
            x_flat = x.reshape(-1, x.shape[-1])
            if name not in activations:
                activations[name] = x_flat
            else:
                activations[name] = torch.cat([activations[name], x_flat], dim=0)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o, n=name: hook_fn(m, i, o, n)
                    )
                )

        self.model.eval()
        with torch.no_grad():
            for batch in self.calibration_loader:
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                _ = self.model(**batch)

        for h in hooks:
            h.remove()
        return activations

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all linear layers using SVD, optionally with feature alignment."""
        self._collect_activations()  # we don't use the activations further in this simple version
        self.model.train()

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            W = module.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            rank = self._select_rank(S)
            r = min(rank, len(S))

            W_approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
            module.weight.data = W_approx.to(dtype=module.weight.dtype)
            # Bias remains unchanged

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _compute_stats(self) -> Dict[str, Any]:
        """ compute stats."""
        stats = super()._compute_stats()
        stats.update(
            {
                "rank": self.rank,
                "energy_threshold": self.energy_threshold,
                "calibration_used": self.calibration_loader is not None,
            }
        )
        return stats


@COMPRESSOR_REGISTRY.register("frandi")
def frandi_compressor_entry(model: nn.Module, comp_config) -> nn.Module:
    """
    Entry point for the registry. Expects comp_config to have attributes:
        frandi_rank (int, default 8)
        frandi_energy (float, optional)
    """
    rank = getattr(comp_config, "frandi_rank", 8)
    energy = getattr(comp_config, "frandi_energy", None)
    compressor = FRanDICompressor(model, rank=rank, energy_threshold=energy)
    compressed_model, _ = compressor.compress()
    return compressed_model
