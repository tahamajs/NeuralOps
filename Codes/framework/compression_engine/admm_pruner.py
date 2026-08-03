"""ADMM pruner (soft thresholding). Registered as 'admm_pruner'."""

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class ADMMPruner(BaseCompressor):
    def __init__(self, model, sparsity=0.5, rho=0.1, max_iter=5, **kwargs):
        super().__init__(model, name="admm_pruner", **kwargs)
        self.sparsity = sparsity
        self.rho = rho
        self.max_iter = max_iter

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                Z = W.copy()
                U = np.zeros_like(W)
                for _ in range(self.max_iter):
                    Z_plus_U = Z + U
                    threshold = np.percentile(np.abs(Z_plus_U), 100 * self.sparsity)
                    Z = np.where(np.abs(Z_plus_U) > threshold, Z_plus_U, 0)
                    U = U + (W - Z)
                module.weight.data = torch.tensor(
                    Z, dtype=module.weight.dtype, device=module.weight.device
                )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("admm_pruner")
def admm_pruner_factory(model, comp_config) -> Any:
    """Admm pruner factory."""
    sparsity = getattr(comp_config, "admm_pruner_sparsity", 0.5)
    rho = getattr(comp_config, "admm_pruner_rho", 0.1)
    max_iter = getattr(comp_config, "admm_pruner_max_iter", 5)
    compressor = ADMMPruner(model, sparsity=sparsity, rho=rho, max_iter=max_iter)
    compressed_model, _ = compressor.compress()
    return compressed_model
