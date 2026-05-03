"""ADMM pruner (soft thresholding). Registered as 'admm_pruner'."""

import torch
import numpy as np
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("admm_pruner")
class ADMMPruner(BaseCompressor):
    def __init__(self, model, sparsity=0.5, rho=0.1, max_iter=5, **kwargs):
        super().__init__(model, name="admm_pruner", **kwargs)
        self.sparsity = sparsity
        self.rho = rho
        self.max_iter = max_iter

    def compress(self):
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
                module.weight.data = torch.tensor(Z, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()