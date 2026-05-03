"""GreGoDec: low‑rank + sparse decomposition via randomized SVD. Registered as 'gregodec'."""

import torch
import numpy as np
from sklearn.decomposition import randomized_svd
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("gregodec")
class GreGoDecCompressor(BaseCompressor):
    def __init__(self, model, rank=64, sparsity=0.1, max_iter=10, **kwargs):
        super().__init__(model, name="gregodec", **kwargs)
        self.rank = rank
        self.sparsity = sparsity
        self.max_iter = max_iter

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                U, S, Vt = randomized_svd(W, n_components=self.rank)
                L = U @ np.diag(S) @ Vt
                S_mat = np.zeros_like(W)
                for _ in range(self.max_iter):
                    residual = W - L
                    threshold = np.percentile(np.abs(residual), 100*(1 - self.sparsity))
                    S_mat = np.where(np.abs(residual) > threshold, residual, 0)
                    U, S_vals, Vt = randomized_svd(W - S_mat, n_components=self.rank)
                    L = U @ np.diag(S_vals) @ Vt
                module.weight.data = torch.tensor(L, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()