"""Robust PCA: decompose W = L + S, keep L as low‑rank. Registered as 'rpca'."""

import torch
import numpy as np
from ..core.base import BaseCompressor
from ..core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("rpca")
class RPCACompressor(BaseCompressor):
    def __init__(self, model, rank=64, mu=1e-3, max_iter=5, **kwargs):
        super().__init__(model, name="rpca", **kwargs)
        self.rank = rank
        self.mu = mu
        self.max_iter = max_iter

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                L = W.copy()
                S = np.zeros_like(W)
                for _ in range(self.max_iter):
                    # Low‑rank update
                    U, S_svd, Vt = np.linalg.svd(L, full_matrices=False)
                    r = min(self.rank, len(S_svd))
                    L = U[:, :r] @ np.diag(S_svd[:r]) @ Vt[:r, :]
                    # Sparse update (soft‑threshold)
                    S = np.sign(W - L) * np.maximum(np.abs(W - L) - self.mu, 0)
                    L = W - S
                module.weight.data = torch.tensor(L, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()