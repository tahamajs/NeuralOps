"""SAES-SVD: Joint intra- and inter-layer error compensation. Registered as 'saes'."""

import torch
import torch.nn as nn
import numpy as np
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
class SAESCompressor(BaseCompressor):
    def __init__(self, model, rank=64, correcting_steps=3, **kwargs):
        super().__init__(model, name="saes", **kwargs)
        self.rank = rank
        self.correcting_steps = correcting_steps

    def compress(self):
        # Placeholder: apply SVD to each linear layer with optional error compensation
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                U, S, Vh = np.linalg.svd(W, full_matrices=False)
                r = min(self.rank, min(W.shape))
                W_approx = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
                module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    