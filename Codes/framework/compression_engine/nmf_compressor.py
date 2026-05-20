"""NMF: Non‑negative Matrix Factorization approximation. Registered as 'nmf'."""

import torch
import numpy as np
from sklearn.decomposition import NMF as skNMF
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class NMFCompressor(BaseCompressor):
    def __init__(self, model, rank=64, **kwargs):
        super().__init__(model, name="nmf", **kwargs)
        self.rank = rank

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                min_val = W.min()
                W_shifted = W - min_val + 1e-5
                nmf = skNMF(n_components=self.rank, init='random', max_iter=200)
                H = nmf.fit_transform(W_shifted)   # (out_features, rank)
                W_comp = H @ nmf.components_ + min_val - 1e-5
                module.weight.data = torch.tensor(W_comp, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    

@COMPRESSOR_REGISTRY.register("nmf")
def nmf_factory(model, comp_config):
    rank = getattr(comp_config, 'nmf_rank', 64)
    compressor = NMFCompressor(model, rank=rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
