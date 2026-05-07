"""KVT Weight Compressor: PCA + quantisation on weight rows. Registered as 'kvt_weight'."""

import torch
import numpy as np
from sklearn.decomposition import PCA
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@COMPRESSOR_REGISTRY.register("kvt_weight")
class KVTWeightCompressor(BaseCompressor):
    def __init__(self, model, bits=4, keep_eig=0.95, **kwargs):
        super().__init__(model, name="kvt_weight", **kwargs)
        self.bits = bits
        self.keep_eig = keep_eig

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                pca = PCA(n_components=self.keep_eig)
                W_pca = pca.fit_transform(W.T).T  # transform along the feature axis? We'll apply on rows
                min_v, max_v = W_pca.min(), W_pca.max()
                if max_v == min_v:
                    max_v = min_v + 1e-6
                scale = (max_v - min_v) / (2**self.bits - 1)
                W_q = np.round((W_pca - min_v) / scale) * scale + min_v
                # Inverse PCA
                W_approx = W_q @ pca.components_
                module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()