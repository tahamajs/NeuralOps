"""SAES‑SVD v2: error suppression with fixed rank ratio. Registered as 'saes_svd_v2'."""

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class SAESSVDCompressorV2(BaseCompressor):
    def __init__(self, model, rank_ratio=0.7, **kwargs):
        super().__init__(model, name="saes_svd_v2", **kwargs)
        self.rank_ratio = rank_ratio

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                rank = max(1, int(min(W.shape) * self.rank_ratio))
                W_approx = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
                module.weight.data = W_approx.to(dtype=module.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("saes_svd_v2")
def saes_svd_v2_factory(model, comp_config):
    rank_ratio = getattr(comp_config, "rank_ratio", 0.7)
    compressor = SAESSVDCompressorV2(model, rank_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model
