"""Swift-SVD: closed‑form SVD with rank ratio. Registered as 'swift_svd'."""

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@COMPRESSOR_REGISTRY.register("swift_svd")
class SwiftSVDCompressor(BaseCompressor):
    def __init__(self, model, rank_ratio=0.5, **kwargs):
        super().__init__(model, name="swift_svd", **kwargs)
        self.rank_ratio = rank_ratio

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                rank = max(1, int(min(W.shape) * self.rank_ratio))
                U_r = U[:, :rank]
                S_r = S[:rank]
                Vt_r = Vt[:rank, :]
                module.weight.data = (U_r * S_r) @ Vt_r
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()