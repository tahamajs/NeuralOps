"""MGIE-SVD: Entropy-driven rank selection. Registered as 'mgie'."""

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class MGIECompressor(BaseCompressor):
    def __init__(self, model, **kwargs):
        super().__init__(model, name="mgie", **kwargs)

    def _compress_layer(self, W):
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        p = S / S.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-8))
        max_rank = min(W.shape)
        rank = max(1, int(entropy * max_rank / (torch.log(torch.tensor(max_rank)) + 1e-8)))
        rank = min(rank, max_rank)
        return U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = self._compress_layer(module.weight.data)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    


@COMPRESSOR_REGISTRY.register("mgie")
def mgie_factory(model, comp_config):
    compressor = MGIECompressor(model)   # takes only model
    compressed_model, _ = compressor.compress()
    return compressed_model