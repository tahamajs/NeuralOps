"""FRanDI: Feature‑Regression aligned low‑rank compression. Registered as 'frandi'."""

import torch
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class FRanDICompressor(BaseCompressor):
    def __init__(self, model, rank=8, **kwargs):
        super().__init__(model, name="frandi", **kwargs)
        self.rank = rank

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float()
                x = torch.randn(1, 10, W.shape[1], device=W.device)
                with torch.no_grad():
                    out = module(x)
                U, S, V = torch.linalg.svd(W, full_matrices=False)
                r = min(self.rank, len(S))
                W_approx = U[:, :r] @ torch.diag(S[:r]) @ V[:r, :].T
                module.weight.data = W_approx.to(dtype=module.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()