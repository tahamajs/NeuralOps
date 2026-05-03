"""LSC: Fourier‑based low‑rank compression. Registered as 'lsc'."""

import torch
import numpy as np
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("lsc")
class LSCCompressor(BaseCompressor):
    def __init__(self, model, rank=32, **kwargs):
        super().__init__(model, name="lsc", **kwargs)
        self.rank = rank

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                W_fft = np.fft.fft2(W)
                magnitude = np.abs(W_fft)
                flat_idx = np.argsort(magnitude.ravel())[::-1][:self.rank*self.rank]
                mask = np.zeros_like(W_fft)
                mask.flat[flat_idx] = 1
                W_comp = np.fft.ifft2(W_fft * mask).real
                module.weight.data = torch.tensor(W_comp, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()