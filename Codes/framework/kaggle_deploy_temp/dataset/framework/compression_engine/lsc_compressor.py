"""LSC: Fourier‑based low‑rank compression. Registered as 'lsc'."""
from typing import Any

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class LSCCompressor(BaseCompressor):
    def __init__(self, model, rank=32, **kwargs):
        super().__init__(model, name="lsc", **kwargs)
        self.rank = rank

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                W_fft = np.fft.fft2(W)
                magnitude = np.abs(W_fft)
                flat_idx = np.argsort(magnitude.ravel())[::-1][: self.rank * self.rank]
                mask = np.zeros_like(W_fft)
                mask.flat[flat_idx] = 1
                W_comp = np.fft.ifft2(W_fft * mask).real
                module.weight.data = torch.tensor(
                    W_comp, dtype=module.weight.dtype, device=module.weight.device
                )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("lsc")
def lsc_factory(model, comp_config) -> Any:
    """Lsc factory."""
    rank = getattr(comp_config, "lsc_rank", 32)
    compressor = LSCCompressor(model, rank=rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
