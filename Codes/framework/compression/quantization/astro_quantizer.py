"""AstroQuantizer: simple uniform quantisation. Registered as 'astro'."""

import torch
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("astro")
class AstroQuantizer(BaseCompressor):
    def __init__(self, model, bits=8, **kwargs):
        super().__init__(model, name="astro", **kwargs)
        self.bits = bits

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float()
                min_v, max_v = W.min(), W.max()
                if max_v == min_v:
                    continue
                scale = (max_v - min_v) / (2**self.bits - 1)
                W_q = torch.round((W - min_v) / scale) * scale + min_v
                module.weight.data = W_q.to(dtype=module.weight.dtype)
                if module.bias is not None:
                    min_b, max_b = module.bias.min(), module.bias.max()
                    if max_b > min_b:
                        scale_b = (max_b - min_b) / (2**self.bits - 1)
                        module.bias.data = torch.round((module.bias.data - min_b) / scale_b) * scale_b + min_b
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()