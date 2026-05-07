"""CoopQ: Shapley‑inspired per‑layer bit allocation. Registered as 'coopq'."""

import torch
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class CoopQQuantizer(BaseCompressor):
    def __init__(self, model, bits=4, **kwargs):
        super().__init__(model, name="coopq", **kwargs)
        self.bits = bits

    def _quantize_layer(self, module, bits):
        W = module.weight.data.float()
        min_v, max_v = W.min(), W.max()
        if max_v == min_v:
            return
        scale = (max_v - min_v) / (2**bits - 1)
        W_q = torch.round((W - min_v) / scale) * scale + min_v
        module.weight.data = W_q.to(dtype=module.weight.dtype)
        if module.bias is not None:
            min_b, max_b = module.bias.min(), module.bias.max()
            if max_b > min_b:
                scale_b = (max_b - min_b) / (2**bits - 1)
                module.bias.data = torch.round((module.bias.data - min_b) / scale_b) * scale_b + min_b

    def compress(self):
        sensitivities = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                sensitivities[name] = module.weight.data.abs().max().item()
        total_sens = sum(sensitivities.values())
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in sensitivities:
                frac = sensitivities[name] / total_sens
                bits_layer = max(1, min(8, int(self.bits + 4 * frac)))
                self._quantize_layer(module, bits_layer)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    

@COMPRESSOR_REGISTRY.register("coopq")
def coopq_factory(model, comp_config):
    bits = getattr(comp_config, 'coopq_bits', 4)
    compressor = CoopQQuantizer(model, bits)
    compressed_model, _ = compressor.compress()
    return compressed_model
