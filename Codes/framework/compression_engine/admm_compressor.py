"""ADMM-based structured pruning + quantization. Registered as 'admm'."""

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class ADMMCompressor(BaseCompressor):
    def __init__(self, model, sparsity=0.5, quant_bits=4, **kwargs):
        super().__init__(model, name="admm", **kwargs)
        self.sparsity = sparsity
        self.quant_bits = quant_bits

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                # Prune: keep top (1-sparsity)% weights
                flat = W.abs().view(-1)
                k = int(flat.numel() * (1 - self.sparsity))
                thr = torch.topk(flat, k, largest=True).values[-1]
                mask = (W.abs() >= thr).float()
                W_pruned = W * mask
                # Quantize
                scale = (2 ** (self.quant_bits - 1) - 1)
                W_scaled = torch.clamp(W_pruned * scale, -scale, scale)
                W_quant = torch.round(W_scaled) / scale
                module.weight.data = W_quant.to(dtype=module.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    


@COMPRESSOR_REGISTRY.register("admm")
def admm_factory(model, comp_config):
    sparsity = getattr(comp_config, 'admm_sparsity', 0.5)
    quant_bits = getattr(comp_config, 'admm_quant_bits', 4)
    compressor = ADMMCompressor(model, sparsity=sparsity, quant_bits=quant_bits)
    compressed_model, _ = compressor.compress()
    return compressed_model
