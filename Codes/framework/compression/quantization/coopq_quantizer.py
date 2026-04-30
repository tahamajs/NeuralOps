import torch
import numpy as np
from sklearn.cluster import KMeans

class CoopQQuantizer:
    def __init__(self, model, bits=4):
        self.model = model
        self.bits = bits

    def compress(self):
        # Approximate Shapley values by layer sensitivity
        sensitivities = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                w = module.weight.data
                sensitivities[name] = w.abs().max().item()  # proxy
        total_sens = sum(sensitivities.values())
        for name, module in self.model.named_modules():
            if name in sensitivities:
                # assign bits proportional to sensitivity
                frac = sensitivities[name] / total_sens
                bits_layer = max(1, min(8, int(self.bits + 2*frac)))
                # quantize layer
                self._quantize_layer(module, bits_layer)

    def _quantize_layer(self, module, bits):
        W = module.weight.data
        min_v, max_v = W.min(), W.max()
        scale = (max_v - min_v) / (2**bits - 1)
        W_q = torch.round((W - min_v) / scale) * scale + min_v
        module.weight.data = W_q