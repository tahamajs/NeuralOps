"""
SDRM: rank assignment proportional to parameter budget for edge inference.
Registered as 'sdrm'.
"""

import torch
import numpy as np
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class SDRMCompressor(BaseCompressor):
    def __init__(self, model, total_param_ratio=0.7, **kwargs):
        super().__init__(model, name="sdrm", **kwargs)
        self.total_param_ratio = total_param_ratio

    def compress(self):
        total_params = 0
        layer_params = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                n, m = module.weight.shape
                params = n * m
                layer_params[name] = (module, params, n, m)
                total_params += params
        target_params = total_params * self.total_param_ratio

        for name, (module, p, n, m) in layer_params.items():
            allowed = p * self.total_param_ratio
            rank = max(1, int(allowed / (n + m)))
            rank = min(rank, min(n, m))
            W = module.weight.data.float().cpu().numpy()
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            r = min(rank, len(S))
            W_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
            module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    



    # ... (rest unchanged) ...

@COMPRESSOR_REGISTRY.register("sdrm")
def sdrm_compressor_factory(model, comp_config):
    total_param_ratio = getattr(comp_config, 'total_param_ratio', 0.7)
    compressor = SDRMCompressor(model, total_param_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model