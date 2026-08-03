"""ProxSparse: regularized learning of semi‑structured sparsity. Registered as 'proxsparse'."""

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class ProxSparsePruner(BaseCompressor):
    def __init__(self, model, sparsity=0.5, lambda_prox=0.01, steps=5, **kwargs):
        super().__init__(model, name="proxsparse", **kwargs)
        self.sparsity = sparsity
        self.lambda_prox = lambda_prox
        self.steps = steps

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                for _ in range(self.steps):
                    threshold = np.percentile(np.abs(W), 100 * (1 - self.sparsity))
                    W_prox = np.where(
                        np.abs(W) > threshold, W - np.sign(W) * threshold, 0
                    )
                    W = W - self.lambda_prox * (W - W_prox)
                module.weight.data = torch.tensor(
                    W_prox, dtype=module.weight.dtype, device=module.weight.device
                )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("proxsparse")
def proxsparse_factory(model, comp_config) -> Any:
    """Proxsparse factory."""
    sparsity = getattr(comp_config, "proxsparse_sparsity", 0.5)
    lambda_prox = getattr(comp_config, "proxsparse_lambda", 0.01)
    steps = getattr(comp_config, "proxsparse_steps", 5)
    compressor = ProxSparsePruner(
        model, sparsity=sparsity, lambda_prox=lambda_prox, steps=steps
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
