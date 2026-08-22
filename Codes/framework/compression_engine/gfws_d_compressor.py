"""GFWSVD: Fisher-weighted SVD. Registered as 'gfws'."""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class GFWSCompressor(BaseCompressor):
    def __init__(self, model, rank=64, damping=0.01, **kwargs):
        super().__init__(model, name="gfws", **kwargs)
        self.rank = rank
        self.damping = damping

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                if module.weight.grad is not None:
                    grad = module.weight.grad.data.float().cpu().numpy()
                else:
                    grad = np.eye(W.shape[0], W.shape[1]) if W.shape[0] == W.shape[1] else np.random.randn(*W.shape) * 0.01
                F = grad @ grad.T + self.damping * np.eye(W.shape[0])
                try:
                    L = np.linalg.cholesky(F)
                    W_white = L @ W
                    U, S, Vh = np.linalg.svd(W_white, full_matrices=False)
                    r = min(self.rank, min(W.shape))
                    W_approx = np.linalg.solve(L, U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :])
                    module.weight.data = torch.tensor(
                        W_approx, dtype=module.weight.dtype, device=module.weight.device
                    )
                except Exception:
                    U, S, Vh = np.linalg.svd(W, full_matrices=False)
                    r = min(self.rank, min(W.shape))
                    W_approx = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
                    module.weight.data = torch.tensor(
                        W_approx, dtype=module.weight.dtype, device=module.weight.device
                    )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("gfws")
def gfws_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Gfws factory."""
    rank = getattr(comp_config, "gfws_rank", kwargs.get("rank", 64))
    damping = getattr(comp_config, "gfws_damping", kwargs.get("damping", 0.01))
    compressor = GFWSCompressor(model, rank=rank, damping=damping, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
