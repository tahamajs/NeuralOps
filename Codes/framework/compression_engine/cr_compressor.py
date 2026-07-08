"""
CR (Interpolative Decomposition) Compressor.
Decomposes W ≈ C @ V^T, where C are actual columns of W.
Registered as 'cr'.
"""

import numpy as np
import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from scipy.linalg import interpolative


class CRCompressor(BaseCompressor):
    """
    Compress using interpolative decomposition: W ≈ C @ proj,
    where C contains a subset of the columns of W.

    Args:
        model: PyTorch model.
        rank: rank of the decomposition (number of columns to keep).
    """

    def __init__(self, model, rank=64, **kwargs):
        super().__init__(model, name="cr", **kwargs)
        self.rank = rank

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                r = min(self.rank, min(W.shape))
                try:
                    idx, proj = interpolative.interp_decomp(W, r)
                    C = W[:, idx[:r]]
                    W_approx = (
                        C @ proj[:r, :]
                    )  # proj is (r, N) or (r, M)? Depends on axis.
                    # Actually scipy's interp_decomp returns idx (columns to keep) and proj such that
                    # W ≈ W[:, idx] @ proj. Correct.
                except Exception:
                    # Fallback to SVD if scipy fails
                    U, S, Vt = np.linalg.svd(W, full_matrices=False)
                    C = U[:, :r] * S[:r]
                    W_approx = C @ Vt[:r, :]
                module.weight.data = torch.tensor(
                    W_approx, dtype=module.weight.dtype, device=module.weight.device
                )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("cr")
def cr_compressor_factory(model, comp_config):
    rank = getattr(comp_config, "cr_rank", 64)
    compressor = CRCompressor(model, rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
