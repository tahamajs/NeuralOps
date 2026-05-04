"""
CUR Compressor: W ≈ C @ U @ R, where C, R are actual columns/rows.
Registered as 'cur'.
"""

import torch
import torch.nn as nn
import numpy as np
from ..core.base import BaseCompressor
from ..core.registry import COMPRESSOR_REGISTRY


@COMPRESSOR_REGISTRY.register("cur")
class CURCompressor(BaseCompressor):
    """
    Compress using CUR decomposition with leverage score sampling.

    Args:
        model: PyTorch model.
        rank: number of columns/rows to select.
        method: 'leverage' or 'uniform'.
    """
    def __init__(self, model, rank=64, method='leverage', **kwargs):
        super().__init__(model, name="cur", **kwargs)
        self.rank = rank
        self.method = method

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                out_dim, in_dim = W.shape
                r = min(self.rank, min(out_dim, in_dim))

                # Column selection
                if self.method == 'leverage':
                    col_norms = np.linalg.norm(W, axis=0)
                    col_probs = col_norms / (col_norms.sum() + 1e-8)
                else:
                    col_probs = np.ones(in_dim) / in_dim
                selected_cols = np.random.choice(in_dim, size=r, replace=False, p=col_probs)
                C = W[:, selected_cols]   # (out, r)

                # Row selection
                if self.method == 'leverage':
                    row_norms = np.linalg.norm(W, axis=1)
                    row_probs = row_norms / (row_norms.sum() + 1e-8)
                else:
                    row_probs = np.ones(out_dim) / out_dim
                selected_rows = np.random.choice(out_dim, size=r, replace=False, p=row_probs)
                R = W[selected_rows, :]   # (r, in)

                # Compute middle matrix U: U = C^+ @ W @ R^+
                C_pinv = np.linalg.pinv(C)
                R_pinv = np.linalg.pinv(R)
                U = C_pinv @ W @ R_pinv   # (r, r)

                W_approx = C @ U @ R
                module.weight.data = torch.tensor(
                    W_approx, dtype=module.weight.dtype, device=module.weight.device
                )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()