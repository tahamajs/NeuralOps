"""
CUR‑MoE Compressor: apply CUR decomposition to MoE expert layers.
Registered as 'cur_moe'.
"""
from typing import Any

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class CURMoECompressor(BaseCompressor):
    """
    Compresses MoE expert layers using CUR decomposition.

    Args:
        model: PyTorch model containing MoE modules.
        rank: rank for CUR decomposition.
        method: 'leverage' or 'uniform'.
    """

    def __init__(self, model, rank=64, method="leverage", **kwargs):
        super().__init__(model, name="cur_moe", **kwargs)
        self.rank = rank
        self.method = method

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if hasattr(module, "experts"):
                for expert in module.experts:
                    for param_name, param in expert.named_parameters():
                        if "weight" in param_name and param.dim() == 2:
                            W = param.data.float().cpu().numpy()
                            out_dim, in_dim = W.shape
                            r = min(self.rank, min(out_dim, in_dim))

                            # Column selection
                            if self.method == "leverage":
                                col_norms = np.linalg.norm(W, axis=0)
                                col_probs = col_norms / (col_norms.sum() + 1e-8)
                            else:
                                col_probs = np.ones(in_dim) / in_dim
                            selected_cols = np.random.choice(
                                in_dim, size=r, replace=False, p=col_probs
                            )
                            C = W[:, selected_cols]

                            # Row selection
                            if self.method == "leverage":
                                row_norms = np.linalg.norm(W, axis=1)
                                row_probs = row_norms / (row_norms.sum() + 1e-8)
                            else:
                                row_probs = np.ones(out_dim) / out_dim
                            selected_rows = np.random.choice(
                                out_dim, size=r, replace=False, p=row_probs
                            )
                            R = W[selected_rows, :]

                            # Compute U
                            C_pinv = np.linalg.pinv(C)
                            R_pinv = np.linalg.pinv(R)
                            U = C_pinv @ W @ R_pinv

                            W_approx = C @ U @ R
                            param.data = torch.tensor(
                                W_approx, dtype=param.dtype, device=param.device
                            )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("cur_moe")
def cur_moe_factory(model, comp_config) -> Any:
    """Cur moe factory."""
    rank = getattr(comp_config, "cur_moe_rank", 64)
    method = getattr(comp_config, "cur_moe_method", "leverage")
    compressor = CURMoECompressor(model, rank=rank, method=method)
    compressed_model, _ = compressor.compress()
    return compressed_model
