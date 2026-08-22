"""
CUR‑MoE: Interpretable CUR decomposition for MoE expert layers.
Registered as 'cur_moe'.
"""
from typing import Any

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


def cur_compress(weight, rank) -> Any:
    """
    Approximate weight matrix W as C @ U @ R, where C and R are column/row subsets.
    """
    out_dim, in_dim = weight.shape
    # Column and row selection (random for simplicity; real CUR uses leverage scores)
    col_idx = torch.randperm(in_dim)[:rank]
    row_idx = torch.randperm(out_dim)[:rank]
    C = weight[:, col_idx]  # (out, rank)
    R = weight[row_idx, :]  # (rank, in)
    # Solve for U: C @ U @ R ≈ W  ->  U ≈ C^+ @ W @ R^+
    C_pinv = torch.linalg.pinv(C.float())
    R_pinv = torch.linalg.pinv(R.float())
    U = C_pinv @ weight.float() @ R_pinv  # (rank, rank)
    return (C @ U @ R).to(weight.dtype)


class CURMoECompressor(BaseCompressor):
    """
    Compresses expert layers using CUR decomposition.

    Args:
        model: PyTorch model.
        rank: rank for CUR decomposition (default 32).
    """

    def __init__(self, model, rank=32, **kwargs):
        super().__init__(model, name="cur_moe", **kwargs)
        self.rank = rank

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "expert" in name:
                module.weight.data = cur_compress(module.weight.data, self.rank)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("cur_moe")
def cur_moe_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Cur moe factory."""
    rank = getattr(comp_config, "cur_moe_rank", kwargs.get("rank", 64))
    method = getattr(comp_config, "cur_moe_method", kwargs.get("method", "leverage"))
    compressor = CURMoECompressor(model, rank=rank, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
