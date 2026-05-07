"""A3: grouped QK/OV/MLP compression. Registered as 'a3'."""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class A3Compressor(BaseCompressor):
    """A3 compression logic (not directly registered)."""
    def __init__(self, model: nn.Module, rank: int = 64, **kwargs):
        super().__init__(model, name="a3", **kwargs)
        self.rank = rank

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and ('attention' in name.lower() or 'attn' in name.lower()):
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(self.rank, min(W.shape), len(S))
                if r < len(S):
                    W_approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
                    module.weight.data = W_approx.to(dtype=module.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


# ✅ Register a factory function, not the class
@COMPRESSOR_REGISTRY.register("a3")
def a3_compressor_factory(model: nn.Module, comp_config) -> nn.Module:
    rank = getattr(comp_config, 'a3_rank', 64)
    compressor = A3Compressor(model, rank=rank)
    compressed_model, _ = compressor.compress()
    return compressed_model