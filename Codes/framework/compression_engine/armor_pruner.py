"""ARMOR: 2:4 sparse core + low-rank error correction. Registered as 'armor_pruner'."""

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class ARMORPruner(BaseCompressor):
    def __init__(self, model, block_size=4, **kwargs):
        super().__init__(model, name="armor_pruner", **kwargs)
        self.block_size = block_size

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                N, M = W.shape
                W_pruned = np.zeros_like(W)
                for i in range(0, N, self.block_size):
                    blk = W[i : i + self.block_size, :]
                    abs_blk = np.abs(blk)
                    # keep top 2 per row
                    threshold = np.sort(abs_blk, axis=1)[:, -2:]
                    mask = abs_blk >= threshold[:, 0:1]
                    W_pruned[i : i + self.block_size] = blk * mask
                # Low-rank correction
                residual = W - W_pruned
                U, S, Vt = np.linalg.svd(residual, full_matrices=False)
                rank = min(16, len(S))
                correction = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
                W_armor = W_pruned + correction
                module.weight.data = torch.tensor(
                    W_armor, dtype=module.weight.dtype, device=module.weight.device
                )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("armor_pruner")
def armor_pruner_factory(model, comp_config) -> Any:
    """Armor pruner factory."""
    block_size = getattr(comp_config, "armor_block_size", 4)
    compressor = ARMORPruner(model, block_size=block_size)
    compressed_model, _ = compressor.compress()
    return compressed_model
