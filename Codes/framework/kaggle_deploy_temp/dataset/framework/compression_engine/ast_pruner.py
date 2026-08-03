"""AST: semi-structured 2:4 sparsity. Registered as 'ast_pruner'."""
from typing import Any

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class ASTPruner(BaseCompressor):
    def __init__(self, model, block_size=4, **kwargs):
        super().__init__(model, name="ast_pruner", **kwargs)
        self.block_size = block_size

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                N, M = W.shape
                W_ast = np.zeros_like(W)
                for i in range(N):
                    for j in range(0, M, self.block_size):
                        sub = W[i, j : j + self.block_size]
                        idx = np.argsort(np.abs(sub))[-2:]  # keep top 2
                        mask = np.zeros(self.block_size)
                        mask[idx] = 1
                        W_ast[i, j : j + self.block_size] = sub * mask
                module.weight.data = torch.tensor(
                    W_ast, dtype=module.weight.dtype, device=module.weight.device
                )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("ast_pruner")
def ast_pruner_factory(model, comp_config) -> Any:
    """Ast pruner factory."""
    block_size = getattr(comp_config, "ast_block_size", 4)
    compressor = ASTPruner(model, block_size)
    compressed_model, _ = compressor.compress()
    return compressed_model
