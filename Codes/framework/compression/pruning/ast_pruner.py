"""AST: semi-structured 2:4 sparsity. Registered as 'ast_pruner'."""

import torch
import numpy as np
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("ast_pruner")
class ASTPruner(BaseCompressor):
    def __init__(self, model, block_size=4, **kwargs):
        super().__init__(model, name="ast_pruner", **kwargs)
        self.block_size = block_size

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                N, M = W.shape
                W_ast = np.zeros_like(W)
                for i in range(N):
                    for j in range(0, M, self.block_size):
                        sub = W[i, j:j+self.block_size]
                        idx = np.argsort(np.abs(sub))[-2:]  # keep top 2
                        mask = np.zeros(self.block_size)
                        mask[idx] = 1
                        W_ast[i, j:j+self.block_size] = sub * mask
                module.weight.data = torch.tensor(W_ast, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()