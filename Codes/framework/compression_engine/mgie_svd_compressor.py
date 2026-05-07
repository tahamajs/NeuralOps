"""MGIE‑SVD with total rank budget. Registered as 'mgie_budget'."""

import torch
import numpy as np
from scipy.stats import entropy
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class MGIESVDBudgetCompressor(BaseCompressor):
    def __init__(self, model, total_rank_budget=1000, **kwargs):
        super().__init__(model, name="mgie_budget", **kwargs)
        self.total_rank_budget = total_rank_budget

    def compress(self):
        layers = []
        entropies = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                _, S, _ = np.linalg.svd(W, full_matrices=False)
                p = S / S.sum()
                e = entropy(p, base=2)
                layers.append((name, module, min(W.shape)))
                entropies.append(e)
        entropies = np.array(entropies)
        total_e = entropies.sum()
        if total_e == 0:
            ranks = np.ones(len(layers), dtype=int)
        else:
            ranks = np.maximum(1, (entropies / total_e * self.total_rank_budget).astype(int))
        for (name, module, max_rank), rank in zip(layers, ranks):
            rank = min(rank, max_rank)
            W = module.weight.data.float().cpu().numpy()
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            W_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
            module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    


@COMPRESSOR_REGISTRY.register("mgie_budget")
def mgie_budget_factory(model, comp_config):
    total_rank_budget = getattr(comp_config, 'total_rank_budget', 1000)
    compressor = MGIESVDBudgetCompressor(model, total_rank_budget=total_rank_budget)
    compressed_model, _ = compressor.compress()
    return compressed_model