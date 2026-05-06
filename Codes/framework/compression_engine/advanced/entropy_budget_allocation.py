"""
EntropyBudgetAllocation: global rank assignment via whitened activation entropy.
Registered as 'entropy_budget'.
"""

import torch
import torch.nn as nn
from ..core.base import BaseCompressor
from ..core.registry import COMPRESSOR_REGISTRY


@COMPRESSOR_REGISTRY.register("entropy_budget")
class EntropyBudgetCompressor(BaseCompressor):
    def __init__(self, model, calibration_loader, total_budget_ratio=0.7, **kwargs):
        super().__init__(model, name="entropy_budget", **kwargs)
        self.calibration_loader = calibration_loader
        self.total_budget_ratio = total_budget_ratio

    def collect_covariances(self):
        activations = {}
        hooks = []
        def hook_fn(module, input, output, name):
            x = input[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).float()
            cov = x_flat.T @ x_flat / x_flat.shape[0]
            if name not in activations:
                activations[name] = cov
            else:
                activations[name] += cov

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))

        with torch.no_grad():
            for batch in self.calibration_loader:
                batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
                self.model(**batch)
        for h in hooks:
            h.remove()
        return activations

    def compress(self):
        components = []
        covs = self.collect_covariances()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in covs:
                W = module.weight.data.float()
                L = torch.linalg.cholesky(covs[name])
                M = W @ L
                U, S, _ = torch.linalg.svd(M, full_matrices=False)
                entropy = 0.5 * torch.sum(torch.log(2 * torch.pi * torch.e * S**2))
                for i, s in enumerate(S):
                    components.append((name, i, s.item(), entropy.item()))

        components.sort(key=lambda x: x[2]**2 / x[3], reverse=True)
        total_components = len(components)
        keep_count = max(1, int(total_components * self.total_budget_ratio))
        keep_set = set((c[0], c[1]) for c in components[:keep_count])

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in covs:
                rank = len([i for (n, i) in keep_set if n == name])
                if rank == 0:
                    continue
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(rank, len(S))
                W_approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
                module.weight.data = W_approx.to(dtype=module.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()