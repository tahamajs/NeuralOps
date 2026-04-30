# framework/compression/entropy_budget_allocation.py
import torch
import torch.nn as nn

class EntropyBudgetAllocationCompressor:
    def __init__(self, model, calibration_loader, total_budget_ratio=0.7):
        self.model = model
        self.calibration_loader = calibration_loader
        self.total_budget_ratio = total_budget_ratio   # fraction of total singular components to keep

    def compress(self):
        # 1. Collect activation covariances and compute whitened singular values
        components = []  # list of (layer_id, singular_value, entropy)
        for (name, module), cov in self.collect_covariances():
            W = module.weight.data
            L = torch.linalg.cholesky(cov)
            M = W @ L
            U, S, Vt = torch.linalg.svd(M, full_matrices=False)
            # Entropy = 0.5 * sum(log(2πe σ_i²))
            entropy = 0.5 * torch.sum(torch.log(2 * torch.pi * torch.e * S**2))
            for i, s in enumerate(S):
                components.append((name, i, s.item(), entropy.item()))

        # 2. Global budget allocation: keep components with highest s² / entropy
        components.sort(key=lambda x: x[2]**2 / x[3], reverse=True)
        total_components = len(components)
        keep_count = int(total_components * self.total_budget_ratio)
        keep_set = set((c[0], c[1]) for c in components[:keep_count])

        # 3. Apply truncated SVD per layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_ranks = [i for (n, i) in keep_set if n == name]
                rank = len(layer_ranks)
                if rank == 0:
                    continue
                U, S, Vt = torch.linalg.svd(module.weight.data, full_matrices=False)
                U_r = U[:, :rank]
                S_r = S[:rank]
                Vt_r = Vt[:rank, :]
                module.weight.data = U_r @ torch.diag(S_r) @ Vt_r
        return self.model