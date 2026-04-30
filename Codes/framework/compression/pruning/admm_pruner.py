import torch
import numpy as np

class ADMMPruner:
    """
    ADMM for weight pruning: solves min_{W} L(W) s.t. sparsity constraints via ADMM.
    Simplified for post‑training: directly project onto structured sparsity.
    """
    def __init__(self, model, sparsity=0.5, rho=0.1, max_iter=5):
        self.model = model
        self.sparsity = sparsity
        self.rho = rho
        self.max_iter = max_iter

    def prune(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                Z = W.copy()
                U = np.zeros_like(W)
                for _ in range(self.max_iter):
                    # W‑update: project onto Euclidean ball (keep close to original)
                    # Z‑update: enforce sparsity
                    Z_plus_U = Z + U
                    # Hard thresholding
                    threshold = np.percentile(np.abs(Z_plus_U), 100 * self.sparsity)
                    Z = np.where(np.abs(Z_plus_U) > threshold, Z_plus_U, 0)
                    U = U + (W - Z)
                module.weight.data = torch.tensor(Z, dtype=module.weight.dtype, device=module.weight.device)
        return self.model