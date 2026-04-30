import torch
import numpy as np
from sklearn.decomposition import randomized_svd

class GreGoDecCompressor:
    """
    GreGoDec: decompose W = L (low‑rank) + S (sparse) + E (noise).
    Uses randomized sketching and greedy selection.
    """
    def __init__(self, model, rank=64, sparsity=0.1, max_iter=10):
        self.model = model
        self.rank = rank
        self.sparsity = sparsity
        self.max_iter = max_iter

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                # Initialize L as truncated SVD
                U, S, Vt = randomized_svd(W, n_components=self.rank)
                L = U @ np.diag(S) @ Vt
                S_mat = np.zeros_like(W)
                for _ in range(self.max_iter):
                    # Update S: keep largest entries of W - L
                    residual = W - L
                    threshold = np.percentile(np.abs(residual), 100*(1 - self.sparsity))
                    S_mat = np.where(np.abs(residual) > threshold, residual, 0)
                    # Update L: rank‑k SVD of W - S_mat
                    U, S_vals, Vt = randomized_svd(W - S_mat, n_components=self.rank)
                    L = U @ np.diag(S_vals) @ Vt
                # Keep only L as compressed weight (dense, low‑rank)
                module.weight.data = torch.tensor(L, dtype=module.weight.dtype, device=module.weight.device)
        return self.model