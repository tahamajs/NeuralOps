import torch
import numpy as np
from sklearn.decomposition import PCA

class RPCACompressor:
    """
    Robust PCA: decompose W = L + S, keep L as low‑rank, discard S as noise.
    Uses alternating minimization (simplified).
    """
    def __init__(self, model, rank=64, mu=1e-3, max_iter=5):
        self.model = model
        self.rank = rank
        self.mu = mu
        self.max_iter = max_iter

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                # Initialize
                L = W.copy()
                S = np.zeros_like(W)
                for _ in range(self.max_iter):
                    # Low‑rank update
                    U, S_svd, Vt = np.linalg.svd(L, full_matrices=False)
                    L = U[:, :self.rank] @ np.diag(S_svd[:self.rank]) @ Vt[:self.rank, :]
                    # Sparse update (soft‑threshold)
                    S = np.sign(W - L) * np.maximum(np.abs(W - L) - self.mu, 0)
                    L = W - S
                # Keep only L as compressed weight
                module.weight.data = torch.tensor(L, dtype=module.weight.dtype, device=module.weight.device)
        return self.model