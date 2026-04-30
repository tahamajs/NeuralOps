import torch
import numpy as np
from scipy.stats import entropy

class MGIESVDCompressor:
    """
    MGIE‑SVD: Multidimensional Gaussian Information Entropy guided SVD.
    Allocates compression budget based on entropy of singular values.
    """
    def __init__(self, model, total_rank_budget=1000):
        self.model = model
        self.total_rank_budget = total_rank_budget

    def compress(self):
        layers = []
        entropies = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                _, S, _ = np.linalg.svd(W, full_matrices=False)
                # normalized singular values
                p = S / S.sum()
                e = entropy(p, base=2)
                layers.append((name, module, S.shape[0]))
                entropies.append(e)
        entropies = np.array(entropies)
        # allocate rank proportional to entropy
        ranks = (entropies / entropies.sum() * self.total_rank_budget).astype(int)
        for (name, module, max_rank), rank in zip(layers, ranks):
            rank = min(max(1, rank), max_rank)
            W = module.weight.data.cpu().numpy()
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vt_r = Vt[:rank, :]
            W_approx = U_r @ np.diag(S_r) @ Vt_r
            module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        return self.model