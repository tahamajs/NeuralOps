import torch
import numpy as np
from sklearn.decomposition import NMF as skNMF

class NMFCompressor:
    """
    Approximate weight matrices with non‑negative factorization.
    Since weights may be negative, we first shift them to positive range.
    """
    def __init__(self, model, rank=64):
        self.model = model
        self.rank = rank

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                min_val = W.min()
                # Shift to non‑negative
                W_shifted = W - min_val + 1e-5
                nmf = skNMF(n_components=self.rank, init='random', max_iter=200)
                H = nmf.fit_transform(W_shifted)   # (out_features, rank)
                W_comp = H @ nmf.components_ + min_val - 1e-5
                module.weight.data = torch.tensor(W_comp, dtype=module.weight.dtype, device=module.weight.device)
        return self.model