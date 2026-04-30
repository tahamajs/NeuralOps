import torch
import numpy as np
from scipy.linalg import interpolative

class CRCompressor:
    """
    Interpolative Decomposition (CR): W ≈ C @ V^T where C are actual columns,
    V is a row selection matrix.
    """
    def __init__(self, model, rank=64):
        self.model = model
        self.rank = rank

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                # Use scipy's interpolative decomposition
                idx, proj = interpolative.interp_decomp(W, self.rank)
                # Keep selected columns
                C = W[:, idx[:self.rank]]
                # Reconstruct
                W_approx = C @ proj
                module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        return self.model