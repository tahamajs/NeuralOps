import torch
import numpy as np

class GFWSVDCompressor:
    """
    GFWSVD: Fisher‑weighted SVD using both diagonal and off‑diagonal elements.
    Approximates Fisher information via squared gradient outer products.
    """
    def __init__(self, model, rank=64, damping=0.01):
        self.model = model
        self.rank = rank
        self.damping = damping

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.weight.grad is not None:
                W = module.weight.data.cpu().numpy()
                grad = module.weight.grad.data.cpu().numpy()
                # Approximate Fisher: F = grad^T grad (full matrix) – simplified as outer product of gradient
                F = grad @ grad.T + self.damping * np.eye(W.shape[0])
                # Cholesky to whiten
                L = np.linalg.cholesky(F)
                W_white = L @ W
                U, S, Vt = np.linalg.svd(W_white, full_matrices=False)
                U_r = U[:, :self.rank]
                S_r = S[:self.rank]
                Vt_r = Vt[:self.rank, :]
                # Unwhiten
                W_approx = np.linalg.solve(L, U_r @ np.diag(S_r) @ Vt_r)
                module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        return self.model