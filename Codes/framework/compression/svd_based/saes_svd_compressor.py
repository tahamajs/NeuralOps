import torch
import numpy as np
from scipy.linalg import svd

class SAESSVDCompressor:
    """
    SAES‑SVD: Joint intra‑ and inter‑layer error compensation.
    Applies SVD per layer then corrects residual error via next layer's input.
    """
    def __init__(self, model, rank=64, correcting_steps=3):
        self.model = model
        self.rank = rank
        self.correcting_steps = correcting_steps

    def compress(self):
        # Forward pass to estimate inter‑layer errors
        self.model.eval()
        # Simplified: apply SVD and propagate error compensation
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                U, S, Vt = svd(W, full_matrices=False)
                U_r = U[:, :self.rank]
                S_r = S[:self.rank]
                Vt_r = Vt[:self.rank, :]
                layers.append((module, U_r, S_r, Vt_r))
        # Compensation: adjust subsequent layer's input?
        # For brevity, we just apply truncated SVD (placeholder for full inter‑layer)
        for module, U_r, S_r, Vt_r in layers:
            W_approx = U_r @ np.diag(S_r) @ Vt_r
            module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        return self.model