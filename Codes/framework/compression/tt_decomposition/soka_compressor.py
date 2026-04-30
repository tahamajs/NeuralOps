import torch
import numpy as np

class SoKACompressor:
    def __init__(self, model, rank=8):
        self.model = model
        self.rank = rank

    def compress(self):
        # Kronecker product approximation via SVD on reshaped matrix
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                n, m = W.shape
                # Assume n = n1*n2, m = m1*m2 (here we use 2 factors)
                n1 = int(np.sqrt(n))
                n2 = n // n1
                m1 = int(np.sqrt(m))
                m2 = m // m1
                # Reshape to order‑4 tensor and apply higher‑order SVD
                # Placeholder: just use regular SVD
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                U_r = U[:, :self.rank]
                Vt_r = Vt[:self.rank, :]
                W_approx = U_r @ np.diag(S[:self.rank]) @ Vt_r
                module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        return self.model