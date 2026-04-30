import torch
import numpy as np

class SLiMCompressor:
    """
    SLiM: one‑shot quantisation, semi‑structured sparsity, low‑rank correction.
    """
    def __init__(self, model, bits=4, sparsity=0.5, rank=8):
        self.model = model
        self.bits = bits
        self.sparsity = sparsity
        self.rank = rank

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                # 1. Quantize to b bits
                min_val, max_val = W.min(), W.max()
                if max_val == min_val:
                    max_val = min_val + 1e-6
                scale = (max_val - min_val) / (2**self.bits - 1)
                W_q = np.round((W - min_val) / scale) * scale + min_val
                # 2. Apply 2:4 structured sparsity
                W_sp = W_q.copy()
                for i in range(W_q.shape[0]):
                    row = W_q[i]
                    for j in range(0, len(row), 4):
                        sub = row[j:j+4]
                        idx = np.argsort(np.abs(sub))[-2:]
                        mask = np.zeros(4)
                        mask[idx] = 1
                        W_sp[i, j:j+4] = sub * mask
                # 3. Low‑rank correction: compute residual, SVD, add back
                residual = W - W_sp
                U, S, Vt = np.linalg.svd(residual, full_matrices=False)
                U_r = U[:, :self.rank]
                S_r = S[:self.rank]
                Vt_r = Vt[:self.rank, :]
                correction = U_r @ np.diag(S_r) @ Vt_r
                W_final = W_sp + correction
                module.weight.data = torch.tensor(W_final, dtype=module.weight.dtype, device=module.weight.device)
        return self.model