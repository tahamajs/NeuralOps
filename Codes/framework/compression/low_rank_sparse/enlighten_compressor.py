import torch
import numpy as np

class ENLightenCompressor:
    def __init__(self, model, rank=64, sparsity=0.5):
        self.model = model
        self.rank = rank
        self.sparsity = sparsity

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                # Low‑rank component via SVD
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                U_r = U[:, :self.rank]
                S_r = S[:self.rank]
                Vt_r = Vt[:self.rank, :]
                L = U_r @ np.diag(S_r) @ Vt_r
                # Structured sparsity: align to photonic granularity (2:4 pattern)
                residual = W - L
                # Apply 2:4 sparsity on residual
                residual_sp = self._apply_24_sparsity(residual)
                W_new = L + residual_sp
                module.weight.data = torch.tensor(W_new, dtype=module.weight.dtype, device=module.weight.device)
        return self.model

    def _apply_24_sparsity(self, mat):
        mat_sp = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            for j in range(0, mat.shape[1], 4):
                sub = mat[i, j:j+4]
                idx = np.argsort(np.abs(sub))[-2:]
                mask = np.zeros(4)
                mask[idx] = 1
                mat_sp[i, j:j+4] = sub * mask
        return mat_sp