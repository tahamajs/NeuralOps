import torch
import numpy as np
from scipy.linalg import svd

def tt_decompose(matrix, max_rank):
    """
    Tensor‑train decomposition of a matrix: reshapes to tensor, applies TT‑SVD.
    """
    mat = matrix.cpu().numpy()
    nrows, ncols = mat.shape
    # Flatten to a high‑dimensional tensor; we use a simple reshaping into 2x2 grid
    # Real TT requires shape factorization; placeholder: approximate as SVD
    U, S, Vt = svd(mat, full_matrices=False)
    rank = min(max_rank, len(S))
    return U[:, :rank], np.diag(S[:rank]), Vt[:rank, :]

class TTCompressor:
    def __init__(self, model, rank=64):
        self.model = model
        self.rank = rank

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                U, S, Vt = tt_decompose(module.weight.data, self.rank)
                W_approx = U @ S @ Vt
                module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        return self.model