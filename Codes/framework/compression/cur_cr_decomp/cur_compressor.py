import torch
import numpy as np
from sklearn.decomposition import randomized_svd
from scipy.linalg import interpolative

class CURCompressor:
    """
    CUR decomposition: W ≈ C @ U @ R, where C,R are actual columns/rows of W.
    Uses column selection via leverage score or interpolative decomposition.
    """
    def __init__(self, model, rank=64, method='leverage'):
        self.model = model
        self.rank = rank
        self.method = method

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                out_dim, in_dim = W.shape
                # Compute column norms (leverage scores)
                col_norms = np.linalg.norm(W, axis=0)
                col_probs = col_norms / col_norms.sum()
                # Select columns
                selected_cols = np.random.choice(in_dim, size=self.rank, replace=False, p=col_probs)
                C = W[:, selected_cols]
                # Compute U
                U, S, Vt = np.linalg.svd(C, full_matrices=False)
                U = U[:, :self.rank]
                # Approximate reconstruction by solving least squares for R
                # W ≈ C @ R' where R' = C^+ W
                R_prime = np.linalg.lstsq(C, W, rcond=None)[0]
                # We can store C and R_prime as small matrices
                new_weight = torch.tensor(C @ R_prime, dtype=module.weight.dtype, device=module.weight.device)
                module.weight.data = new_weight
        return self.modelimport torch
import numpy as np
from sklearn.decomposition import randomized_svd
from scipy.linalg import interpolative

class CURCompressor:
    """
    CUR decomposition: W ≈ C @ U @ R, where C,R are actual columns/rows of W.
    Uses column selection via leverage score or interpolative decomposition.
    """
    def __init__(self, model, rank=64, method='leverage'):
        self.model = model
        self.rank = rank
        self.method = method

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                out_dim, in_dim = W.shape
                # Compute column norms (leverage scores)
                col_norms = np.linalg.norm(W, axis=0)
                col_probs = col_norms / col_norms.sum()
                # Select columns
                selected_cols = np.random.choice(in_dim, size=self.rank, replace=False, p=col_probs)
                C = W[:, selected_cols]
                # Compute U
                U, S, Vt = np.linalg.svd(C, full_matrices=False)
                U = U[:, :self.rank]
                # Approximate reconstruction by solving least squares for R
                # W ≈ C @ R' where R' = C^+ W
                R_prime = np.linalg.lstsq(C, W, rcond=None)[0]
                # We can store C and R_prime as small matrices
                new_weight = torch.tensor(C @ R_prime, dtype=module.weight.dtype, device=module.weight.device)
                module.weight.data = new_weight
        return self.model