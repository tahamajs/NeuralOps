import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("factorised_matrix_state")
class FactorisedMatrixState(nn.Module):
    """Maintains a low‑rank factorised state for linear attention."""
    def __init__(self, d_model, rank=32):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.L = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.R = nn.Parameter(torch.randn(d_model, rank) * 0.01)

    def forward(self, x, k, v):
        """
        Args:
            x: (B, D) query token
            k: (B, D) key token
            v: (B, D) value token
        Returns:
            updated state (B, D, D) and output (B, D)
        """
        B, D = x.shape
        H = self.L @ self.R.T  # (D, D)
        Hk = H @ k.unsqueeze(-1)  # (B, D, 1)
        u = x.unsqueeze(-1) - Hk  # (B, D, 1)
        # Update factors with outer product
        self.L.data = torch.cat([self.L, u.squeeze(-1).unsqueeze(-1)], dim=-1)  # (D, r+1)
        self.R.data = torch.cat([self.R, v.unsqueeze(-1)], dim=-1)             # (D, r+1)
        # Truncation via SVD
        M = self.L @ self.R.T
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        self.L.data = U[:, :self.rank]
        self.R.data = Vt[:self.rank, :].T @ torch.diag(S[:self.rank])
        # Output: attention result = H x
        out = (self.L @ self.R.T) @ x.unsqueeze(-1)  # (B, D, 1)
        return out.squeeze(-1)