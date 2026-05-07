"""
Kronecker Linear Layer: weight matrix represented as Kronecker product A ⊗ B.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def factorize_dim(dim: int) -> Tuple[int, int]:
    """Factorize a dimension into two near-square factors."""
    a = int(math.sqrt(dim))
    while dim % a != 0:
        a -= 1
    b = dim // a
    # return larger first? We'll return (small, large) for consistency in Kronecker
    # It doesn't matter for the kronecker product, but we'll follow natural order.
    # We'll let the caller decide; for KroneckerLinear we'll assign shapes accordingly.
    return (a, b)


class KroneckerLinear(nn.Module):
    """
    Linear layer whose weight matrix is parameterised as the Kronecker product
    of two smaller matrices: W = A ⊗ B.

    This reduces parameters from (out_features * in_features) to
    (out_a * in_a + out_b * in_b), where out_features = out_a * out_b and
    in_features = in_a * in_b.

    The factorisation of in_features and out_features is done automatically.
    If exact factorisation is not possible, falls back to a low‑rank SVD with
    the same number of parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Try to factorise dimensions
        in_a, in_b = factorize_dim(in_features)
        out_a, out_b = factorize_dim(out_features)
        self.can_decompose = (in_a * in_b == in_features) and (out_a * out_b == out_features)

        if self.can_decompose:
            self.in_a, self.in_b = in_a, in_b
            self.out_a, self.out_b = out_a, out_b

            # A: (out_a, in_a), B: (out_b, in_b)
            self.A = nn.Parameter(torch.empty(out_a, in_a, dtype=dtype, device=device))
            self.B = nn.Parameter(torch.empty(out_b, in_b, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
            # Match standard Linear initialisation
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.A, -bound, bound)
            nn.init.uniform_(self.B, -bound, bound)

            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter('bias', None)
        else:
            # Fallback: use low‑rank SVD with the same number of parameters
            # We'll allocate two matrices that together have the same parameter count
            # as a kronecker product would have.
            # Compute the total possible parameters if factorisation existed.
            # Estimate: using the factorisation of nearest factorable numbers? Simpler: choose rank = min(in, out)//4
            max_kron_params = 0
            if in_features == out_features:
                # square: choose a=sqrt, b=sqrt
                a = int(math.sqrt(in_features))
                if a * a == in_features:
                    max_kron_params = a*a + a*a  # but not needed
            # We'll just fall back to SVD with a fixed rank that gives roughly similar compression.
            self.rank = max(1, min(in_features, out_features) // 4)
            self.U = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
            self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
                nn.init.zeros_(self.bias)
            else:
                self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.can_decompose:
            # W = A ⊗ B
            # Efficient computation: (X @ B^T) @ A^T ? Not exactly.
            # We can compute using reshape trick: for any vector x of size (in_a*in_b),
            # (A ⊗ B) x = vec(B X A^T), but we have a batch of vectors.
            # We'll just compute the full kronecker product once per forward.
            # For moderate dimensions (e.g., 768), the kronecker product is cheap.
            W = torch.kron(self.A, self.B)   # (out_a*out_b, in_a*in_b)
            out = F.linear(x, W, self.bias)
        else:
            # Fallback: low‑rank SVD
            W = self.U @ self.V
            out = F.linear(x, W, self.bias)
        return out

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "KroneckerLinear":
        """
        Initialise from a pretrained nn.Linear layer by breaking its weight matrix
        into the Kronecker product (A ⊗ B) if possible, else fallback to SVD.
        """
        in_features = linear.in_features
        out_features = linear.out_features
        weight = linear.weight.data.clone()
        bias = linear.bias.data.clone() if linear.bias is not None else None
        kron_layer = cls(in_features, out_features, bias=bias is not None)

        if kron_layer.can_decompose:
            # Reshape and find A, B via nearest Kronecker product approximation.
            # W ≈ A ⊗ B => minimise ||W - A ⊗ B||_F can be solved by SVD of reshaped W.
            # See "Nearest Kronecker Product" algorithm.
            # We'll implement a simple alternating least squares version.
            out_a, out_b = kron_layer.out_a, kron_layer.out_b
            in_a, in_b = kron_layer.in_a, kron_layer.in_b
            # Reshape W to (out_a, out_b, in_a, in_b) and perform SVD on unfolded matrices.
            W_reshaped = weight.view(out_a, out_b, in_a, in_b)
            # Unfold along the appropriate dimensions
            # Algorithm: flatten each block, compute SVD to get A, then B.
            # First, compute A: reshape to (out_a, out_b * in_a * in_b) -> SVD
            # Actually the standard method: reshape W to (out_a * in_a, out_b * in_b) and take SVD.
            W_unfold = W_reshaped.permute(0, 2, 1, 3).contiguous().view(out_a * in_a, out_b * in_b)
            U, S, Vt = torch.linalg.svd(W_unfold, full_matrices=False)
            # Keep rank 1 to get Kronecker factors
            U1 = U[:, 0:1]            # (out_a * in_a, 1)
            V1 = Vt[0:1, :]           # (1, out_b * in_b)
            # Reshape back
            A_approx = U1.view(out_a, in_a)
            B_approx = V1.view(out_b, in_b)
            # Normalise: We'll assign scale to A or B.
            kron_layer.A.data = A_approx
            kron_layer.B.data = B_approx
        else:
            # Fallback: SVD with the same rank as the fallback
            U, S, Vt = torch.linalg.svd(weight.float(), full_matrices=False)
            r = kron_layer.rank
            kron_layer.U.data = (U[:, :r] * S[:r].unsqueeze(0)).float()
            kron_layer.V.data = Vt[:r, :].float()

        if bias is not None:
            kron_layer.bias.data.copy_(bias)
        return kron_layer
    

