"""
Kronecker Linear Layer: weight matrix represented as Kronecker product A ⊗ B.
Provides true sub-quadratic FLOP execution and factorized memory efficiency.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def factorize_dim(dim: int) -> Tuple[int, int]:
    """
    Factorize a dimension into two balanced factors (a, b) such that a * b = dim
    minimizing |a - b|.
    """
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    a = int(math.isqrt(dim))
    while a > 1:
        if dim % a == 0:
            return (a, dim // a)
        a -= 1
    return (1, dim)


class KroneckerLinear(nn.Module):
    """
    Linear layer whose weight matrix is parameterised as the Kronecker product
    of two smaller factor matrices: W = A ⊗ B.

    For W in R^{out_features x in_features}, with out_features = m1 * m2 and
    in_features = n1 * n2:
    - A in R^{m1 x n1}
    - B in R^{m2 x n2}

    Forward contraction uses the vectorization identity:
    (A ⊗ B) x = vec(B X A^T) without materializing the full (out x in) matrix,
    reducing FLOPs from O(B * in * out) to O(B * (in * m2 + out * n1)).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: Optional[int] = None,
        m1: Optional[int] = None,
        n1: Optional[int] = None,
        m2: Optional[int] = None,
        n2: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Determine factor dimensions
        if m1 is not None and m2 is not None and n1 is not None and n2 is not None:
            if m1 * m2 != out_features or n1 * n2 != in_features:
                raise ValueError(
                    f"Factor dimensions mismatch: {m1}*{m2}!={out_features} or {n1}*{n2}!={in_features}"
                )
            self.m1, self.m2 = m1, m2
            self.n1, self.n2 = n1, n2
        else:
            self.n1, self.n2 = factorize_dim(in_features)
            self.m1, self.m2 = factorize_dim(out_features)

        # Decomposable if neither factor is trivial 1xD when dim > 3
        self.can_decompose = (
            (self.n1 * self.n2 == in_features)
            and (self.m1 * self.m2 == out_features)
            and (self.n1 > 1 or in_features <= 3)
            and (self.m1 > 1 or out_features <= 3)
        )

        if self.can_decompose:
            # A: (m1, n1), B: (m2, n2)
            self.A = nn.Parameter(torch.empty(self.m1, self.n1, dtype=dtype, device=device))
            self.B = nn.Parameter(torch.empty(self.m2, self.n2, dtype=dtype, device=device))

            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1.0
            nn.init.uniform_(self.A, -bound, bound)
            nn.init.uniform_(self.B, -bound, bound)

            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter("bias", None)
        else:
            # Low-rank SVD fallback for prime or non-factorable dimensions
            effective_rank = rank or max(1, min(in_features, out_features) // 4)
            self.rank = min(effective_rank, min(in_features, out_features))
            self.U = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
            self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
            else:
                self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast forward pass using efficient tensor contractions.
        """
        if self.can_decompose:
            orig_shape = x.shape
            x_flat = x.reshape(-1, self.in_features)  # (K, n1 * n2)
            K = x_flat.size(0)

            # Step 1: contract with B over n2 dimension
            # x_flat viewed as (K * n1, n2) @ B^T (n2, m2) -> (K * n1, m2)
            x_b = torch.matmul(x_flat.view(K * self.n1, self.n2), self.B.t())  # (K * n1, m2)

            # Step 2: transpose to (K, m2, n1) -> (K * m2, n1) @ A^T (n1, m1) -> (K * m2, m1)
            x_b = x_b.view(K, self.n1, self.m2).transpose(1, 2).reshape(K * self.m2, self.n1)
            x_a = torch.matmul(x_b, self.A.t())  # (K * m2, m1)

            # Step 3: transpose back to (K, m1, m2) -> (K, m1 * m2)
            out = x_a.view(K, self.m2, self.m1).transpose(1, 2).reshape(K, self.out_features)

            if self.bias is not None:
                out = out + self.bias

            return out.view(*orig_shape[:-1], self.out_features)
        else:
            # Low-rank SVD fallback: y = (x @ V^T) @ U^T + bias
            out = F.linear(F.linear(x, self.V), self.U, self.bias)
            return out

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        if self.can_decompose:
            return torch.kron(self.A, self.B)
        else:
            return torch.matmul(self.U, self.V)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: Optional[int] = None,
        m1: Optional[int] = None,
        n1: Optional[int] = None,
        m2: Optional[int] = None,
        n2: Optional[int] = None,
    ) -> "KroneckerLinear":
        """
        Decompose a pretrained nn.Linear layer into Kronecker factors A and B
        using Nearest Kronecker Product (Van Loan & Pitsianis 1993).
        """
        in_features = linear.in_features
        out_features = linear.out_features
        weight = linear.weight.data.clone()
        bias = linear.bias.data.clone() if linear.bias is not None else None

        kron_layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            rank=rank,
            m1=m1,
            n1=n1,
            m2=m2,
            n2=n2,
            dtype=weight.dtype,
            device=weight.device,
        )

        if kron_layer.can_decompose:
            m1, m2 = kron_layer.m1, kron_layer.m2
            n1, n2 = kron_layer.n1, kron_layer.n2

            # Rearrange W (m1*m2, n1*n2) -> (m1, m2, n1, n2) -> (m1, n1, m2, n2) -> (m1*n1, m2*n2)
            W_tensor = weight.view(m1, m2, n1, n2).permute(0, 2, 1, 3).contiguous()
            W_reordered = W_tensor.view(m1 * n1, m2 * n2).float()

            # Optimal Rank-1 SVD approximation of reordered matrix
            U, S, Vt = torch.linalg.svd(W_reordered, full_matrices=False)
            sigma1 = S[0].item() if S.numel() > 0 else 1.0
            scale = math.sqrt(max(sigma1, 1e-12))

            u1 = U[:, 0] * scale
            v1 = Vt[0, :] * scale

            kron_layer.A.data.copy_(u1.view(m1, n1).to(dtype=weight.dtype))
            kron_layer.B.data.copy_(v1.view(m2, n2).to(dtype=weight.dtype))
        else:
            # Fallback to SVD
            U, S, Vt = torch.linalg.svd(weight.float(), full_matrices=False)
            r = kron_layer.rank
            U_r = U[:, :r] * torch.sqrt(S[:r].unsqueeze(0))
            V_r = torch.sqrt(S[:r].unsqueeze(1)) * Vt[:r, :]
            kron_layer.U.data.copy_(U_r.to(dtype=weight.dtype))
            kron_layer.V.data.copy_(V_r.to(dtype=weight.dtype))

        if bias is not None and kron_layer.bias is not None:
            kron_layer.bias.data.copy_(bias)

        return kron_layer
