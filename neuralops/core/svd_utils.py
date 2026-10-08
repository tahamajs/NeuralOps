"""
Canonical low-rank SVD utilities for NeuralOps.
"""

from __future__ import annotations

import torch


def randomized_svd(
    W: torch.Tensor, rank: int, n_iter: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Compute fast randomized Singular Value Decomposition (SVD) for 2D matrices.

    Approximates $W \approx U \Sigma V^T$ with top-`rank` singular components using
    subspace iteration.

    Args:
        W (torch.Tensor): 2D input tensor of shape `(M, N)`.
        rank (int): Target decomposition rank.
        n_iter (int, optional): Number of subspace power iterations. Defaults to 4.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - U (torch.Tensor): Left singular vectors of shape `(M, rank)`.
            - S (torch.Tensor): Singular values of shape `(rank,)`.
            - Vh (torch.Tensor): Right singular vectors of shape `(rank, N)`.
    """
    if W.dim() != 2:
        raise ValueError(f"randomized_svd expects a 2D matrix, got shape {W.shape}")

    m, n = W.shape
    eff_rank = max(1, min(int(rank), m, n))
    target_k = min(eff_rank + 10, m, n)

    if m == 0 or n == 0:
        device, dtype = W.device, W.dtype
        return (
            torch.empty(m, eff_rank, device=device, dtype=dtype),
            torch.empty(eff_rank, device=device, dtype=dtype),
            torch.empty(eff_rank, n, device=device, dtype=dtype),
        )

    omega = torch.randn(n, target_k, device=W.device, dtype=W.dtype)
    Y = W @ omega

    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(Y)
        Z = W.T @ Q
        Q_z, _ = torch.linalg.qr(Z)
        Y = W @ Q_z

    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ W
    U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_b

    return U[:, :eff_rank], S[:eff_rank], Vh[:eff_rank, :]


# Backward-compatibility alias
_randomized_svd = randomized_svd

__all__ = ["randomized_svd", "_randomized_svd"]
