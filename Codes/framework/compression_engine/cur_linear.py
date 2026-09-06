"""
CUR Linear Layer: low-rank cross-approximation using column and row skeleton selection.
W ≈ C @ U @ R
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_leverage_scores(W: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exact statistical leverage scores from the top-k singular vectors of W.
    W: shape (out_features, in_features)
    Returns:
        (col_scores, row_scores) normalized to valid probability distributions summing to 1.0.
    """
    m, n = W.shape
    k = max(1, min(rank, m, n))
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    # Row leverage scores: norm of rows of U_k
    row_scores = (U[:, :k] ** 2).sum(dim=1) / float(k)
    # Column leverage scores: norm of cols of V_k (rows of Vt_k)
    col_scores = (Vt[:k, :] ** 2).sum(dim=0) / float(k)
    # Normalize leverage scores
    row_scores = row_scores / row_scores.sum().clamp_min(1e-12)
    col_scores = col_scores / col_scores.sum().clamp_min(1e-12)
    return col_scores, row_scores


def compute_energy_scores(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Frobenius energy sampling probabilities for columns and rows.
    """
    frob_sq = (W.float() ** 2).sum().clamp_min(1e-12)
    col_scores = (W.float() ** 2).sum(dim=0) / frob_sq
    row_scores = (W.float() ** 2).sum(dim=1) / frob_sq
    return col_scores, row_scores


def select_indices(scores: torch.Tensor, k: int, deterministic: bool = True) -> torch.Tensor:
    """
    Select k indices based on scores (either deterministic top-k or multinomial sampling).
    """
    n = scores.numel()
    k = min(k, n)
    if deterministic:
        _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
        return idx
    else:
        probs = (scores / scores.sum().clamp_min(1e-12)).clamp_min(1e-8)
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, num_samples=k, replacement=False)
        return idx


def tikhonov_pinv(A: torch.Tensor, reg: float = 1e-3) -> torch.Tensor:
    """
    Computes Tikhonov regularized pseudo-inverse of matrix A: (A^T A + reg * I)^(-1) A^T
    or via SVD regularized inversion for numerical stability.
    """
    try:
        u, s, vt = torch.linalg.svd(A.float(), full_matrices=False)
        eps = reg * s.max().clamp_min(1e-6)
        s_inv = s / (s**2 + eps**2)
        return torch.matmul(vt.T, torch.matmul(torch.diag(s_inv), u.T)).to(dtype=A.dtype)
    except Exception:
        return torch.linalg.pinv(A.float(), rcond=reg).to(dtype=A.dtype)


class CURLinear(nn.Module):
    """
    Linear layer whose weight matrix is factorized via CUR cross-approximation:
    W ≈ C @ U @ R
    - C in R^{out_features x rank}: selected columns of W
    - U in R^{rank x rank}: Moore-Penrose pseudo-inverse core matrix
    - R in R^{rank x in_features}: selected rows of W

    Executes forward pass in factorized form:
    y = ((x @ R^T) @ U^T) @ C^T + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        method: str = "leverage",
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_skeletons: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.method = method
        self.lora_r = max(0, int(lora_r))
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / max(1, self.lora_r)) if self.lora_r > 0 else 1.0

        # C: (out_features, rank), U: (rank, rank), R: (rank, in_features)
        self.C = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(self.rank, self.rank, dtype=dtype, device=device))
        self.R = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.C, -bound, bound)
        nn.init.eye_(self.U)
        nn.init.uniform_(self.R, -bound, bound)

        if freeze_skeletons:
            self.C.requires_grad = False
            self.R.requires_grad = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_r, in_features, dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_r, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        W_core = torch.matmul(torch.matmul(self.C, self.U), self.R)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            W_core = W_core + (torch.matmul(self.lora_B, self.lora_A) * self.scaling)
        return W_core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward execution:
        x: (..., in_features)
        y = ((x @ R^T) @ U^T) @ C^T + bias
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Factorized forward pass:
        # x_r = x @ R^T: (K, rank)
        x_r = F.linear(x_flat, self.R)
        # x_ru = x_r @ U^T: (K, rank)
        x_ru = F.linear(x_r, self.U)
        # out = x_ru @ C^T + bias: (K, out_features)
        out = F.linear(x_ru, self.C, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize reconstructed weight matrix W = C @ U @ R."""
        return self.weight

    @classmethod
    def from_linear(
        cls,
        linear: Union[nn.Linear, Any],
        rank: int = 64,
        method: str = "leverage",
        deterministic: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_skeletons: bool = False,
    ) -> "CURLinear":
        """
        Construct CURLinear from existing nn.Linear or Conv1D using leverage scores or energy sampling.
        """
        target_mod = getattr(linear, "base_layer", getattr(linear, "linear", linear))
        raw_w = target_mod.weight.data
        bias = target_mod.bias.data if getattr(target_mod, "bias", None) is not None else None
        cls_name = target_mod.__class__.__name__.lower()
        wrapper_name = linear.__class__.__name__.lower()

        is_conv1d = "conv1d" in cls_name or "conv1d" in wrapper_name or hasattr(target_mod, "nf") or hasattr(linear, "nf")
        if is_conv1d:
            W = raw_w.t().contiguous()
        else:
            W = raw_w.contiguous()

        out_features, in_features = W.shape
        r = max(1, min(rank, out_features, in_features))

        cur_layer = cls(
            in_features=in_features,
            out_features=out_features,
            rank=r,
            bias=bias is not None,
            method=method,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            freeze_skeletons=freeze_skeletons,
            dtype=W.dtype,
            device=W.device,
        )

        W_float = W.float()
        if method == "leverage":
            col_scores, row_scores = compute_leverage_scores(W_float, rank=r)
        elif method == "energy":
            col_scores, row_scores = compute_energy_scores(W_float)
        else:
            col_scores = torch.ones(in_features, device=W.device)
            row_scores = torch.ones(out_features, device=W.device)

        col_idx = select_indices(col_scores, k=r, deterministic=deterministic)
        row_idx = select_indices(row_scores, k=r, deterministic=deterministic)

        # C = W[:, col_idx] (out_features, r)
        # R = W[row_idx, :] (r, in_features)
        C = W_float[:, col_idx]
        R = W_float[row_idx, :]

        # Optimal core matrix: U = C^+ @ W @ R^+
        # This is the exact least-squares solution argmin_U ||W - C U R||_F
        try:
            C_pinv = torch.linalg.pinv(C, rcond=1e-4)
            R_pinv = torch.linalg.pinv(R, rcond=1e-4)
            U = torch.matmul(torch.matmul(C_pinv, W_float), R_pinv)
        except Exception:
            W_int = W_float[row_idx][:, col_idx]
            U = torch.linalg.pinv(W_int, rcond=1e-3)

        cur_layer.C.data.copy_(C.to(dtype=W.dtype))
        cur_layer.U.data.copy_(U.to(dtype=W.dtype))
        cur_layer.R.data.copy_(R.to(dtype=W.dtype))

        if bias is not None and cur_layer.bias is not None:
            cur_layer.bias.data.copy_(bias)

        return cur_layer
