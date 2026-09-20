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
    is_custom_attention: bool = True
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
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

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

        requires_grad = not (self.lora_r > 0)
        if self.can_decompose:
            # A: (m1, n1), B: (m2, n2)
            self.A = nn.Parameter(torch.empty(self.m1, self.n1, dtype=dtype, device=device), requires_grad=requires_grad)
            self.B = nn.Parameter(torch.empty(self.m2, self.n2, dtype=dtype, device=device), requires_grad=requires_grad)

            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1.0
            nn.init.uniform_(self.A, -bound, bound)
            nn.init.uniform_(self.B, -bound, bound)

            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device), requires_grad=requires_grad)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter("bias", None)
        else:
            # Low-rank SVD fallback for prime or non-factorable dimensions
            effective_rank = rank or max(1, min(in_features, out_features) // 4)
            self.rank = min(effective_rank, min(in_features, out_features))
            self.U = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device), requires_grad=requires_grad)
            self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device), requires_grad=requires_grad)
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device), requires_grad=requires_grad)
            else:
                self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=torch.float32, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self._cached_weight = None

    def _invalidate_cache(self) -> None:
        self._cached_weight = None

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        if self._cached_weight is None:
            self._cached_weight = self.get_weight()
        return self._cached_weight

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        if self.can_decompose:
            return torch.kron(self.A, self.B)
        else:
            return torch.matmul(self.U, self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast forward pass using efficient tensor contractions with optional LoRA adapter.
        """
        if self.can_decompose:
            orig_shape = x.shape
            x_flat = x.reshape(-1, self.in_features)  # (K, n1 * n2)
            K = x_flat.size(0)

            # Step 1: contract with B over n2 dimension
            x_b = torch.matmul(x_flat.reshape(K * self.n1, self.n2), self.B.t().contiguous())  # (K * n1, m2)

            # Step 2: transpose to (K, m2, n1) -> (K * m2, n1) @ A^T (n1, m1) -> (K * m2, m1)
            x_b = x_b.reshape(K, self.n1, self.m2).transpose(1, 2).contiguous().reshape(K * self.m2, self.n1)
            x_a = torch.matmul(x_b, self.A.t().contiguous())  # (K * m2, m1)

            # Step 3: transpose back to (K, m1, m2) -> (K, m1 * m2)
            out = x_a.reshape(K, self.m2, self.m1).transpose(1, 2).contiguous().reshape(K, self.out_features)

            if self.bias is not None:
                out = out + self.bias

            out = out.reshape(*orig_shape[:-1], self.out_features)
        else:
            # Low-rank SVD fallback: y = (x @ V^T) @ U^T + bias
            out = F.linear(F.linear(x, self.V), self.U, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x.float(), self.lora_A), self.lora_B
            ).to(out.dtype)

        return out

    def _fit_svd(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        rank: Optional[int] = None,
        max_error: float = 0.20,
    ) -> None:
        """Fit low-rank SVD factors ensuring parameter reduction."""
        in_f = self.in_features
        out_f = self.out_features
        target_dtype = weight.dtype
        target_device = weight.device
        W_norm = torch.norm(weight, p="fro").item()

        # Parameter reduction constraint: r * (in + out) < in * out
        max_allowed_rank = max(1, min(in_f, out_f, int(0.75 * (in_f * out_f) / (in_f + out_f))))
        base_r = rank if rank is not None and rank > 0 else max(1, min(in_f, out_f) // 4)
        base_r = min(base_r, max_allowed_rank)

        try:
            U, S, Vt = torch.linalg.svd(weight.float(), full_matrices=False)
            best_r = base_r
            candidate_ranks = [base_r]
            if base_r * 2 <= max_allowed_rank:
                candidate_ranks.append(base_r * 2)
            if base_r * 4 <= max_allowed_rank:
                candidate_ranks.append(base_r * 4)

            for cand_r in candidate_ranks:
                U_r = U[:, :cand_r] * torch.sqrt(S[:cand_r].unsqueeze(0))
                V_r = torch.sqrt(S[:cand_r].unsqueeze(1)) * Vt[:cand_r, :]
                W_cand = torch.matmul(U_r, V_r)
                err = float(torch.norm(weight.float() - W_cand, p="fro").item() / max(W_norm, 1e-12))
                best_r = cand_r
                if err <= max_error:
                    break

            U_r = U[:, :best_r] * torch.sqrt(S[:best_r].unsqueeze(0))
            V_r = torch.sqrt(S[:best_r].unsqueeze(1)) * Vt[:best_r, :]
        except Exception:
            best_r = base_r
            U_r = torch.zeros(out_f, best_r, device=target_device, dtype=torch.float32)
            V_r = torch.zeros(best_r, in_f, device=target_device, dtype=torch.float32)

        self.can_decompose = False
        if hasattr(self, "A"):
            del self.A
        if hasattr(self, "B"):
            del self.B

        self.rank = best_r
        self.U = nn.Parameter(U_r.to(dtype=target_dtype, device=target_device))
        self.V = nn.Parameter(V_r.to(dtype=target_dtype, device=target_device))
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().to(dtype=target_dtype, device=target_device))
        elif not hasattr(self, "bias") or self.bias is None:
            self.register_parameter("bias", None)
        self._invalidate_cache()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: Optional[int] = None,
        max_error: float = 0.20,
        m1: Optional[int] = None,
        n1: Optional[int] = None,
        m2: Optional[int] = None,
        n2: Optional[int] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> "KroneckerLinear":
        """
        Decompose a pretrained nn.Linear, Conv1D, or LoRALinear layer into Kronecker factors A and B
        using Nearest Kronecker Product (Van Loan & Pitsianis 1993), falling back to SVD
        if error exceeds max_error or dimensions do not factorize.
        """
        target_mod = getattr(linear, "base_layer", linear)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data
        if is_conv1d:
            in_features = raw_w.shape[0]
            out_features = raw_w.shape[1]
            weight = raw_w.t().contiguous().float()
        else:
            out_features = raw_w.shape[0]
            in_features = raw_w.shape[1]
            weight = raw_w.float()

        bias = target_mod.bias.data.clone() if getattr(target_mod, "bias", None) is not None else None
        target_dtype = raw_w.dtype
        target_device = raw_w.device

        # Check for existing LoRA adapters on the layer
        existing_lora_A = getattr(linear, "lora_A", None)
        existing_lora_B = getattr(linear, "lora_B", None)
        effective_lora_r = lora_r
        effective_lora_alpha = lora_alpha
        if existing_lora_A is not None and existing_lora_B is not None:
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            effective_lora_r = w_A.shape[0]
            if hasattr(linear, "scaling") and linear.scaling:
                effective_lora_alpha = linear.scaling * effective_lora_r
            elif hasattr(linear, "lora_alpha"):
                effective_lora_alpha = linear.lora_alpha

        kron_layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            rank=rank,
            m1=m1,
            n1=n1,
            m2=m2,
            n2=n2,
            dtype=target_dtype,
            device=target_device,
            lora_r=effective_lora_r,
            lora_alpha=effective_lora_alpha,
        )

        if existing_lora_A is not None and existing_lora_B is not None:
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            w_B = getattr(existing_lora_B, "weight", existing_lora_B)
            if hasattr(w_A, "data") and hasattr(w_B, "data") and kron_layer.lora_A is not None and kron_layer.lora_B is not None:
                kron_layer.lora_A.data.copy_(w_A.data)
                kron_layer.lora_B.data.copy_(w_B.data)
                kron_layer.lora_A.requires_grad = True
                kron_layer.lora_B.requires_grad = True

        W_norm = torch.norm(weight, p="fro").item()

        if kron_layer.can_decompose:
            m1, m2 = kron_layer.m1, kron_layer.m2
            n1, n2 = kron_layer.n1, kron_layer.n2

            try:
                # Rearrange W (m1*m2, n1*n2) -> (m1, m2, n1, n2) -> (m1, n1, m2, n2) -> (m1*n1, m2*n2)
                W_tensor = weight.view(m1, m2, n1, n2).permute(0, 2, 1, 3).contiguous()
                W_reordered = W_tensor.view(m1 * n1, m2 * n2).float()

                # Optimal Rank-1 SVD approximation of reordered matrix
                U, S, Vt = torch.linalg.svd(W_reordered, full_matrices=False)
                sigma1 = S[0].item() if S.numel() > 0 else 1.0
                scale = math.sqrt(max(sigma1, 1e-12))

                u1 = U[:, 0] * scale
                v1 = Vt[0, :] * scale

                kron_layer.A.data.copy_(u1.view(m1, n1).to(dtype=target_dtype, device=target_device))
                kron_layer.B.data.copy_(v1.view(m2, n2).to(dtype=target_dtype, device=target_device))

                # Check reconstruction error
                W_recon = torch.kron(kron_layer.A, kron_layer.B).float()
                err = float(torch.norm(weight - W_recon, p="fro").item() / max(W_norm, 1e-12))
            except Exception:
                err = 1.0

            if err > max_error:
                kron_layer._fit_svd(weight, bias, rank=rank, max_error=max_error)
        else:
            kron_layer._fit_svd(weight, bias, rank=rank, max_error=max_error)

        if bias is not None and kron_layer.bias is not None:
            kron_layer.bias.data.copy_(bias.to(dtype=target_dtype, device=target_device))

        kron_layer._invalidate_cache()
        return kron_layer
