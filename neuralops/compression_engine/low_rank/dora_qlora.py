"""LoRA Merging, Extraction, and QLoRA / DoRA implementations."""
from __future__ import annotations
import math
import torch
import torch.nn as nn

__all__ = [
    "merge_lora_into_linear",
    "extract_lora_from_diff",
    "QLoRALinear",
    "DoRALinear",
]


def merge_lora_into_linear(
    base_linear: nn.Linear,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float = 1.0,
) -> None:
    """Merges LoRA matrices into base linear layer in-place: W = W + scaling * (B @ A)."""
    delta = (lora_b @ lora_a) * scaling
    base_linear.weight.data += delta.to(base_linear.weight.dtype)


def extract_lora_from_diff(
    diff: torch.Tensor,
    rank: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts LoRA A and B adapters from weight delta via truncated SVD."""
    U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    # B = U_r * sqrt(S_r), A = sqrt(S_r) * Vh_r
    sqrt_s = torch.diag(torch.sqrt(S_r))
    B = U_r @ sqrt_s
    A = sqrt_s @ Vh_r
    return A, B


class QLoRALinear(nn.Module):
    """Linear layer simulating 4-bit NormalFloat (NF4) base weights with FP32/FP16 LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / rank

        # Quantized base weight representation (simulated INT8/NF4)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = torch.matmul(x, self.base_weight.t())
        lora_out = torch.matmul(torch.matmul(x, self.lora_A.t()), self.lora_B.t()) * self.scaling
        return base_out + lora_out


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA): separates directional adaptation from magnitude."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / rank

        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        # Magnitude parameter m = ||W_0||_c
        norm_0 = torch.linalg.norm(self.weight.data, dim=1, keepdim=True)
        self.magnitude = nn.Parameter(norm_0.clone())

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full adapted directional matrix: V = W + scaling * (B @ A)
        V = self.weight + (self.lora_B @ self.lora_A) * self.scaling
        norm_v = torch.linalg.norm(V, dim=1, keepdim=True).clamp(min=1e-8)
        # Decomposed weight: W' = m * (V / ||V||)
        W_dora = self.magnitude * (V / norm_v)
        return torch.matmul(x, W_dora.t())
