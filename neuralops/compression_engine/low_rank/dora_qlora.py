"""Production LoRA Merging, SVD Extraction, QLoRA, and DoRA modules."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "merge_lora_into_linear",
    "extract_lora_from_diff",
    "QuantizedLinear",
    "QLoRALinear",
    "DoRALinear",
]


def merge_lora_into_linear(
    linear: nn.Linear,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float = 1.0,
) -> nn.Linear:
    """Merge LoRA into base linear weight in-place: W += scaling * (B @ A)."""
    delta_W = (lora_B @ lora_A) * scaling
    linear.weight.data += delta_W.to(linear.weight.dtype)
    return linear


def extract_lora_from_diff(
    delta_W: torch.Tensor,
    rank: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract LoRA matrices (A, B) from weight delta using truncated SVD."""
    U, S, Vt = torch.linalg.svd(delta_W.to(torch.float32), full_matrices=False)
    r = min(rank, S.numel())
    A = Vt[:r]  # shape: [r, in_features]
    B = U[:, :r] * S[:r].unsqueeze(0)  # shape: [out_features, r]
    return A.to(delta_W.dtype), B.to(delta_W.dtype)


class QuantizedLinear(nn.Module):
    """Linear layer simulating symmetric INT4/INT8 quantized weights with dequantization."""

    def __init__(self, in_features: int, out_features: int, bits: int = 4, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.q_max = 2 ** (bits - 1) - 1
        self.q_min = -2 ** (bits - 1)

        raw_w = torch.randn(out_features, in_features)
        scale = raw_w.abs().amax(dim=-1, keepdim=True) / self.q_max
        q_w = torch.clamp(torch.round(raw_w / scale), self.q_min, self.q_max)

        self.weight = nn.Parameter(q_w, requires_grad=False)
        self.scales = nn.Parameter(scale, requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_deq = self.weight * self.scales
        return F.linear(x, w_deq, self.bias)


class QLoRALinear(nn.Module):
    """Production QLoRA layer with frozen quantized base weights and trainable low-rank adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bits: int = 4,
        lora_alpha: float = 16.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / max(rank, 1)

        # Frozen quantized base
        self.base = QuantizedLinear(in_features, out_features, bits=bits, bias=bias)
        self.base.weight.requires_grad = False
        self.base.scales.requires_grad = False

        # Trainable LoRA
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, rank) * 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA): separates directional adaptation from magnitude."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        lora_alpha: float = 16.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / max(rank, 1)

        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad = False

        # Magnitude parameter m = ||W_0||_c
        self.magnitude = nn.Parameter(self.base.weight.norm(dim=1, keepdim=True).clone())
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, rank) * 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Effective directional weight: V = W + scaling * (B @ A)
        W_dir = self.base.weight + (self.lora_B @ self.lora_A) * self.scaling
        W_dir = W_dir / W_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        W_eff = self.magnitude * W_dir
        return F.linear(x, W_eff, self.base.bias)
