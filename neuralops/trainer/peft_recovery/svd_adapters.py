"""
SVD-Native Low-Rank Recovery Adapters: PiSSA (Principal Singular values and Singular vectors Adaptation)
and DoRA (Weight-Decomposed Low-Rank Adaptation).
"""

from __future__ import annotations

import math
from typing import Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PiSSALinear(nn.Module):
    """
    PiSSA: Principal Singular values and Singular vectors Adaptation.
    Initializes trainable low-rank adapters (B, A) from top-r SVD components of W:
        W = U @ S @ V^T
        B_init = U[:, :r] @ S[:r, :r]^(1/2)
        A_init = S[:r, :r]^(1/2) @ V[:, :r]^T
        W_residual = W - B_init @ A_init
    At step 0:
        W_residual + B @ A == W (exact machine precision identity).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = scaling

        # Decompose base weight in FP32
        w = base_layer.weight.data.float()
        if hasattr(base_layer, "weight") and base_layer.__class__.__name__ == "Conv1D":
            w = w.t().contiguous()

        r = min(self.rank, self.in_features, self.out_features)
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)

        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]

        sqrt_S = torch.diag(torch.sqrt(S_r))
        B_data = torch.matmul(U_r, sqrt_S)
        A_data = torch.matmul(sqrt_S, Vh_r)

        # Frozen residual matrix
        w_res = w - torch.matmul(B_data, A_data)
        self.register_buffer("weight_residual", w_res.to(base_layer.weight.dtype))

        # Trainable low-rank factors
        self.lora_A = nn.Parameter(A_data.to(base_layer.weight.dtype))
        self.lora_B = nn.Parameter(B_data.to(base_layer.weight.dtype))

        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.data.clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base residual forward + LoRA delta
        res = F.linear(x, self.weight_residual, self.bias)
        adapter = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return res + adapter

    def merge_and_unload(self) -> nn.Linear:
        """Folds B @ A back into weight_residual, returning a standard nn.Linear."""
        merged_w = self.weight_residual + (self.lora_B @ self.lora_A) * self.scaling
        lin = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight_residual.device,
            dtype=self.weight_residual.dtype,
        )
        lin.weight.data.copy_(merged_w)
        if self.bias is not None:
            lin.bias.data.copy_(self.bias)
        return lin


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA)."""

    def __init__(self, base_layer: nn.Linear, rank: int = 16, lora_alpha: float = 32.0) -> None:
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = lora_alpha / rank

        self.register_buffer("weight_base", base_layer.weight.data.clone())
        self.magnitude = nn.Parameter(self.weight_base.norm(p=2, dim=1, keepdim=True))

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.data.clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_weight = (self.lora_B @ self.lora_A) * self.scaling
        full_w = self.weight_base + d_weight
        w_norm = full_w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        w_effective = self.magnitude * (full_w / w_norm)
        return F.linear(x, w_effective, self.bias)


def inject_pissa_adapters(
    model: nn.Module,
    rank: int = 16,
    target_modules: Optional[List[str]] = None,
) -> int:
    """Replaces target linear layers in model with PiSSALinear adapters in-place."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "q_lin", "k_lin", "v_lin"]

    injected = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if any(t in child_name for t in target_modules) and isinstance(child, nn.Linear):
                adapter = PiSSALinear(child, rank=rank)
                setattr(module, child_name, adapter)
                injected += 1
    return injected


__all__ = ["PiSSALinear", "DoRALinear", "inject_pissa_adapters"]
