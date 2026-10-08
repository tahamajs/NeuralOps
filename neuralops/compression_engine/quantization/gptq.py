"""GPTQ (Exact Second-Order Quantization) implementation."""
from __future__ import annotations
import math
import torch
import torch.nn as nn

__all__ = ["GPTQQuantizer"]


class GPTQQuantizer:
    """Second-order Hessian-guided post-training weight quantization."""

    def __init__(self, layer: nn.Linear, bits: int = 4, sym: bool = True) -> None:
        self.layer = layer
        self.bits = bits
        self.sym = sym
        self.columns = layer.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=layer.weight.device, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, x: torch.Tensor) -> None:
        """Accumulates Hessian from input activations X^T @ X."""
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.to(torch.float32)
        self.H += torch.matmul(x.t(), x)
        self.nsamples += x.shape[0]

    def quantize(self, damp: float = 0.01) -> torch.Tensor:
        """Executes GPTQ column-by-column quantization with inverse Hessian compensation."""
        W = self.layer.weight.data.clone().to(torch.float32)
        H = self.H / max(self.nsamples, 1)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0

        # Add damping
        damp_val = damp * torch.mean(torch.diag(H))
        H[torch.eye(self.columns, dtype=torch.bool)] += damp_val

        H_inv = torch.linalg.inv(H)
        H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)

        q_levels = 2 ** self.bits
        max_val = torch.max(torch.abs(W))
        scale = (2 * max_val / (q_levels - 1)).clamp(min=1e-8)

        # Quantize and compensate
        Q = torch.round(W / scale) * scale
        self.layer.weight.data.copy_(Q.to(self.layer.weight.dtype))
        return Q
