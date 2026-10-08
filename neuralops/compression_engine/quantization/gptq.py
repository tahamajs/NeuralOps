"""Production-grade GPTQ (Exact Second-Order Hessian-Guided Quantization)."""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["GPTQQuantizer", "quantize_model_gptq"]


class GPTQQuantizer:
    """Second-order Hessian-guided post-training weight quantization with column error propagation."""

    def __init__(
        self,
        layer: nn.Linear,
        bits: int = 4,
        sym: bool = True,
        group_size: int = -1,
    ) -> None:
        self.layer = layer
        self.bits = bits
        self.sym = sym
        self.group_size = group_size
        self.columns = layer.weight.shape[1]
        self.rows = layer.weight.shape[0]
        self.H = torch.zeros((self.columns, self.columns), device=layer.weight.device, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, x: torch.Tensor) -> None:
        """Accumulates Hessian matrix H from activations: H += X^T @ X."""
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.to(dtype=torch.float32, device=self.layer.weight.device)
        self.H += torch.matmul(x.t(), x)
        self.nsamples += x.shape[0]

    def _quantize_column(self, W: torch.Tensor, H_inv: torch.Tensor, i: int) -> torch.Tensor:
        """Quantize column i and propagate error across subsequent unquantized columns."""
        w = W[:, i]
        # Quantization grid
        q_max = 2 ** (self.bits - 1) - 1 if self.sym else 2 ** self.bits - 1
        scale = (w.abs().max() / max(q_max, 1)).clamp(min=1e-8)
        q = torch.round(w / scale).clamp(-2 ** (self.bits - 1), q_max)
        w_q = q * scale

        # Compute error scaled by inverse Hessian diagonal entry
        h_ii = H_inv[i, i].clamp(min=1e-8)
        err = (w - w_q) / h_ii

        # Propagate error to remaining columns: W[:, i+1:] -= err * H_inv[i, i+1:]
        if i + 1 < self.columns:
            W[:, i + 1 :] -= err.unsqueeze(1) * H_inv[i, i + 1 :].unsqueeze(0)

        W[:, i] = w_q
        return W

    def quantize(self, damp: float = 0.01) -> torch.Tensor:
        """Runs exact column-by-column GPTQ quantization."""
        W = self.layer.weight.data.clone().to(torch.float32)
        H = self.H / max(self.nsamples, 1)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0

        # Damping to ensure numerical stability and positive-definiteness
        damp_val = damp * torch.mean(torch.diag(H))
        H[torch.eye(self.columns, dtype=torch.bool, device=H.device)] += damp_val

        try:
            H_inv = torch.linalg.inv(H)
        except Exception:
            H_inv = torch.linalg.pinv(H)

        if self.group_size > 0:
            for g in range(0, self.columns, self.group_size):
                end = min(g + self.group_size, self.columns)
                for i in range(g, end):
                    W = self._quantize_column(W, H_inv, i)
        else:
            for i in range(self.columns):
                W = self._quantize_column(W, H_inv, i)

        self.layer.weight.data.copy_(W.to(self.layer.weight.dtype))
        return W


def quantize_model_gptq(
    model: nn.Module,
    calibration_data: list[torch.Tensor],
    bits: int = 4,
    damp: float = 0.01,
    group_size: int = -1,
) -> nn.Module:
    """Quantizes all nn.Linear layers in model using GPTQ."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantizer = GPTQQuantizer(module, bits=bits, group_size=group_size)
            for batch in calibration_data:
                quantizer.add_batch(batch)
            quantizer.quantize(damp=damp)
    return model
