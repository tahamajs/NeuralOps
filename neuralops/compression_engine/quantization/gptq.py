#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
GPTQ: Accurate Post-Training Quantization for Large Language Models

Minimizes layer reconstruction error using second-order Taylor expansions of the loss and Cholesky decomposition of the Hessian.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.registry import COMPRESSOR_REGISTRY

__all__ = ["GPTQQuantizer", "GPTQCompressor", "quantize_model_gptq"]


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
        q_max = 2 ** (self.bits - 1) - 1 if self.sym else 2 ** self.bits - 1
        scale = (w.abs().max() / max(q_max, 1)).clamp(min=1e-8)
        q = torch.round(w / scale).clamp(-2 ** (self.bits - 1), q_max)
        w_q = q * scale

        h_ii = H_inv[i, i].clamp(min=1e-8)
        err = (w - w_q) / h_ii

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


@COMPRESSOR_REGISTRY.register("gptq_real", is_alias=True)
@COMPRESSOR_REGISTRY.register("gptq_compressor", is_alias=True)
class GPTQCompressor(BaseCompressor):
    """Unified BaseCompressor wrapper for GPTQ quantization with compressed-tensors support."""

    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        sym: bool = True,
        group_size: int = 128,
        damp: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="gptq", **kwargs)
        self.bits = bits
        self.sym = sym
        self.group_size = group_size
        self.damp = damp
        self.quantized_weights: dict[str, torch.Tensor] = {}

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Compresses all linear layers in the model using GPTQ."""
        stats = {"bits": self.bits, "group_size": self.group_size, "layers": []}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                quantizer = GPTQQuantizer(
                    module, bits=self.bits, sym=self.sym, group_size=self.group_size
                )
                quantizer.quantize(damp=self.damp)
                self.quantized_weights[f"{name}.weight" if name else "weight"] = module.weight.data.clone()
                if module.bias is not None:
                    self.quantized_weights[f"{name}.bias" if name else "bias"] = module.bias.data.clone()
                stats["layers"].append(name)
        if not stats["layers"] and isinstance(self.model, nn.Linear):
            quantizer = GPTQQuantizer(
                self.model, bits=self.bits, sym=self.sym, group_size=self.group_size
            )
            quantizer.quantize(damp=self.damp)
            self.quantized_weights["weight"] = self.model.weight.data.clone()
            if self.model.bias is not None:
                self.quantized_weights["bias"] = self.model.bias.data.clone()
            stats["layers"].append("root")
        return self.model, stats

    def compress_to_tensors(self, model: Optional[nn.Module] = None) -> dict[str, torch.Tensor]:
        """Returns compressed tensor dictionary compatible with compressed-tensors / safetensors."""
        m = model if model is not None else self.model
        tensors: dict[str, torch.Tensor] = {}
        for name, param in m.named_parameters():
            tensors[name] = param.data
        for name, buffer in m.named_buffers():
            tensors[name] = buffer.data
        return tensors

    def get_quantization_config(self) -> dict[str, Any]:
        """Returns quantization configuration compatible with compressed-tensors and vLLM."""
        format_str = f"W{self.bits}A16"
        return {
            "quant_method": "compressed-tensors",
            "format": format_str,
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.sym,
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": self.bits,
                        "type": "int",
                        "symmetric": self.sym,
                        "strategy": "group" if self.group_size > 0 else "channel",
                        "group_size": self.group_size,
                    },
                    "targets": ["Linear"],
                }
            },
        }


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
