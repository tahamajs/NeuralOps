#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FFT Spectral Compression Engine.

Compresses weight matrices by transforming parameters into the 2D frequency domain,
truncating high-frequency spectral components, and preserving dominant energy modes.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class FFTLinear(nn.Module):
    """
    Linear layer retaining truncated 2D spectral components in frequency domain.
    Reconstructs the effective weight using 2D inverse real FFT.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        keep_ratio: float = 0.5,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.keep_ratio = float(keep_ratio)

        factory_kwargs = {"device": device, "dtype": dtype}
        # In frequency domain for (out_features, in_features), rfft2 produces shape (out_features, in_features // 2 + 1)
        freq_cols = in_features // 2 + 1
        self.freq_rows = max(1, int(out_features * self.keep_ratio))
        self.freq_cols = max(1, int(freq_cols * self.keep_ratio))

        self.freq_weight = nn.Parameter(
            torch.zeros((self.freq_rows, self.freq_cols), dtype=torch.complex64, device=device)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        keep_ratio: float = 0.5,
    ) -> FFTLinear:
        """Create FFTLinear from an existing linear layer by spectral truncation."""
        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            keep_ratio=keep_ratio,
            bias=(linear.bias is not None),
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            w = linear.weight.data.float()
            w_fft = torch.fft.rfft2(w)
            mod.freq_weight.data.copy_(w_fft[: mod.freq_rows, : mod.freq_cols])
            if linear.bias is not None and mod.bias is not None:
                mod.bias.data.copy_(linear.bias.data)
        return mod

    def reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct the spatial weight matrix from truncated frequency components."""
        full_fft = torch.zeros(
            (self.out_features, self.in_features // 2 + 1),
            dtype=torch.complex64,
            device=self.freq_weight.device,
        )
        full_fft[: self.freq_rows, : self.freq_cols] = self.freq_weight
        recon = torch.fft.irfft2(full_fft, s=(self.out_features, self.in_features))
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using dynamically reconstructed spectral weight."""
        w_eff = self.reconstruct_weight().to(dtype=x.dtype)
        return F.linear(x, w_eff, self.bias)


@COMPRESSOR_REGISTRY.register("fft_compression")
@COMPRESSOR_REGISTRY.register("fft", is_alias=True)
class FFTCompressor(BaseCompressor):
    """FFT Spectral Compression Engine."""

    def __init__(
        self,
        model: nn.Module,
        keep_ratio: float = 0.5,
        max_error: float = 0.70,
        fallback_to_original: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="fft_compression", **kwargs)
        self.keep_ratio = float(keep_ratio)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        orig_params = self._count_params(self.model)
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, FFTLinear):
                orig_w = module.weight.data
                fft_layer = FFTLinear.from_linear(module, keep_ratio=self.keep_ratio)
                recon_w = fft_layer.reconstruct_weight().to(orig_w.device)
                rel_err = (torch.norm(orig_w - recon_w) / torch.norm(orig_w).clamp(min=1e-8)).item()

                if rel_err > self.max_error and self.fallback_to_original:
                    self.fallback_layers.append(name)
                    continue

                replace_module(self.model, name, fft_layer)
                self.compressed_layers.append(name)

        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "num_compressed": len(self.compressed_layers),
                "keep_ratio": self.keep_ratio,
            }
        )
        self.stats = stats
        return self.model, stats


__all__ = ["FFTLinear", "FFTCompressor"]
