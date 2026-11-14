#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Wavelet Multi-Resolution Compression Engine.

Compresses weight matrices using 2D discrete Haar wavelet decomposition,
preserving core low-frequency approximation subbands while thresholding high-frequency details.
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


def haar_2d_forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D Haar Wavelet Forward Decomposition."""
    # Ensure even dimensions
    h, w = x.shape
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    sub = x[:h_even, :w_even]

    # Row transform: [A+B, A-B] / sqrt(2)
    x00 = sub[0::2, 0::2]
    x01 = sub[0::2, 1::2]
    x10 = sub[1::2, 0::2]
    x11 = sub[1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def haar_2d_inverse(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
    target_shape: tuple[int, int],
) -> torch.Tensor:
    """2D Haar Wavelet Inverse Reconstruction."""
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5

    h_sub, w_sub = ll.shape
    out = torch.zeros(target_shape, dtype=ll.dtype, device=ll.device)
    out[0 : 2 * h_sub : 2, 0 : 2 * w_sub : 2] = x00
    out[0 : 2 * h_sub : 2, 1 : 2 * w_sub : 2] = x01
    out[1 : 2 * h_sub : 2, 0 : 2 * w_sub : 2] = x10
    out[1 : 2 * h_sub : 2, 1 : 2 * w_sub : 2] = x11
    return out


class WaveletLinear(nn.Module):
    """Linear layer compressed via 2D Haar Wavelet Decomposition."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float = 0.05,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = float(threshold)

        factory_kwargs = {"device": device, "dtype": dtype}
        h_even = out_features - (out_features % 2)
        w_even = in_features - (in_features % 2)
        h_sub, w_sub = h_even // 2, w_even // 2

        self.sub_shape = (h_sub, w_sub)
        self.ll = nn.Parameter(torch.zeros((h_sub, w_sub), **factory_kwargs))
        self.lh = nn.Parameter(torch.zeros((h_sub, w_sub), **factory_kwargs))
        self.hl = nn.Parameter(torch.zeros((h_sub, w_sub), **factory_kwargs))
        self.hh = nn.Parameter(torch.zeros((h_sub, w_sub), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        threshold: float = 0.05,
    ) -> WaveletLinear:
        """Create WaveletLinear from an existing linear layer."""
        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            threshold=threshold,
            bias=(linear.bias is not None),
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            w = linear.weight.data
            ll, lh, hl, hh = haar_2d_forward(w)
            mod.ll.data.copy_(ll)

            # Soft-threshold high frequency detail subbands
            for band, target in ((lh, mod.lh), (hl, mod.hl), (hh, mod.hh)):
                band_mask = band.abs() > threshold * band.abs().max()
                target.data.copy_(band * band_mask.to(band.dtype))

            if linear.bias is not None and mod.bias is not None:
                mod.bias.data.copy_(linear.bias.data)
        return mod

    def reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct the spatial weight matrix from wavelet subbands."""
        return haar_2d_inverse(self.ll, self.lh, self.hl, self.hh, (self.out_features, self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using reconstructed wavelet weight."""
        w_eff = self.reconstruct_weight().to(dtype=x.dtype)
        return F.linear(x, w_eff, self.bias)


@COMPRESSOR_REGISTRY.register("wavelet_compression")
@COMPRESSOR_REGISTRY.register("wavelet", is_alias=True)
class WaveletCompressor(BaseCompressor):
    """Wavelet Multi-Resolution Compression Engine."""

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.05,
        max_error: float = 0.70,
        fallback_to_original: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="wavelet_compression", **kwargs)
        self.threshold = float(threshold)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        orig_params = self._count_params(self.model)
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, WaveletLinear):
                orig_w = module.weight.data
                wv_layer = WaveletLinear.from_linear(module, threshold=self.threshold)
                recon_w = wv_layer.reconstruct_weight().to(orig_w.device)
                rel_err = (torch.norm(orig_w - recon_w) / torch.norm(orig_w).clamp(min=1e-8)).item()

                if rel_err > self.max_error and self.fallback_to_original:
                    self.fallback_layers.append(name)
                    continue

                replace_module(self.model, name, wv_layer)
                self.compressed_layers.append(name)

        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "num_compressed": len(self.compressed_layers),
                "wavelet_threshold": self.threshold,
            }
        )
        self.stats = stats
        return self.model, stats


__all__ = ["WaveletLinear", "WaveletCompressor", "haar_2d_forward", "haar_2d_inverse"]
