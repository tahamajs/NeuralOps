#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Quantized Low-Rank Adaptation (QLoRA)

Provides quantized base linear projections with floating-point low-rank adapters,
enabling memory-efficient fine-tuning without degrading baseline fidelity.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, zeros_

from .registry import register_method

logger = logging.getLogger(__name__)


class QuantizedLinear(nn.Module):
    """Uniformly quantized linear layer with frozen integer weights and dequantization scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        block_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))

        # Quantized weight and scale buffers (non-trainable)
        self.register_buffer("weight_q", torch.zeros(out_features, in_features, dtype=torch.int8))
        num_blocks = math.ceil((out_features * in_features) / block_size)
        self.register_buffer("scales", torch.ones(num_blocks, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 4, block_size: int = 64) -> QuantizedLinear:
        """Construct a quantized linear layer from an existing nn.Linear module."""
        qlin = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            block_size=block_size,
        )
        with torch.no_grad():
            w = linear.weight.float()
            flat_w = w.view(-1)
            pad_size = (block_size - (flat_w.numel() % block_size)) % block_size
            if pad_size > 0:
                padded = F.pad(flat_w, (0, pad_size))
            else:
                padded = flat_w

            blocks = padded.view(-1, block_size)
            scales = (blocks.abs().max(dim=1, keepdim=True).values / qlin.qmax).clamp_min(1e-8)
            q_blocks = torch.clamp(torch.round(blocks / scales), qlin.qmin, qlin.qmax).to(torch.int8)

            unpadded_q = q_blocks.view(-1)[: flat_w.numel()]
            qlin.weight_q.copy_(unpadded_q.view_as(w))
            qlin.scales.copy_(scales.squeeze(1))

            if linear.bias is not None:
                qlin.bias.copy_(linear.bias)

        return qlin

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize 4-bit / 8-bit weight buffer back to float tensor."""
        flat_q = self.weight_q.float().view(-1)
        pad_size = (self.block_size - (flat_q.numel() % self.block_size)) % self.block_size
        if pad_size > 0:
            padded = F.pad(flat_q, (0, pad_size))
        else:
            padded = flat_q

        blocks = padded.view(-1, self.block_size)
        scales = self.scales.unsqueeze(1)
        deq_blocks = blocks * scales
        deq = deq_blocks.view(-1)[: flat_q.numel()].view_as(self.weight_q)
        return deq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly weight dequantization."""
        weight = self.dequantize_weight().to(dtype=x.dtype, device=x.device)
        bias = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, weight, bias)


class QLoRALinear(nn.Module):
    """QLoRA layer wrapping a QuantizedLinear base with low-rank adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bits: int = 4,
        bias: bool = False,
        r: int | None = None,
        block_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(r if r is not None else rank)
        self.r = self.rank
        self.alpha = float(alpha)
        self.scaling = (self.alpha / self.rank) if self.rank > 0 else 1.0

        self.base = QuantizedLinear(in_features, out_features, bias=bias, bits=bits, block_size=block_size)
        if self.rank > 0:
            self.lora_A = nn.Linear(in_features, self.rank, bias=False)
            self.lora_B = nn.Linear(self.rank, out_features, bias=False)
            kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            zeros_(self.lora_B.weight)
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bits: int = 4,
        r: int | None = None,
        block_size: int = 64,
    ) -> QLoRALinear:
        """Construct QLoRALinear from an existing linear layer."""
        effective_r = r if r is not None else rank
        qlora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=effective_r,
            alpha=alpha,
            dropout=dropout,
            bits=bits,
            bias=linear.bias is not None,
            block_size=block_size,
        )
        qlora.base = QuantizedLinear.from_linear(linear, bits=bits, block_size=block_size)
        return qlora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining quantized base and adapter outputs."""
        base_out = self.base(x)
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            adapter_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            return base_out + adapter_out
        return base_out


@register_method("qlora")
def apply_qlora(model: nn.Module, config: Any) -> None:
    """Inject QLoRA layers into model projections."""
    rank = getattr(config, "lora_r", getattr(config, "r", 4))
    alpha = getattr(config, "lora_alpha", getattr(config, "alpha", 16.0))
    dropout = getattr(config, "lora_dropout", getattr(config, "dropout", 0.0))
    bits = getattr(config, "bits", getattr(config, "qlora_bits", 4))

    target_substrings = [
        "query",
        "key",
        "value",
        "dense",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "c_attn",
        "c_proj",
    ]
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(ts in name for ts in target_substrings):
            parent_name = ".".join(name.split(".")[:-1]) or ""
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            try:
                new_mod = QLoRALinear.from_linear(module, rank=rank, alpha=alpha, dropout=dropout, bits=bits)
                setattr(parent, child_name, new_mod)
            except Exception as e:
                logger.warning("Failed to inject QLoRA into %s: %s", name, e)


__all__ = [
    "QuantizedLinear",
    "QLoRALinear",
    "apply_qlora",
]
