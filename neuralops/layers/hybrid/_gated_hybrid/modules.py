#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Sub-modules, factorized linear projections, and kernel feature maps for GatedHybridKernelAttention."""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.amp_utils import maybe_half
from neuralops.layers.hybrid._gated_hybrid.config import _clamp_to_dtype

logger = logging.getLogger(__name__)

class FactorisedLinear(nn.Module):
    """
    SVD-factorized linear layer + LoRA adapters.

    Structure:
        y = (x · V · diag(S) · Uᵀ) + bias + LoRA(x) · scaling

    - U, S, V: frozen Parameters (SVD of the original weight).
    - lora_A, lora_B: trainable LoRA tensors.
    - bias: stored on the output side.

    Compatible with the following call patterns:
        FactorisedLinear(in_features, out_features, svd_rank, lora_rank)
        FactorisedLinear(in_features, out_features, svd_rank, lora_rank,
                         lora_alpha=32.0, bias=True)
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int = 192,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        bias: bool = True,
        energy: float = 0.95,
        min_rank: int = 4,
        max_rank: int = 384,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.in_features: int = int(in_features)
        self.out_features: int = int(out_features)
        self.svd_rank: int = int(min(svd_rank, self.in_features, self.out_features))
        self.lora_rank: int = int(lora_rank)
        self.lora_alpha: float = float(lora_alpha)

        self.lora_scaling: float = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0

        self.energy: float = float(energy)
        self.min_rank: int = int(min_rank)
        self.max_rank: int = int(max_rank)

        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.U = nn.Parameter(
            torch.zeros(self.out_features, self.svd_rank, **factory_kwargs),
            requires_grad=False,
        )
        self.S = nn.Parameter(
            torch.ones(self.svd_rank, **factory_kwargs),
            requires_grad=False,
        )
        self.V = nn.Parameter(
            torch.zeros(self.in_features, self.svd_rank, **factory_kwargs),
            requires_grad=False,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, **factory_kwargs),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features, **factory_kwargs))
            self.lora_B = nn.Parameter(torch.randn(self.out_features, self.lora_rank, **factory_kwargs) * 1e-4)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
                rep = self.out_features // W.shape[0]
                W = W.repeat(rep, 1)
                if bias is not None:
                    bias = bias.repeat(rep)
            elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
                rep = W.shape[0] // self.out_features
                W = W.view(self.out_features, rep, -1).mean(dim=1)
                if bias is not None:
                    bias = bias.view(self.out_features, rep).mean(dim=1)
            elif W.shape[0] != self.out_features or W.shape[1] != self.in_features:
                padded = torch.zeros(
                    self.out_features,
                    self.in_features,
                    device=W.device,
                    dtype=W.dtype,
                )
                r = min(self.out_features, W.shape[0])
                c = min(self.in_features, W.shape[1])
                padded[:r, :c] = W[:r, :c]
                W = padded

            dev = W.device
            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            except Exception:
                U, S, Vt = torch.linalg.svd(W.detach().cpu().float(), full_matrices=False)
                U = U.to(dev)
                S = S.to(dev)
                Vt = Vt.to(dev)
            r = min(self.svd_rank, U.size(1), Vt.size(0))

            self.U.data.zero_()
            self.V.data.zero_()
            self.S.data.zero_()

            self.U.data[:, :r].copy_(U[:, :r].to(device=self.U.device, dtype=self.U.dtype))
            self.S.data[:r].copy_(S[:r].to(device=self.S.device, dtype=self.S.dtype))
            self.V.data[:, :r].copy_(Vt[:r, :].t().contiguous().to(device=self.V.device, dtype=self.V.dtype))

            if bias is not None:
                n = min(self.bias.numel(), bias.numel())
                self.bias.data[:n].copy_(bias[:n].to(device=self.bias.device, dtype=self.bias.dtype))

    def freeze_base_weights(self) -> None:
        """Freeze SVD base weights. Only LoRA tensors remain trainable."""
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        """Unfreeze SVD base weights (rarely used)."""
        self.U.requires_grad = True
        self.S.requires_grad = True
        self.V.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    @property
    def base_weight(self) -> torch.Tensor:
        """Reconstruct the SVD-only weight: U · diag(S) · Vᵀ."""
        return ((self.U.float() * self.S.float().unsqueeze(0)) @ self.V.float().t().contiguous()).to(self.U.dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Full effective weight: base + LoRA."""
        svd_w = (self.U.float() * self.S.float().unsqueeze(0)) @ self.V.float().t().contiguous()
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_w = self.lora_B.float() @ self.lora_A.float()
            return (svd_w + lora_w * self.lora_scaling).to(self.U.dtype)
        return svd_w.to(self.U.dtype)

    @property
    def A(self):
        return self.lora_B

    @property
    def B(self):
        return self.lora_A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()

        xV = torch.matmul(x_f, self.V.float())
        xV_S = xV * self.S.float().unsqueeze(0).unsqueeze(0)
        out = torch.matmul(xV_S, self.U.float().t().contiguous())
        if self.bias is not None:
            out = out + self.bias.float()

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            xa = torch.matmul(x_f, self.lora_A.float().t().contiguous())
            lora_out = torch.matmul(xa, self.lora_B.float().t().contiguous())
            out = out + lora_out * self.lora_scaling

        return _clamp_to_dtype(out, orig_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"svd_rank={self.svd_rank}, "
            f"lora_rank={self.lora_rank}, "
            f"lora_alpha={self.lora_alpha}, "
            f"lora_scaling={self.lora_scaling:.4f}"
        )


class StableKernelFeatureMap(nn.Module):
    """FAVOR+ orthogonal random features with FP16-safe scaling."""

    def __init__(
        self,
        head_dim: int,
        num_features: int,
        method: str = "orthogonal",
        seed: int | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.m = num_features
        self.scale = 1.0 / math.sqrt(num_features + 1e-5)
        self.register_buffer("proj", self._build_projection(method, seed))

    def _build_projection(self, method: str, seed: int | None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        d, m = self.head_dim, self.m
        if method != "orthogonal":
            return torch.randn(d, m) * math.sqrt(d)
        if m <= d:
            q, _ = torch.linalg.qr(torch.randn(d, m))
            return q * math.sqrt(d)
        num_blocks = (m + d - 1) // d
        blocks = [torch.linalg.qr(torch.randn(d, d))[0] for _ in range(num_blocks)]
        return torch.cat(blocks, dim=1)[:, :m] * math.sqrt(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_scaled = x.float() * (self.head_dim**-0.25)
            proj = self.proj.float() / math.sqrt(self.head_dim)
            phi = torch.matmul(x_scaled, proj).clamp(max=10.0)
            norm_sq = (x_scaled * x_scaled).sum(dim=-1, keepdim=True) * 0.5
            diff = torch.clamp(phi - norm_sq, min=-30.0, max=10.0)
            feat = self.scale * torch.exp(diff)
            return _clamp_to_dtype(feat, x.dtype)


class GatingModule(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, use_query, use_input):
        super().__init__()
        if use_query:
            self.lora_query_gate = nn.Linear(head_dim, 1, bias=False)
            nn.init.zeros_(self.lora_query_gate.weight)
        else:
            self.lora_query_gate = None

        if use_input:
            self.lora_input_gate = nn.Linear(hidden_size, num_heads, bias=True)
            nn.init.zeros_(self.lora_input_gate.weight)
            nn.init.constant_(self.lora_input_gate.bias, 2.5)
        else:
            self.lora_input_gate = None

    def forward(self, hidden_states, Q):
        q_gate = torch.sigmoid(self.lora_query_gate(Q)) if self.lora_query_gate is not None else None
        in_gate = None
        if self.lora_input_gate is not None:
            in_gate = torch.sigmoid(self.lora_input_gate(hidden_states)).permute(0, 2, 1).unsqueeze(-1)
        return q_gate, in_gate


class ContextFusionGate(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, init_bias=3.0):
        super().__init__()
        self.hidden_proj = (
            nn.Linear(hidden_size, num_heads * head_dim, bias=False) if hidden_size != num_heads * head_dim else None
        )
        if self.hidden_proj is not None:
            nn.init.zeros_(self.hidden_proj.weight)

        self.lora_proj = nn.Linear(3 * head_dim, 1, bias=True)
        nn.init.zeros_(self.lora_proj.weight)
        nn.init.constant_(self.lora_proj.bias, init_bias)

    def forward(self, hidden_states, local_out, global_out):
        B, h, N, d = local_out.shape
        h_exp = self.hidden_proj(hidden_states) if self.hidden_proj is not None else hidden_states
        h_per_head = h_exp.view(B, N, h, d).permute(0, 2, 1, 3)
        return torch.sigmoid(self.lora_proj(torch.cat([h_per_head, local_out, global_out], dim=-1)))

