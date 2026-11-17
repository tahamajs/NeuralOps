import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorisedLinear(nn.Module):
    is_custom_attention: bool = True
    r"""
    SVD-factorized linear layer with energy-based rank selection and LoRA adapters.

    Forward: y = up(down(x)) + (B @ A)(x) * (lora_alpha / lora_rank)
    Base SVD weights (down, up) are frozen; only LoRA (A, B) is trainable.

    Note on naming: `lora_A` has shape (rank, in) and `lora_B` has shape (out, rank),
    matching the PEFT convention. This differs from AegisAttentionV6 which uses the
    opposite convention.
    """

    def __init__(
        self,
        in_features_or_weight: Any,
        out_features_or_bias: Any | None = None,
        energy: float = 0.95,
        min_rank: int = 4,
        max_rank: int = 384,
        rank: int | None = None,
        bias: bool = True,
        lora_rank: int = 64,
        lora_alpha: float = 32.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.energy = float(energy)
        self.min_rank = int(min_rank)
        self.max_rank = int(max_rank)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0

        if isinstance(in_features_or_weight, torch.Tensor):
            W = in_features_or_weight
            b = out_features_or_bias if isinstance(out_features_or_bias, torch.Tensor) else None
            out_dim, in_dim = W.shape
            self.in_features = in_dim
            self.out_features = out_dim
            self.rank = min(self.max_rank, max(self.min_rank, min(in_dim, out_dim)))
            device = W.device
            dtype = W.dtype

            self.down = nn.Linear(in_dim, self.rank, bias=False, device=device, dtype=dtype)
            self.up = nn.Linear(
                self.rank,
                out_dim,
                bias=(b is not None or bias),
                device=device,
                dtype=dtype,
            )
            if self.lora_rank > 0:
                self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_dim, device=device, dtype=dtype))
                self.lora_B = nn.Parameter(torch.randn(out_dim, self.lora_rank, device=device, dtype=dtype) * 1e-4)
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
            else:
                self.register_parameter("lora_A", None)
                self.register_parameter("lora_B", None)

            self.decompose_from_weight(W, b, self.energy)

        else:
            in_dim = int(in_features_or_weight)
            out_dim = int(out_features_or_bias) if out_features_or_bias is not None else in_dim
            self.in_features = in_dim
            self.out_features = out_dim
            self.rank = int(rank) if rank is not None else min(self.max_rank, max(self.min_rank, min(in_dim, out_dim)))

            self.down = nn.Linear(in_dim, self.rank, bias=False)
            self.up = nn.Linear(self.rank, out_dim, bias=bias)
            if self.lora_rank > 0:
                self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_dim))
                self.lora_B = nn.Parameter(torch.randn(out_dim, self.lora_rank) * 1e-4)
            else:
                self.register_parameter("lora_A", None)
                self.register_parameter("lora_B", None)

            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        if self.lora_rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        energy: float | None = None,
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
                padded = torch.zeros(self.out_features, self.in_features, device=W.device, dtype=W.dtype)
                r = min(self.out_features, W.shape[0])
                c = min(self.in_features, W.shape[1])
                padded[:r, :c] = W[:r, :c]
                W = padded

            dev = W.device
            W_cpu = W.detach().cpu().float()
            U_cpu, S_cpu, Vt_cpu = torch.linalg.svd(W_cpu, full_matrices=False)
            U = U_cpu.to(dev)
            S = S_cpu.to(dev)
            Vt = Vt_cpu.to(dev)

            target_energy = float(energy) if energy is not None else self.energy
            total = (S**2).sum()
            cum = torch.cumsum(S**2, dim=0) / (total + 1e-5)
            calc_rank = int((cum < target_energy).sum().item()) + 1

            r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
            r = max(1, r)

            A = U[:, :r] * S[:r].unsqueeze(0)
            B = Vt[:r, :]

            self.down.weight.data.zero_()
            self.up.weight.data.zero_()
            self.down.weight.data[:r, :] = B.to(self.down.weight.dtype)
            self.up.weight.data[:, :r] = A.to(self.up.weight.dtype)

            if bias is not None and self.up.bias is not None:
                n = min(len(self.up.bias), len(bias))
                self.up.bias.data[:n].copy_(bias[:n].to(self.up.bias.dtype))
            elif self.up.bias is not None:
                self.up.bias.data.zero_()

            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = True
        self.up.weight.requires_grad = True
        if self.up.bias is not None:
            self.up.bias.requires_grad = True
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    @property
    def weight(self) -> torch.Tensor:
        base_w = torch.matmul(self.up.weight, self.down.weight)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            return base_w + (self.lora_B @ self.lora_A) * self.lora_scaling
        return base_w

    @property
    def bias(self) -> torch.Tensor | None:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out
