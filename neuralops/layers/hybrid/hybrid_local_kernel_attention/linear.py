import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorisedLinear(nn.Module):
    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 384,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        energy: float = 0.95,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.svd_rank = self.rank
        self.energy = energy
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=bias)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.randn(out_features, lora_rank) * 1e-4)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
            self.lora_B.requires_grad = True

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
            S = torch.clamp(S, min=1e-6)
            target_energy = energy if energy is not None else self.energy
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
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
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
        orig_dtype = x.dtype
        x_down = x.to(self.down.weight.dtype)
        latent = self.down(x_down)
        latent_up = latent.to(self.up.weight.dtype)
        out = self.up(latent_up)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(orig_dtype) if self.lora_A.dtype != orig_dtype else self.lora_A
            b = self.lora_B.to(orig_dtype) if self.lora_B.dtype != orig_dtype else self.lora_B
            out = out.to(orig_dtype) + F.linear(F.linear(x, a), b) * self.lora_scaling
        return out.to(orig_dtype)
