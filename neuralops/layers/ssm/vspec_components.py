# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Component layers and routers for VSpec Attention."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

def _masked_mean(x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Computes mean over valid non-padded tokens."""
    if valid_mask is not None:
        m = valid_mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
    return x.mean(dim=1)


class VariationalSpectralGate(nn.Module):
    r"""Input-dependent variational gate over truncated singular values with masked pooling."""

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, rank * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias[:rank], 3.0)
        nn.init.constant_(self.net[-1].bias[rank:], -3.0)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        net_dtype = next(self.net.parameters()).dtype
        pooled = _masked_mean(x, valid_mask).to(net_dtype)
        stats = self.net(pooled)
        mu, log_sigma = stats.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=2.0)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(log_sigma)
        else:
            z = mu
        g = torch.sigmoid(z).to(x.dtype)
        return g, mu, log_sigma


class HeadGate(nn.Module):
    r"""Learned per-example head importance scalars with masked pooling."""

    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, n_heads))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 3.0)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        net_dtype = next(self.net.parameters()).dtype
        pooled = _masked_mean(x, valid_mask).to(net_dtype)
        return torch.sigmoid(self.net(pooled)).to(x.dtype)


class TokenRouter(nn.Module):
    r"""Top-k token selector based on learned saliency scores with guaranteed [CLS] retention."""

    def __init__(self, hidden_dim: int, keep_ratio: float = 0.75, min_tokens: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        is_decoder: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        net_dtype = next(self.net.parameters()).dtype
        scores = torch.sigmoid(self.net(x.to(net_dtype)).squeeze(-1))
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, -10000.0)

        if is_decoder and N > 1:
            causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask[0], -10000.0)

        scores[:, 0] = 1e4

        keep = min(
            N,
            max(
                1,
                (min(self.min_tokens, N) if self.min_tokens >= N else max(self.min_tokens, int(N * self.keep_ratio))),
            ),
        )
        idx = scores.topk(keep, dim=1).indices
        idx, _ = torch.sort(idx, dim=1)

        routed = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        return routed, idx


class DynamicLowRankProjector(nn.Module):
    is_custom_attention: bool = True
    r"""SVD-initialized low-rank projector with dynamic variational gate scaling and LoRA adapters."""

    def __init__(
        self,
        rank: int = 64,
        in_dim: int = 768,
        out_dim: int = 768,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.rank = rank
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lora_rank = lora_rank
        self.scaling = lora_alpha / max(1, lora_rank)

        self.U = nn.Parameter(torch.randn(out_dim, rank, **factory_kwargs) * 0.02)
        self.S = nn.Parameter(torch.ones(rank, **factory_kwargs))
        self.V = nn.Parameter(torch.randn(in_dim, rank, **factory_kwargs) * 0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_dim), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_dim, lora_rank), **factory_kwargs))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def set_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            eff_rank = min(self.rank, U.size(1), Vh.size(0))

            self.U.data.zero_()
            self.S.data.zero_()
            self.V.data.zero_()

            min_u = min(self.U.size(0), U.size(0))
            min_v = min(self.V.size(0), Vh.size(1))

            self.U.data[:min_u, :eff_rank].copy_(U[:min_u, :eff_rank])
            self.S.data[:eff_rank].copy_(S[:eff_rank])
            self.V.data[:min_v, :eff_rank].copy_(Vh[:eff_rank, :min_v].t().contiguous())

            if bias is not None and self.bias is not None:
                b_val = bias.data if isinstance(bias, nn.Parameter) else bias
                min_b = min(self.bias.size(0), b_val.size(0))
                self.bias.data[:min_b].copy_(b_val[:min_b])

            rec = (self.U[:, :eff_rank] * self.S[:eff_rank].unsqueeze(0)) @ self.V[:, :eff_rank].t().contiguous()
            f_norm = float(torch.linalg.norm(rec))
            if f_norm < 1e-4:
                logger.warning(
                    f"DynamicLowRankProjector: reconstructed norm is near zero ({f_norm}), re-initializing randomly"
                )
                nn.init.orthogonal_(self.U.data)
                nn.init.orthogonal_(self.V.data)
                self.S.data.fill_(1.0)

    def copy_lora_from(self, sub: Any) -> None:
        """Preserve source LoRA adapters."""
        if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
            src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
            src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
            if hasattr(src_lA, "data") and self.lora_A is not None:
                r_min = min(self.lora_A.shape[0], src_lA.shape[0])
                in_min = min(self.lora_A.shape[1], src_lA.shape[1])
                self.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                self.lora_A.requires_grad = True
            if hasattr(src_lB, "data") and self.lora_B is not None:
                out_min = min(self.lora_B.shape[0], src_lB.shape[0])
                r_min = min(self.lora_B.shape[1], src_lB.shape[1])
                self.lora_B.data[:out_min, :r_min].copy_(src_lB.data[:out_min, :r_min])
                self.lora_B.requires_grad = True
            if hasattr(sub, "lora_scaling"):
                self.scaling = sub.lora_scaling

    def _pytorch_forward(self, x: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
        v = self.V.to(x.dtype)
        s = self.S.to(x.dtype)
        u = self.U.to(x.dtype)
        z = torch.matmul(x, v)
        scale = s.view(*([1] * (z.dim() - 1)), -1)
        if gate is not None:
            g = gate.to(x.dtype)
            while g.dim() < z.dim():
                g = g.unsqueeze(-2)
            scale = scale * g
        z = z * scale
        out = torch.matmul(z, u.t().contiguous())
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
        out: torch.Tensor | None = None
        if x.is_cuda:
            if gate is None:
                try:
                    from neuralops.ops.cuda.low_rank_svd import fused_lowrank_gated_gemm

                    # U already has shape (out_dim, rank) matching fused_lowrank_gated_gemm's U: [N, R].
                    u_eff = (self.U * self.S.unsqueeze(0)).to(x.dtype).contiguous()
                    v_eff = self.V.t().to(x.dtype).contiguous()
                    b_eff = self.bias.to(x.dtype) if self.bias is not None else None
                    out = fused_lowrank_gated_gemm(x, v_eff, u_eff, alpha=None, bias=b_eff)
                except Exception:
                    out = None

            if out is None and x.dim() == 3 and (gate is None or gate.dim() == 2):
                try:
                    from neuralops.ops.triton.low_rank import low_rank_forward

                    out = low_rank_forward(x, self.V, self.S, self.U, gate, self.bias)
                except Exception:
                    out = None

        if out is None:
            out = self._pytorch_forward(x, gate)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            la = self.lora_A.to(x.dtype)
            lb = self.lora_B.to(x.dtype)
            lora_out = F.linear(F.linear(x, la), lb) * self.scaling
            out = out + lora_out

        return out



__all__ = [
    "_masked_mean",
    "VariationalSpectralGate",
    "HeadGate",
    "TokenRouter",
    "DynamicLowRankProjector",
]
