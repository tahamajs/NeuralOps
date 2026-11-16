# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""SVD S6 State Space module for MambaSVDFlash."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SVDS6StateSpace(nn.Module):
    """
    Per-token low-rank SVD state transition.

        U_t, Σ_t, V_t, B_t, C_t = Bottleneck(x_t)
        h_t = U_t · diag(Σ_t) · V_t · h_{t-1} + B_t · x_t
        y_t = C_t · h_t

    Contractive by construction (MSF-FIX-03):
      ||U_t||_2 ≤ 1, ||V_t||_2 ≤ 1, Σ_t ∈ (0, 1)   ⇒  ||UΣV||_2 ≤ 1.
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 16,
        bottleneck_dim: int = 64,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = max(1, int(rank))
        self.bottleneck_dim = max(self.rank, int(bottleneck_dim))

        fk = {"device": device, "dtype": dtype}

        self.down = nn.Linear(d_model, self.bottleneck_dim, bias=False, **fk)

        self.up_u = nn.Linear(self.bottleneck_dim, d_model * self.rank, bias=False, **fk)
        self.up_v = nn.Linear(self.bottleneck_dim, self.rank * d_model, bias=False, **fk)
        self.up_sigma = nn.Linear(self.bottleneck_dim, self.rank, bias=True, **fk)
        self.up_b = nn.Linear(self.bottleneck_dim, d_model, bias=True, **fk)
        self.up_c = nn.Linear(self.bottleneck_dim, d_model, bias=True, **fk)

        with torch.no_grad():
            nn.init.zeros_(self.up_sigma.weight)
            nn.init.constant_(self.up_sigma.bias, 1.75)

        with torch.no_grad():
            if self.up_c.bias is not None:
                nn.init.zeros_(self.up_c.bias)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        r = self.rank

        z = self.down(x)

        U = self.up_u(z).view(B, L, D, r)
        V = self.up_v(z).view(B, L, r, D)
        sigma = torch.sigmoid(self.up_sigma(z))
        B_seq = self.up_b(z)
        C_seq = self.up_c(z)

        U = F.normalize(U.float(), p=2, dim=-2, eps=1e-6).to(x.dtype)
        V = F.normalize(V.float(), p=2, dim=-2, eps=1e-6).to(x.dtype)

        return U, sigma, V, B_seq, C_seq



__all__ = ["SVDS6StateSpace"]
