"""
Tensor Train (TT) Decomposition Layer for PyTorch.
Uses torchtt if available; falls back to a low‑rank SVD approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

try:
    import torchtt
    _HAS_TORCHTT = True
except ImportError:
    _HAS_TORCHTT = False


class TTLinearLayer(nn.Module):
    """
    Linear layer with weights stored in Tensor Train (TT) format.

    If torchtt is installed, true TT decomposition is used.
    Otherwise, falls back to a low‑rank SVD approximation with the same
    parameter savings and a unified interface.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tt_rank: int = 16,
        tensorized_shape_in: Optional[Tuple[int, ...]] = None,
        tensorized_shape_out: Optional[Tuple[int, ...]] = None,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank
        self.use_tt = _HAS_TORCHTT

        # Determine tensorized shapes automatically if not provided
        if tensorized_shape_in is None:
            tensorized_shape_in = self._auto_tensorize(in_features)
        if tensorized_shape_out is None:
            tensorized_shape_out = self._auto_tensorize(out_features)

        self.tensorized_shape_in = tensorized_shape_in
        self.tensorized_shape_out = tensorized_shape_out

        if self.use_tt:
            # Build a rank list for torchtt: [1, r1, r2, ..., r_{d-1}, 1]
            num_dims = len(tensorized_shape_in)
            rank_list = [1] + [tt_rank] * (num_dims - 1) + [1]
            self.tt_layer = torchtt.nn.LinearLayerTT(
                size_in=list(tensorized_shape_in),
                size_out=list(tensorized_shape_out),
                rank=rank_list,
                dtype=dtype,
                initializer='He',
            )
        else:
            # Fallback: simple low‑rank SVD decomposition
            r = min(tt_rank, min(out_features, in_features))
            self.U = nn.Parameter(torch.empty(out_features, r))
            self.S = nn.Parameter(torch.empty(r))
            self.V = nn.Parameter(torch.empty(r, in_features))
            nn.init.normal_(self.U, std=0.02)
            nn.init.ones_(self.S)
            nn.init.normal_(self.V, std=0.02)

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def _auto_tensorize(dim: int) -> Tuple[int, int]:
        """Factorize a dimension into two near‑square factors."""
        a = int(math.sqrt(dim))
        while dim % a != 0:
            a -= 1
        b = dim // a
        # Return the larger first
        return (b, a) if b > a else (a, b)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        tt_rank: int,
        **kwargs,
    ) -> "TTLinearLayer":
        """
        Initialise from a pretrained nn.Linear layer.

        Args:
            linear: original nn.Linear layer.
            tt_rank: TT‑rank for the decomposition.
        """
        tt_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            tt_rank=tt_rank,
            bias=linear.bias is not None,
            **kwargs,
        )
        weight = linear.weight.data

        if _HAS_TORCHTT:
            # Reshape to tensorized shape and decompose with torchtt
            tensorized_shape = tt_layer.tensorized_shape_out + tt_layer.tensorized_shape_in
            weight_reshaped = weight.view(tensorized_shape).cpu()
            tt_tensor = torchtt.tensor(weight_reshaped, eps=1e-6, rmax=tt_rank)
            for i, core in enumerate(tt_tensor.cores):
                tt_layer.tt_layer.cores[i].data = core.clone()
        else:
            # Fallback: SVD decomposition
            U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
            r = min(tt_rank, min(weight.shape))
            tt_layer.U.data = (U[:, :r] * S[:r].unsqueeze(0)).float()
            tt_layer.S.data = S[:r].float()
            tt_layer.V.data = Vh[:r, :].float()

        if linear.bias is not None:
            tt_layer.bias.data.copy_(linear.bias.data)
        return tt_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_tt:
            batch_shape = x.shape[:-1]
            x_flat = x.view(-1, self.in_features)
            y = self.tt_layer(x_flat)
            if self.bias is not None:
                y = y + self.bias
            return y.view(*batch_shape, self.out_features)
        else:
            # Fallback: (x @ V^T) * S @ U^T
            xV = F.linear(x, self.V)  # (..., r)
            xV = xV * self.S.unsqueeze(0) if xV.dim() > 2 else xV * self.S
            out = F.linear(xV, self.U.t())
            if self.bias is not None:
                out = out + self.bias
            return out

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'tt_rank={self.tt_rank}, bias={self.bias is not None}, '
                f'use_tt={self.use_tt}')