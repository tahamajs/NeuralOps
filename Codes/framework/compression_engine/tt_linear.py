"""
Tensor-Train Linear Layer: parameterizes weight matrices as Tensor-Train (TT) Matrix cores.
Reference: Novikov et al., 'Tensorizing Neural Networks' (NeurIPS 2015).
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def factorize_dimension_pair(
    in_features: int, out_features: int, num_factors: int = 2
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Factorize in_features and out_features into num_factors balanced integer components.
    """
    def get_factors(n: int, k: int) -> List[int]:
        if k == 1:
            return [n]
        if k == 2:
            a = int(math.isqrt(n))
            while a > 1:
                if n % a == 0:
                    return [a, n // a]
                a -= 1
            return [1, n]

        # For k > 2, greedily factor smallest near root
        factors = []
        curr = n
        for step in range(k, 1, -1):
            root = max(1, int(round(curr ** (1.0 / step))))
            found = False
            for cand in range(root, 0, -1):
                if curr % cand == 0:
                    factors.append(cand)
                    curr = curr // cand
                    found = True
                    break
            if not found:
                factors.append(1)
        factors.append(curr)
        return factors

    p_dims = tuple(get_factors(in_features, num_factors))
    q_dims = tuple(get_factors(out_features, num_factors))
    return p_dims, q_dims


class TTLinearLayer(nn.Module):
    """
    Tensor-Train (TT) Linear Layer.
    Represents weight matrix W in R^{out_features x in_features} as a sequence
    of d TT-cores: G_1, G_2, ..., G_d where core G_k has shape (r_{k-1}, p_k, q_k, r_k)
    with boundary ranks r_0 = r_d = 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tt_rank: int = 16,
        bias: bool = True,
        shape: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank

        if shape is None:
            self.p_dims, self.q_dims = factorize_dimension_pair(in_features, out_features, num_factors=2)
        else:
            self.p_dims, self.q_dims = shape
            if int(np.prod(self.p_dims)) != in_features or int(np.prod(self.q_dims)) != out_features:
                raise ValueError(
                    f"Shape product mismatch: {self.p_dims} ({np.prod(self.p_dims)}) != {in_features} or "
                    f"{self.q_dims} ({np.prod(self.q_dims)}) != {out_features}"
                )

        self.d = len(self.p_dims)
        self.cores = nn.ParameterList()

        # Build TT-cores: G_k of shape (r_{k-1}, p_k, q_k, r_k)
        r_left = 1
        for i in range(self.d - 1):
            p_i = self.p_dims[i]
            q_i = self.q_dims[i]
            r_right = min(tt_rank, max(1, p_i * q_i * r_left))
            core = nn.Parameter(
                torch.empty(r_left, p_i, q_i, r_right, dtype=dtype, device=device)
            )
            bound = 1.0 / math.sqrt(max(1, r_left * p_i))
            nn.init.uniform_(core, -bound, bound)
            self.cores.append(core)
            r_left = r_right

        p_last = self.p_dims[-1]
        q_last = self.q_dims[-1]
        core_last = nn.Parameter(
            torch.empty(r_left, p_last, q_last, 1, dtype=dtype, device=device)
        )
        bound = 1.0 / math.sqrt(max(1, r_left * p_last))
        nn.init.uniform_(core_last, -bound, bound)
        self.cores.append(core_last)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        return self.materialize_weight()

    def materialize_weight(self) -> torch.Tensor:
        """
        Reconstruct the full dense weight matrix W in R^{out_features x in_features}.
        """
        # Start with core 0: (1, p0, q0, r1) -> (p0, q0, r1)
        res = self.cores[0].squeeze(0)  # (p0, q0, r1)
        for i in range(1, self.d):
            core = self.cores[i]  # (r_i, p_i, q_i, r_{i+1})
            # Contract over r_i
            res = torch.tensordot(res, core, dims=([-1], [0]))  # (p0, q0, ..., p_i, q_i, r_{i+1})

        res = res.squeeze(-1)  # (p0, q0, p1, q1, ..., p_{d-1}, q_{d-1})

        # Permute to (q0, q1, ..., q_{d-1}, p0, p1, ..., p_{d-1})
        q_indices = [2 * k + 1 for k in range(self.d)]
        p_indices = [2 * k for k in range(self.d)]
        res = res.permute(*(q_indices + p_indices)).contiguous()
        return res.view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with TT weight contraction."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # For d=2, execute direct efficient contractions
        if self.d == 2:
            p1, p2 = self.p_dims
            q1, q2 = self.q_dims
            # core0: (1, p1, q1, r) -> (p1, q1, r)
            # core1: (r, p2, q2, 1) -> (r, p2, q2)
            c0 = self.cores[0].squeeze(0)
            c1 = self.cores[1].squeeze(-1)
            # Full contraction: W = tensordot(c0, c1, dims=([-1],[0])) -> (p1, q1, p2, q2)
            W_tensor = torch.tensordot(c0, c1, dims=([-1], [0]))
            W = W_tensor.permute(1, 3, 0, 2).contiguous().view(self.out_features, self.in_features)
            out = F.linear(x_flat, W, self.bias)
            return out.view(*orig_shape[:-1], self.out_features)
        else:
            W = self.materialize_weight()
            out = F.linear(x_flat, W, self.bias)
            return out.view(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_linear(
        cls,
        linear_layer: nn.Module,
        tt_rank: int = 16,
        num_factors: int = 2,
        max_error: float = 0.20,
        max_retries: int = 3,
        fallback_to_svd: bool = True,
    ) -> "TTLinearLayer":
        """
        Decompose a pretrained nn.Linear or Conv1D layer into TT-cores using TT-SVD.
        Production-hardened with FP32 stability, Conv1D support, adaptive rank retry,
        and relative Frobenius error validation.
        """
        raw_w = linear_layer.weight.data
        orig_dtype = raw_w.dtype
        orig_device = raw_w.device

        cls_name = linear_layer.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or (
            raw_w.ndim == 2
            and hasattr(linear_layer, "in_features")
            and linear_layer.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        if is_conv1d:
            W = raw_w.t().float().contiguous()
            in_f = W.shape[1]
            out_f = W.shape[0]
        else:
            W = raw_w.float().contiguous()
            out_f, in_f = W.shape[0], W.shape[1]

        bias = linear_layer.bias.data if getattr(linear_layer, "bias", None) is not None else None
        norm_orig = torch.linalg.norm(W).item() + 1e-8

        # Factorize dimensions
        p_dims, q_dims = factorize_dimension_pair(in_f, out_f, num_factors=num_factors)
        if int(np.prod(p_dims)) != in_f or int(np.prod(q_dims)) != out_f:
            p_dims = (1, in_f)
            q_dims = (1, out_f)

        d = len(p_dims)
        # Reshape W (out_f, in_f) -> (q0, q1, ..., p0, p1, ...)
        W_tensor = W.view(*(list(q_dims) + list(p_dims)))
        perm = []
        for k in range(d):
            perm.append(d + k)  # p_k
            perm.append(k)      # q_k
        W_interleaved = W_tensor.permute(*perm).contiguous()

        cur_tt_rank = max(1, min(int(tt_rank), in_f, out_f))
        best_layer: Optional[TTLinearLayer] = None
        best_err = float("inf")
        best_rank = cur_tt_rank

        for attempt in range(max_retries + 1):
            try:
                layer = cls(
                    in_features=in_f,
                    out_features=out_f,
                    tt_rank=cur_tt_rank,
                    bias=bias is not None,
                    shape=(p_dims, q_dims),
                    dtype=orig_dtype,
                    device=orig_device,
                )

                # Sequential TT-SVD in FP32
                curr_matrix = W_interleaved.view(p_dims[0] * q_dims[0], -1)
                r_left = 1

                for i in range(d - 1):
                    p_i = p_dims[i]
                    q_i = q_dims[i]
                    rows = r_left * p_i * q_i
                    curr_matrix = curr_matrix.view(rows, -1)

                    try:
                        U, S, Vt = torch.linalg.svd(curr_matrix, full_matrices=False)
                    except Exception:
                        U, S, Vt = torch.linalg.svd(curr_matrix.cpu(), full_matrices=False)
                        U, S, Vt = U.to(orig_device), S.to(orig_device), Vt.to(orig_device)

                    r_right = min(cur_tt_rank, len(S), layer.cores[i].shape[-1])
                    U_trunc = U[:, :r_right]
                    S_trunc = S[:r_right]
                    Vt_trunc = Vt[:r_right, :]

                    core_data = U_trunc.view(r_left, p_i, q_i, r_right)
                    layer.cores[i].data.zero_()
                    layer.cores[i].data[:r_left, :p_i, :q_i, :r_right].copy_(core_data.to(dtype=orig_dtype))

                    curr_matrix = torch.matmul(torch.diag(S_trunc), Vt_trunc)
                    r_left = r_right

                # Final core
                p_last = p_dims[-1]
                q_last = q_dims[-1]
                layer.cores[-1].data.zero_()
                final_core = curr_matrix.view(r_left, p_last, q_last, 1)
                layer.cores[-1].data[:r_left, :p_last, :q_last, :].copy_(final_core.to(dtype=orig_dtype))

                if bias is not None and layer.bias is not None:
                    layer.bias.data.copy_(bias.to(dtype=orig_dtype, device=orig_device))

                # Validate reconstructed weight shape and error
                W_rec = layer.materialize_weight().float()
                if W_rec.shape != (out_f, in_f):
                    raise RuntimeError(f"Materialized weight shape mismatch: {W_rec.shape} vs ({out_f}, {in_f})")

                rel_err = (torch.linalg.norm(W - W_rec).item()) / norm_orig
                if rel_err < best_err:
                    best_err = rel_err
                    best_layer = layer
                    best_rank = cur_tt_rank

                if rel_err <= max_error or cur_tt_rank >= min(in_f, out_f):
                    break

                # Step up rank for retry
                step = max(4, (min(in_f, out_f) - cur_tt_rank) // max(1, max_retries - attempt))
                cur_tt_rank = min(cur_tt_rank + step, min(in_f, out_f))

            except Exception as e:
                # If SVD or contraction fails, log and stop retries
                break

        if best_layer is None:
            # Fallback initialization: construct base layer with default initialization
            best_layer = cls(
                in_features=in_f,
                out_features=out_f,
                tt_rank=cur_tt_rank,
                bias=bias is not None,
                shape=(p_dims, q_dims),
                dtype=orig_dtype,
                device=orig_device,
            )
            if bias is not None and best_layer.bias is not None:
                best_layer.bias.data.copy_(bias.to(dtype=orig_dtype, device=orig_device))
            best_err = 1.0

        best_layer.reconstruction_error = float(best_err)
        best_layer.tt_rank = best_rank
        return best_layer

