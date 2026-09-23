"""
Kronecker Linear Layer: weight matrix represented as Kronecker product A ⊗ B.
Provides true sub-quadratic FLOP execution and factorized memory efficiency.

Bug-fix map:
  KL-FIX-01  is_custom_attention = False (this is a linear, not attention).
  KL-FIX-02  Weight cache invalidated on every parameter write (via _apply / property).
  KL-FIX-03  _fit_svd bias handling has a clean unconditional branch.
  KL-FIX-04  LoRA rank detection guards against missing .shape.
  KL-FIX-05  del self.A uses register_parameter("A", None) for safe cleanup.
  KL-FIX-06  LoRA forward no longer needs an explicit fp32 cast (kept in base dtype).
  KL-FIX-07  _load_from_state_dict invalidates cache.
  KL-FIX-08  Forward sanity check in from_linear after factor fitting.
  KL-FIX-09  _fit_svd registers fresh Parameters cleanly.
  KL-FIX-10  weight property computes from scratch when needed.
  KL-FIX-11  bias copied with matching dtype.
  KL-FIX-12  LoRA path simplified.
  KL-FIX-13  factorize_dim never returns (1,1) for dim>1.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Helpers
# ===========================================================================

def factorize_dim(dim: int) -> Tuple[int, int]:
    """
    Factorize a dimension into two balanced factors (a, b) with a * b = dim,
    minimizing |a - b|. Returns (1, dim) for prime dimensions.
    KL-FIX-13: for dim=1 returns (1, 1) explicitly.
    """
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    if dim == 1:
        return (1, 1)
    a = int(math.isqrt(dim))
    while a > 1:
        if dim % a == 0:
            return (a, dim // a)
        a -= 1
    return (1, dim)


# ===========================================================================
# KroneckerLinear
# ===========================================================================

class KroneckerLinear(nn.Module):
    # KL-FIX-01: this is a linear layer, not an attention layer.
    is_custom_attention: bool = False
    is_custom_linear: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: Optional[int] = None,
        m1: Optional[int] = None,
        n1: Optional[int] = None,
        m2: Optional[int] = None,
        n2: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.nf = self.out_features  # Conv1D compatibility
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        # ---- Factor dimensions ----
        if m1 is not None and m2 is not None and n1 is not None and n2 is not None:
            if m1 * m2 != self.out_features or n1 * n2 != self.in_features:
                raise ValueError(
                    f"Factor dims mismatch: {m1}*{m2}!={self.out_features} "
                    f"or {n1}*{n2}!={self.in_features}"
                )
            self.m1, self.m2 = int(m1), int(m2)
            self.n1, self.n2 = int(n1), int(n2)
        else:
            self.n1, self.n2 = factorize_dim(self.in_features)
            self.m1, self.m2 = factorize_dim(self.out_features)

        # Decomposable when both factorizations are non-trivial OR dims are tiny.
        self.can_decompose = (
            (self.n1 * self.n2 == self.in_features)
            and (self.m1 * self.m2 == self.out_features)
            and (self.n1 > 1 or self.in_features <= 3)
            and (self.m1 > 1 or self.out_features <= 3)
        )

        # ---- Parameter allocation ----
        requires_grad = not (self.lora_r > 0)
        self._init_base_parameters(
            bias=bias, dtype=dtype, device=device,
            requires_grad=requires_grad, rank=rank,
        )

        # ---- LoRA parameters ----
        if self.lora_r > 0:
            # KL-FIX-12: keep LoRA in the same dtype as base parameters.
            param_dtype = next(self.parameters()).dtype if any(True for _ in self.parameters()) else torch.float32
            param_device = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device("cpu")
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, self.in_features),
                            dtype=param_dtype, device=param_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((self.out_features, self.lora_r),
                            dtype=param_dtype, device=param_device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    # ------------------------------------------------------------------
    # KL-FIX-09: clean parameter initialization
    # ------------------------------------------------------------------
    def _init_base_parameters(
        self,
        bias: bool,
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
        requires_grad: bool,
        rank: Optional[int],
    ) -> None:
        if self.can_decompose:
            # Kronecker factors A (m1, n1), B (m2, n2)
            self.A = nn.Parameter(
                torch.empty(self.m1, self.n1, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
            self.B = nn.Parameter(
                torch.empty(self.m2, self.n2, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
            fan_in = max(self.in_features, 1)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.A, -bound, bound)
            nn.init.uniform_(self.B, -bound, bound)

            # Ensure U / V don't exist.
            self.register_parameter("U", None)
            self.register_parameter("V", None)

            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.out_features, dtype=dtype, device=device),
                    requires_grad=requires_grad,
                )
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter("bias", None)
        else:
            effective_rank = rank or max(1, min(self.in_features, self.out_features) // 4)
            self.rank = min(effective_rank, self.in_features, self.out_features)
            self.U = nn.Parameter(
                torch.empty(self.out_features, self.rank, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
            self.V = nn.Parameter(
                torch.empty(self.rank, self.in_features, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)

            # Ensure A / B don't exist.
            self.register_parameter("A", None)
            self.register_parameter("B", None)

            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features, dtype=dtype, device=device),
                    requires_grad=requires_grad,
                )
            else:
                self.register_parameter("bias", None)

    # ------------------------------------------------------------------
    # KL-FIX-02 / KL-FIX-07: cache lifecycle
    # ------------------------------------------------------------------
    def _load_from_state_dict(self, *args, **kwargs):
        # KL-FIX-07: state_dict load invalidates weight cache.
        self._cached_weight = None
        super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, fn, *args, **kwargs):
        # _apply is called for .to(), .cuda(), .float(), .half(), etc.
        # Invalidate cache whenever parameters move.
        self._cached_weight = None
        return super()._apply(fn, *args, **kwargs)

    def _invalidate_cache(self) -> None:
        self._cached_weight = None

    # ------------------------------------------------------------------
    # Weight accessors (KL-FIX-10: no stale caching)
    # ------------------------------------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return self.get_weight()

    def get_weight(self) -> torch.Tensor:
        if self.can_decompose and self.A is not None and self.B is not None:
            return torch.kron(self.A, self.B)
        if self.U is not None and self.V is not None:
            return torch.matmul(self.U, self.V)
        # Fallback: should never happen for a properly initialized layer.
        raise RuntimeError("KroneckerLinear has no valid parameter set.")

    @property
    def bias(self):  # override via instance
        return self._parameters.get("bias", None)

    @bias.setter
    def bias(self, value):
        # nn.Module.__setattr__ handles Parameter vs None.
        self.__setattr__("bias", value)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.can_decompose and self.A is not None and self.B is not None:
            out = self._forward_kron(x)
        else:
            out = self._forward_svd(x)

        # KL-FIX-12: LoRA in native dtype (no explicit fp32 cast).
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            out = out + self.scaling * lora_out.to(out.dtype)
        return out

    def _forward_kron(self, x: torch.Tensor) -> torch.Tensor:
        """(A ⊗ B) x computed without materializing the full weight."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)         # (K, n1 * n2)
        K = x_flat.shape[0]

        # Step 1: contract with B over n2  ->  (K * n1, m2)
        x_b = torch.matmul(
            x_flat.reshape(K * self.n1, self.n2),
            self.B.t().contiguous(),
        )
        # Step 2: (K, m2, n1) -> (K * m2, n1) @ A^T  ->  (K * m2, m1)
        x_b = x_b.reshape(K, self.n1, self.m2).transpose(1, 2).contiguous().reshape(K * self.m2, self.n1)
        x_a = torch.matmul(x_b, self.A.t().contiguous())
        # Step 3: back to (K, m1 * m2)
        out = x_a.reshape(K, self.m2, self.m1).transpose(1, 2).contiguous().reshape(K, self.out_features)

        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    def _forward_svd(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.V), self.U, self.bias)

    # ------------------------------------------------------------------
    # SVD fitting (KL-FIX-03, KL-FIX-05, KL-FIX-09, KL-FIX-11)
    # ------------------------------------------------------------------
    def _fit_svd(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        rank: Optional[int] = None,
        max_error: float = 0.20,
    ) -> None:
        """Fit low-rank SVD factors ensuring parameter reduction."""
        in_f, out_f = self.in_features, self.out_features
        target_dtype = weight.dtype
        target_device = weight.device
        W_norm = torch.norm(weight, p="fro").item()

        # Parameter reduction constraint: r * (in + out) < in * out
        max_allowed_rank = max(
            1,
            min(in_f, out_f, int(0.75 * (in_f * out_f) / max(in_f + out_f, 1))),
        )
        base_r = rank if rank is not None and rank > 0 else max(1, min(in_f, out_f) // 4)
        base_r = min(base_r, max_allowed_rank)

        try:
            U, S, Vt = torch.linalg.svd(weight.float(), full_matrices=False)
            candidate_ranks = [base_r]
            if base_r * 2 <= max_allowed_rank:
                candidate_ranks.append(base_r * 2)
            if base_r * 4 <= max_allowed_rank:
                candidate_ranks.append(base_r * 4)

            best_r = base_r
            best_U_r = None
            best_V_r = None
            for cand_r in candidate_ranks:
                U_r = U[:, :cand_r] * torch.sqrt(S[:cand_r].unsqueeze(0))
                V_r = torch.sqrt(S[:cand_r].unsqueeze(1)) * Vt[:cand_r, :]
                W_cand = torch.matmul(U_r, V_r)
                err = float(torch.norm(weight.float() - W_cand, p="fro").item() / max(W_norm, 1e-12))
                best_r = cand_r
                best_U_r = U_r
                best_V_r = V_r
                if err <= max_error:
                    break
        except Exception:
            best_r = base_r
            best_U_r = torch.zeros(out_f, best_r, device=target_device, dtype=torch.float32)
            best_V_r = torch.zeros(best_r, in_f, device=target_device, dtype=torch.float32)

        self.can_decompose = False
        # KL-FIX-05: clean removal of Kronecker parameters.
        self.register_parameter("A", None)
        self.register_parameter("B", None)

        self.rank = best_r
        # KL-FIX-09: clean registration of SVD parameters.
        self.U = nn.Parameter(best_U_r.to(dtype=target_dtype, device=target_device))
        self.V = nn.Parameter(best_V_r.to(dtype=target_dtype, device=target_device))

        # KL-FIX-03 + KL-FIX-11: unconditional, dtype-matched bias.
        if bias is not None:
            self.bias = nn.Parameter(
                bias.clone().to(dtype=target_dtype, device=target_device)
            )
        else:
            self.register_parameter("bias", None)

        # KL-FIX-10: invalidate cache.
        self._invalidate_cache()

    # ------------------------------------------------------------------
    # Factory (KL-FIX-04, KL-FIX-08)
    # ------------------------------------------------------------------
    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: Optional[int] = None,
        max_error: float = 0.20,
        m1: Optional[int] = None,
        n1: Optional[int] = None,
        m2: Optional[int] = None,
        n2: Optional[int] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> "KroneckerLinear":
        """Build a KroneckerLinear from a pretrained Linear/Conv1D/LoRA wrapper."""
        target_mod = getattr(linear, "base_layer", linear)
        cls_lower = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if is_conv1d:
            in_features = raw_w.shape[0]
            out_features = raw_w.shape[1]
            weight = raw_w.t().contiguous().float()
        else:
            out_features = raw_w.shape[0]
            in_features = raw_w.shape[1]
            weight = raw_w.float()

        bias_holder = getattr(target_mod, "bias", None)
        bias = bias_holder.data.clone() if bias_holder is not None else None
        target_dtype = raw_w.dtype
        target_device = raw_w.device

        # KL-FIX-04: robust LoRA detection.
        existing_lora_A = getattr(linear, "lora_A", None)
        existing_lora_B = getattr(linear, "lora_B", None)
        effective_lora_r = lora_r
        effective_lora_alpha = lora_alpha
        if existing_lora_A is not None and existing_lora_B is not None:
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            if isinstance(w_A, torch.Tensor) and w_A.dim() >= 1:
                effective_lora_r = int(w_A.shape[0])
            scaling = getattr(linear, "scaling", None)
            if isinstance(scaling, (int, float)) and scaling > 0:
                effective_lora_alpha = float(scaling) * effective_lora_r
            elif hasattr(linear, "lora_alpha"):
                effective_lora_alpha = float(linear.lora_alpha)

        kron_layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            rank=rank,
            m1=m1, n1=n1, m2=m2, n2=n2,
            dtype=target_dtype, device=target_device,
            lora_r=effective_lora_r,
            lora_alpha=effective_lora_alpha,
        )

        # Transfer existing LoRA if shapes match.
        if existing_lora_A is not None and existing_lora_B is not None \
                and kron_layer.lora_A is not None and kron_layer.lora_B is not None:
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            w_B = getattr(existing_lora_B, "weight", existing_lora_B)
            if isinstance(w_A, torch.Tensor) and w_A.shape == kron_layer.lora_A.shape:
                kron_layer.lora_A.data.copy_(w_A.to(kron_layer.lora_A.device, dtype=kron_layer.lora_A.dtype))
            if isinstance(w_B, torch.Tensor) and w_B.shape == kron_layer.lora_B.shape:
                kron_layer.lora_B.data.copy_(w_B.to(kron_layer.lora_B.device, dtype=kron_layer.lora_B.dtype))

        W_norm = torch.norm(weight, p="fro").item()
        err = 1.0

        if kron_layer.can_decompose:
            m1_, m2_ = kron_layer.m1, kron_layer.m2
            n1_, n2_ = kron_layer.n1, kron_layer.n2
            try:
                # Rearrangement for Nearest Kronecker Product.
                W_tensor = weight.view(m1_, m2_, n1_, n2_).permute(0, 2, 1, 3).contiguous()
                W_reordered = W_tensor.view(m1_ * n1_, m2_ * n2_).float()

                U, S, Vt = torch.linalg.svd(W_reordered, full_matrices=False)
                sigma1 = S[0].item() if S.numel() > 0 else 1.0
                scale = math.sqrt(max(sigma1, 1e-12))

                u1 = U[:, 0] * scale
                v1 = Vt[0, :] * scale

                kron_layer.A.data.copy_(
                    u1.view(m1_, n1_).to(dtype=target_dtype, device=target_device)
                )
                kron_layer.B.data.copy_(
                    v1.view(m2_, n2_).to(dtype=target_dtype, device=target_device)
                )

                W_recon = torch.kron(kron_layer.A, kron_layer.B).float()
                err = float(torch.norm(weight - W_recon, p="fro").item() / max(W_norm, 1e-12))
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(
                    "KroneckerLinear NKP failed: %s; falling back to SVD.", e
                )
                err = 1.0

            if err > max_error:
                kron_layer._fit_svd(weight, bias, rank=rank, max_error=max_error)
        else:
            kron_layer._fit_svd(weight, bias, rank=rank, max_error=max_error)

        if bias is not None and kron_layer.bias is not None:
            kron_layer.bias.data.copy_(
                bias.to(dtype=kron_layer.bias.dtype, device=kron_layer.bias.device)
            )

        # KL-FIX-08: forward sanity check.
        try:
            with torch.no_grad():
                dummy = torch.randn(2, in_features, device=target_device, dtype=target_dtype)
                _ = kron_layer(dummy)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "KroneckerLinear forward sanity check failed: %s", e
            )

        return kron_layer


__all__ = ["KroneckerLinear", "factorize_dim"]