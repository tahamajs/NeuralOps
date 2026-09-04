"""
Spectral Residual Low-Rank Adaptation (Res-SVD / Spectral Residual LoRA).

Decomposes dense weight matrices into:
1. Frozen Base Low-Rank Factors: $W_{\\text{base}} = U_r \\Sigma_r V_r^\\top$ (rank $r$, frozen).
2. Trainable Residual Adapters: Initialized directly from the top-$k$ singular directions of the
   residual error $E = W - W_{\\text{base}} = \\sum_{j=r+1}^{r+k} \\sigma_j u_j v_j^\\top$.
   $$A = \\sqrt{\\Sigma_{\\text{tail}}} V_{\\text{tail}}^\\top, \\quad B = U_{\\text{tail}} \\sqrt{\\Sigma_{\\text{tail}}}$$
   ensuring $\\Delta W = B A \\approx E$ at zero-shot initialization.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class ResSVDLinear(nn.Module):
    """
    Spectral Residual LoRA (Res-SVD) Linear Layer.

    Executes forward pass as:
    y = x @ W_base^T + scaling * (x @ A^T) @ B^T + bias
    where W_base is either factorized as (base_V, base_U) with requires_grad=False,
    and (lora_A, lora_B) are trainable adapter matrices initialized from the residual tail.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_rank: int = 32,
        residual_rank: int = 8,
        scaling: float = 1.0,
        factorize_base: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        max_possible_rank = min(in_features, out_features)

        # Bounds validation
        self.base_rank = max(1, min(base_rank, max_possible_rank))
        remaining_rank = max(0, max_possible_rank - self.base_rank)
        self.residual_rank = max(0, min(residual_rank, remaining_rank))
        self.scaling = float(scaling)
        self.factorize_base = factorize_base

        # Base parameters (Frozen)
        if self.factorize_base:
            self.base_V = nn.Parameter(
                torch.empty(self.base_rank, in_features, dtype=dtype, device=device),
                requires_grad=False,
            )
            self.base_U = nn.Parameter(
                torch.empty(out_features, self.base_rank, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.base_weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=dtype, device=device),
                requires_grad=False,
            )

        # Trainable Residual LoRA Adapters
        if self.residual_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.residual_rank, in_features, dtype=dtype, device=device),
                requires_grad=True,
            )
            self.lora_B = nn.Parameter(
                torch.empty(out_features, self.residual_rank, dtype=dtype, device=device),
                requires_grad=True,
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform parameter initialization for base and zero for adapters."""
        if self.factorize_base:
            nn.init.kaiming_uniform_(self.base_V, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.base_U, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        if self.residual_rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        if self.factorize_base:
            base = torch.matmul(self.base_U, self.base_V)
        else:
            base = self.base_weight
        if self.residual_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            res = self.scaling * torch.matmul(self.lora_B, self.lora_A)
            return base + res
        return base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward execution:
        y = Base(x) + scaling * (x @ A^T) @ B^T + bias
        """
        # RESEARCH FIX: Two-stage factorized forward for frozen base to minimize memory footprint
        if self.factorize_base:
            base_intermediate = F.linear(x, self.base_V)
            base_out = F.linear(base_intermediate, self.base_U)
        else:
            base_out = F.linear(x, self.base_weight)

        # Trainable spectral residual LoRA adapter forward
        if self.residual_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            res_intermediate = F.linear(x, self.lora_A)
            res_out = self.scaling * F.linear(res_intermediate, self.lora_B)
            out = base_out + res_out
        else:
            out = base_out

        if self.bias is not None:
            out = out + self.bias

        return out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        base_rank: int = 32,
        residual_rank: int = 8,
        scaling: float = 1.0,
        factorize_base: bool = True,
        freeze_base: bool = True,
    ) -> ResSVDLinear:
        """
        Construct ResSVDLinear by decomposing a pretrained nn.Linear weight:
        1. Base factors U_r, V_r from the dominant top-r singular components.
        2. Residual adapter factors A, B directly initialized from the residual tail (r+1 to r+k).

        Args:
            linear: Source dense linear layer.
            base_rank: Rank r for the frozen base matrix.
            residual_rank: Rank k for the trainable residual adapter.
            scaling: Multiplier alpha / k for adapter output.
            factorize_base: Whether to store base weights as factorized U, V matrices.
            freeze_base: Whether to freeze base factors (requires_grad=False).

        Returns:
            ResSVDLinear: Spectral residual LoRA module.
        """
        in_f = linear.in_features
        out_f = linear.out_features
        max_rank = min(in_f, out_f)

        r = max(1, min(base_rank, max_rank))
        rem = max(0, max_rank - r)
        k = max(0, min(residual_rank, rem))

        new_layer = cls(
            in_features=in_f,
            out_features=out_f,
            base_rank=r,
            residual_rank=k,
            scaling=scaling,
            factorize_base=factorize_base,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )

        W = linear.weight.data.float()  # (out_f, in_f)

        # SVD decomposition of source weight
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # 1. Base Truncation (Dominant Top-r Singular Components)
        U_base = U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)  # (out_f, r)
        V_base = torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]  # (r, in_f)

        if factorize_base:
            new_layer.base_U.data.copy_(U_base.to(dtype=linear.weight.dtype))
            new_layer.base_V.data.copy_(V_base.to(dtype=linear.weight.dtype))
            new_layer.base_U.requires_grad = not freeze_base
            new_layer.base_V.requires_grad = not freeze_base
        else:
            W_base = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
            new_layer.base_weight.data.copy_(W_base.to(dtype=linear.weight.dtype))
            new_layer.base_weight.requires_grad = not freeze_base

        # 2. Residual Tail Initialization (Singular directions r to r+k)
        # RESEARCH FIX: Analytical spectral initialization of LoRA adapter weights (A, B) directly
        # from the residual tail E = W - U_r V_r minimizes initial reconstruction loss to zero-shot optimality.
        if k > 0:
            S_tail = S[r : r + k]
            U_tail = U[:, r : r + k]
            Vh_tail = Vh[r : r + k, :]

            # Scale S_tail by adapter scaling factor gamma: (B @ A) * gamma = E_tail
            scale_factor = max(1e-8, float(scaling))
            S_tail_scaled = S_tail / scale_factor

            sqrt_S = torch.sqrt(torch.clamp(S_tail_scaled, min=0.0))
            lora_A_init = sqrt_S.unsqueeze(1) * Vh_tail  # (k, in_f)
            lora_B_init = U_tail * sqrt_S.unsqueeze(0)  # (out_f, k)

            new_layer.lora_A.data.copy_(lora_A_init.to(dtype=linear.weight.dtype))
            new_layer.lora_B.data.copy_(lora_B_init.to(dtype=linear.weight.dtype))
            new_layer.lora_A.requires_grad = True
            new_layer.lora_B.requires_grad = True

        # Copy bias if present
        if linear.bias is not None:
            new_layer.bias.data.copy_(linear.bias.data)

        return new_layer

    def merge_weights(self) -> nn.Linear:
        """
        Merge frozen base weights and trainable residual adapter weights into a dense nn.Linear.

        Returns:
            nn.Linear: Standard dense linear layer with merged weights.
        """
        device = next(self.parameters()).device
        dtype = (
            self.base_U.dtype if self.factorize_base else self.base_weight.dtype
        )

        merged_linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            if self.factorize_base:
                W_base = self.base_U.float() @ self.base_V.float()
            else:
                W_base = self.base_weight.float()

            if self.residual_rank > 0 and self.lora_A is not None and self.lora_B is not None:
                delta_W = self.scaling * (self.lora_B.float() @ self.lora_A.float())
                W_total = W_base + delta_W
            else:
                W_total = W_base

            merged_linear.weight.data.copy_(W_total.to(dtype=dtype))
            if self.bias is not None:
                merged_linear.bias.data.copy_(self.bias.data)

        return merged_linear

    def get_residual_error_norm(
        self, original_weight: Optional[torch.Tensor] = None
    ) -> float:
        """
        Calculate relative Frobenius norm reconstruction error:
        || W_orig - (W_base + scaling * B @ A) ||_F / || W_orig ||_F
        """
        with torch.no_grad():
            if self.factorize_base:
                W_approx = self.base_U.float() @ self.base_V.float()
            else:
                W_approx = self.base_weight.float()

            if self.residual_rank > 0 and self.lora_A is not None and self.lora_B is not None:
                W_approx = W_approx + self.scaling * (
                    self.lora_B.float() @ self.lora_A.float()
                )

            if original_weight is not None:
                W_orig = original_weight.float().to(W_approx.device)
                error = torch.norm(W_orig - W_approx, p="fro") / (
                    torch.norm(W_orig, p="fro") + 1e-12
                )
                return float(error.item())

            return 0.0

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"base_rank={self.base_rank}, residual_rank={self.residual_rank}, "
            f"scaling={self.scaling}, factorize_base={self.factorize_base}"
        )


class ResSVDCompressor(BaseCompressor):
    """
    Spectral Residual LoRA (Res-SVD) Compressor.

    Replaces standard Linear layers with ResSVDLinear modules, truncating base weights
    at rank r (frozen) and initializing adapter matrices A, B directly from the residual error tail.
    """

    def __init__(
        self,
        model: nn.Module,
        base_rank: int = 32,
        residual_rank: int = 8,
        scaling: float = 1.0,
        factorize_base: bool = True,
        freeze_base: bool = True,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="res_svd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            base_rank = getattr(cfg, "base_rank", getattr(cfg, "rank", base_rank))
            residual_rank = getattr(cfg, "residual_rank", residual_rank)
            scaling = getattr(cfg, "scaling", getattr(cfg, "lora_alpha", scaling))
            factorize_base = getattr(cfg, "factorize_base", factorize_base)
            freeze_base = getattr(cfg, "freeze_base", freeze_base)

        if "rank" in kwargs and kwargs["rank"] is not None:
            base_rank = kwargs["rank"]

        self.base_rank = base_rank
        self.residual_rank = residual_rank
        self.scaling = scaling
        self.factorize_base = factorize_base
        self.freeze_base = freeze_base
        self.target_modules = target_modules
        self.replaced_layers: List[str] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Check whether module is an eligible nn.Linear layer matching filter."""
        if not isinstance(module, nn.Linear):
            return False
        if isinstance(module, ResSVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Compress target linear layers into ResSVDLinear modules.

        Returns:
            Tuple[nn.Module, Dict[str, Any]]: Compressed model and compression statistics.
        """
        self.replaced_layers = []

        # Find target linear layers
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("No compressible linear layers found in model.")
            return self.model, self.get_compression_stats()

        for name, old_linear in target_layers:
            res_layer = ResSVDLinear.from_linear(
                linear=old_linear,
                base_rank=self.base_rank,
                residual_rank=self.residual_rank,
                scaling=self.scaling,
                factorize_base=self.factorize_base,
                freeze_base=self.freeze_base,
            )
            replace_module(self.model, name, res_layer)
            self.replaced_layers.append(name)

        # Calculate statistics
        total_params = self._count_params(self.model)
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        stats = self.get_compression_stats()
        stats["trainable_parameters"] = trainable_params
        stats["frozen_parameters"] = frozen_params
        stats["trainable_percentage"] = (
            (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
        )
        stats["base_rank"] = self.base_rank
        stats["residual_rank"] = self.residual_rank
        stats["replaced_layers"] = self.replaced_layers

        return self.model, stats


# Alias class for PEFT terminology
class SpectralResidualLoRA(ResSVDCompressor):
    """PEFT-focused wrapper for Spectral Residual LoRA."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = "spectral_residual_lora"


# Register under both canonical and alias names
@COMPRESSOR_REGISTRY.register("res_svd")
class ResSVDCompressorRegistered(ResSVDCompressor):
    pass


@COMPRESSOR_REGISTRY.register("spectral_residual_lora")
class SpectralResidualLoRARegistered(SpectralResidualLoRA):
    pass


def res_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Res-SVD factory function."""
    compressor = ResSVDCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


def spectral_residual_lora_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Spectral Residual LoRA factory function."""
    compressor = SpectralResidualLoRA(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


res_svd_compressor_factory = res_svd_factory

__all__ = [
    "ResSVDLinear",
    "ResSVDCompressor",
    "SpectralResidualLoRA",
    "res_svd_factory",
    "spectral_residual_lora_factory",
    "res_svd_compressor_factory",
]
