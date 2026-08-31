"""
ADMM Pruner: Alternating Direction Method of Multipliers for Structured & Semi-Structured Pruning with STE & Dynamic Rho.
Registered in COMPRESSOR_REGISTRY as 'admm_pruner'.

Mathematical Formulation:
$$\\min_W \\frac{1}{2} \\|W - W_0\\|_F^2 + g(Z) \\quad \\text{s.t.} \\quad W = Z$$

ADMM Updates with Dynamic Rho Scheduling:
1. $Z^{(t+1)} = \\Pi_{\\mathcal{S}}(W^{(t)} + U^{(t)})$
2. $W^{(t+1)} = \\frac{1}{1 + \\rho^{(t)}} \\left( W_0 + \\rho^{(t)} (Z^{(t+1)} - U^{(t)}) \\right)$
3. $U^{(t+1)} = U^{(t)} + W^{(t+1)} - Z^{(t+1)}$
4. Primal residual $r^{(t+1)} = \\|W^{(t+1)} - Z^{(t+1)}\\|_F$, Dual residual $s^{(t+1)} = \\rho^{(t)} \\|Z^{(t+1)} - Z^{(t)}\\|_F$
   $$\\rho^{(t+1)} = \\begin{cases} \\rho^{(t)} \\cdot \\tau_{\\text{incr}} & \\text{if } \\|r^{(t+1)}\\| > \\mu \\|s^{(t+1)}\\| \\\\ \\rho^{(t)} / \\tau_{\\text{decr}} & \\text{if } \\|s^{(t+1)}\\| > \\mu \\|r^{(t+1)}\\| \\\\ \\rho^{(t)} & \\text{otherwise} \\end{cases}$$
5. Straight-Through Estimator (STE) gradient pass for continuous backpropagation through binary masks.
6. Post-pruning LayerNorm recalibration to restore activation variance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    apply_block_mask,
    recalibrate_layernorms,
)

logger = logging.getLogger(__name__)


class ADMMSTEPrune(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) autograd function for ADMM sparse weights.
    Forwards masked weights and passes gradients through uninhibited on backward pass.
    """

    @staticmethod
    def forward(ctx: Any, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mask)
        return weight * mask

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (mask,) = ctx.saved_tensors
        # STE allows gradients to flow to all continuous weight coordinates
        grad_weight = grad_output.clone()
        return grad_weight, None


class ADMMMaskedLinear(nn.Module):
    """
    Linear layer maintaining continuous weights and an ADMM-derived binary mask with STE gradient backprop.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        self.register_buffer("mask", torch.ones(out_features, in_features, dtype=dtype, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, mask: torch.Tensor) -> ADMMMaskedLinear:
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        layer.weight.data.copy_(linear.weight.data)
        layer.mask.copy_(mask.to(dtype=linear.weight.dtype, device=linear.weight.device))
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_masked = ADMMSTEPrune.apply(self.weight, self.mask)
        return F.linear(x, w_masked, self.bias)


class ADMMPruner(BaseCompressor):
    """
    ADMM Pruner supporting unstructured sparsity, block-wise structured sparsity,
    and 2:4 semi-structured tensor core acceleration with STE gradients and dynamic rho scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        rho: float = 0.1,
        max_iter: int = 6,
        structured_24: bool = True,
        block_size: Tuple[int, int] = (4, 4),
        dynamic_rho: bool = True,
        use_ste: bool = True,
        recalibrate_ln: bool = True,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="admm_pruner", **kwargs)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.initial_rho = float(rho)
        self.max_iter = max(1, int(max_iter))
        self.structured_24 = structured_24
        self.block_size = block_size
        self.dynamic_rho = dynamic_rho
        self.use_ste = use_ste
        self.recalibrate_ln = recalibrate_ln
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.stats: Dict[str, Any] = {}

    def _project_sparse(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects continuous weights onto sparse constraint set and returns (pruned_tensor, binary_mask)."""
        if self.structured_24:
            pruned, mask = apply_24_structured_mask(tensor, saliency=torch.abs(tensor), dim=1)
            return pruned, mask
        elif self.block_size != (1, 1):
            pruned, mask = apply_block_mask(tensor, block_size=self.block_size, sparsity=self.sparsity)
            return pruned, mask
        else:
            flat = tensor.abs().reshape(-1)
            k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
            thr = torch.topk(flat, k, largest=True).values[-1]
            mask = (tensor.abs() >= thr).float()
            return tensor * mask, mask

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes ADMM pruning with dynamic rho scheduling, STE integration, and LayerNorm recalibration."""
        pruned_layers_count = 0

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                W0 = module.weight.data.clone().float()
                device = W0.device
                dtype = module.weight.dtype

                W = W0.clone()
                Z, mask = self._project_sparse(W0)
                U = torch.zeros_like(W0)

                rho = self.initial_rho
                mu = 10.0
                tau_incr = 1.5
                tau_decr = 1.5
                rho_min, rho_max = 1e-4, 100.0

                for it in range(self.max_iter):
                    Z_prev = Z.clone()

                    # 1. Update Z (Sparse projection)
                    Z, mask = self._project_sparse(W + U)

                    # 2. Update W
                    W = (W0 + rho * (Z - U)) / (1.0 + rho)

                    # 3. Update Dual variable U
                    U = U + W - Z

                    # 4. Dynamic Rho Scheduling based on primal/dual residuals
                    if self.dynamic_rho:
                        primal_res = torch.norm(W - Z).item()
                        dual_res = (rho * torch.norm(Z - Z_prev)).item()
                        if primal_res > mu * (dual_res + 1e-8):
                            rho = min(rho * tau_incr, rho_max)
                        elif dual_res > mu * (primal_res + 1e-8):
                            rho = max(rho / tau_decr, rho_min)

                # Final assignment to projection Z and STE Module conversion
                # Always install the trainable masked wrapper.  The legacy
                # branch that assigned ``module.weight.data = Z`` permanently
                # destroyed recoverability and bypassed gradients.
                admm_layer = ADMMMaskedLinear.from_linear(module, mask=mask)
                if name == "":
                    self.model = admm_layer
                else:
                    replace_module(self.model, name, admm_layer)

                pruned_layers_count += 1

        # LayerNorm Recalibration
        if self.recalibrate_ln:
            recalibrate_layernorms(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "sparsity_target": self.sparsity,
            "structured_24": self.structured_24,
            "admm_iterations": self.max_iter,
            "dynamic_rho": self.dynamic_rho,
            "use_ste": self.use_ste,
            "recalibrate_ln": self.recalibrate_ln,
            "method": "admm_pruner_ste_dynamic_rho",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("admm_pruner")
def admm_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ADMM pruner."""
    sparsity = getattr(comp_config, "admm_pruner_sparsity", getattr(comp_config, "sparsity", 0.5)) if comp_config else kwargs.get("sparsity", 0.5)
    rho = getattr(comp_config, "admm_pruner_rho", getattr(comp_config, "rho", 0.1)) if comp_config else kwargs.get("rho", 0.1)
    max_iter = getattr(comp_config, "admm_pruner_max_iter", getattr(comp_config, "max_iter", 6)) if comp_config else kwargs.get("max_iter", 6)
    structured_24 = getattr(comp_config, "admm_pruner_structured_24", getattr(comp_config, "structured_24", True)) if comp_config else kwargs.get("structured_24", True)
    dynamic_rho = getattr(comp_config, "admm_pruner_dynamic_rho", kwargs.get("dynamic_rho", True))
    use_ste = getattr(comp_config, "admm_pruner_use_ste", kwargs.get("use_ste", True))
    recalibrate_ln = getattr(comp_config, "admm_pruner_recalibrate_ln", kwargs.get("recalibrate_ln", True))

    compressor = ADMMPruner(
        model,
        sparsity=sparsity,
        rho=rho,
        max_iter=max_iter,
        structured_24=structured_24,
        dynamic_rho=dynamic_rho,
        use_ste=use_ste,
        recalibrate_ln=recalibrate_ln,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

__all__ = [
    "ADMMSTEPrune",
    "ADMMMaskedLinear",
    "ADMMPruner",
    "admm_pruner_factory",
]
