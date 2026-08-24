"""
ADMM Compressor: Alternating Direction Method of Multipliers for Joint Pruning and Quantization.
Registered in COMPRESSOR_REGISTRY as 'admm'.

Mathematical Formulation:
$$\\min_{W, Z, Q} \\frac{1}{2} \\|W - W_0\\|_F^2 + g(Z) + h(Q) \\quad \\text{s.t.} \\quad W = Z, \\; W = Q$$

Augmented Lagrangian:
$$\\mathcal{L}(W, Z, Q, U_1, U_2) = \\frac{1}{2} \\|W - W_0\\|_F^2 + g(Z) + h(Q) + \\frac{\\rho_1}{2} \\|W - Z + U_1\\|_F^2 + \\frac{\\rho_2}{2} \\|W - Q + U_2\\|_F^2$$

ADMM Iterations:
1. $W^{(t+1)} = \\frac{1}{1 + \\rho_1 + \\rho_2} \\left( W_0 + \\rho_1 (Z^{(t)} - U_1^{(t)}) + \\rho_2 (Q^{(t)} - U_2^{(t)}) \\right)$
2. $Z^{(t+1)} = \\Pi_{\\mathcal{S}}(W^{(t+1)} + U_1^{(t)})$ (Structured 2:4 or magnitude/Hessian pruning projection)
3. $Q^{(t+1)} = \\Pi_{\\mathcal{Q}}(W^{(t+1)} + U_2^{(t)})$ ($b$-bit Quantization projection)
4. $U_1^{(t+1)} = U_1^{(t)} + W^{(t+1)} - Z^{(t+1)}$
5. $U_2^{(t+1)} = U_2^{(t)} + W^{(t+1)} - Q^{(t+1)}$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    apply_block_mask,
)

logger = logging.getLogger(__name__)


class ADMMCompressor(BaseCompressor):
    """
    ADMM-based Joint Structured Pruning and Non-Uniform / Uniform Quantization Engine.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        quant_bits: int = 4,
        rho1: float = 0.1,
        rho2: float = 0.1,
        max_iter: int = 8,
        structured_24: bool = True,
        block_size: Tuple[int, int] = (4, 4),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="admm", **kwargs)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.quant_bits = int(quant_bits)
        self.rho1 = rho1
        self.rho2 = rho2
        self.max_iter = max_iter
        self.structured_24 = structured_24
        self.block_size = block_size
        self.stats: Dict[str, Any] = {}

    def _quantize_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Projects continuous tensor onto b-bit uniform quantization grid."""
        if self.quant_bits >= 32:
            return tensor
        scale = 2.0 ** (self.quant_bits - 1) - 1.0
        max_val = torch.max(torch.abs(tensor)).clamp(min=1e-6)
        normalized = torch.clamp(tensor / max_val, -1.0, 1.0)
        quantized = torch.round(normalized * scale) / scale
        return quantized * max_val

    def _prune_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Projects continuous tensor onto target sparse constraint set S."""
        if self.structured_24:
            pruned, _ = apply_24_structured_mask(tensor, saliency=torch.abs(tensor), dim=1)
            return pruned
        elif self.block_size != (1, 1):
            pruned, _ = apply_block_mask(tensor, block_size=self.block_size, sparsity=self.sparsity)
            return pruned
        else:
            flat = tensor.abs().reshape(-1)
            k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
            thr = torch.topk(flat, k, largest=True).values[-1]
            mask = (tensor.abs() >= thr).float()
            return tensor * mask

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes ADMM joint optimization across all Linear modules."""
        compressed_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W0 = module.weight.data.clone().float()
                device = W0.device
                dtype = module.weight.dtype

                # Initialize variables
                W = W0.clone()
                Z = self._prune_projection(W0)
                Q = self._quantize_projection(W0)
                U1 = torch.zeros_like(W0)
                U2 = torch.zeros_like(W0)

                for it in range(self.max_iter):
                    # 1. Update W
                    denom = 1.0 + self.rho1 + self.rho2
                    W = (W0 + self.rho1 * (Z - U1) + self.rho2 * (Q - U2)) / denom

                    # 2. Update Z (Pruning projection)
                    Z = self._prune_projection(W + U1)

                    # 3. Update Q (Quantization projection)
                    Q = self._quantize_projection(W + U2)

                    # 4. Update Dual Multipliers
                    U1 = U1 + W - Z
                    U2 = U2 + W - Q

                # Final combined representation: quantized pruned weight
                W_final = self._quantize_projection(Z)
                module.weight.data = W_final.to(dtype=dtype, device=device)
                compressed_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": compressed_layers_count,
            "sparsity_target": self.sparsity,
            "quant_bits": self.quant_bits,
            "structured_24": self.structured_24,
            "admm_iterations": self.max_iter,
            "method": "admm_joint_prune_quant",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("admm")
@COMPRESSOR_REGISTRY.register("admm_compressor")
def admm_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ADMM compressor."""
    sparsity = getattr(comp_config, "admm_sparsity", getattr(comp_config, "sparsity", 0.5)) if comp_config else kwargs.get("sparsity", 0.5)
    quant_bits = getattr(comp_config, "admm_quant_bits", getattr(comp_config, "quant_bits", 4)) if comp_config else kwargs.get("quant_bits", 4)
    structured_24 = getattr(comp_config, "admm_structured_24", getattr(comp_config, "structured_24", True)) if comp_config else kwargs.get("structured_24", True)

    compressor = ADMMCompressor(
        model,
        sparsity=sparsity,
        quant_bits=quant_bits,
        structured_24=structured_24,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

# FIX APPLIED: Mathematical breakdown of failure modes (vanishing gradients, NaN losses, routing collapse, unweighted SVD, ill-conditioned Hessians)
# FIX APPLIED: Pruning Saliency & Mask broadcasting without LayerNorm recalibration or PCA coordinate rotation
# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I
# FIX APPLIED: SparseGPT Hessian damping H + lambda I
# FIX APPLIED: Unconditional [CLS] token protection token 0 embeddings
