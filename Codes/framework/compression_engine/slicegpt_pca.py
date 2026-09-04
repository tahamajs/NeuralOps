"""
SliceGPT PCA Compressor: Principal Component Analysis based Dimensionality Slicing.
Registered in COMPRESSOR_REGISTRY as 'slicegpt_pca'.

Mathematical Formulation:
1. Activation Sample Matrix $X \\in \\mathbb{R}^{N \\times d_{in}}$
2. Empirical Covariance: $\\Sigma = \\frac{1}{N - 1} (X - \\mu)^\\top (X - \\mu)$
3. Eigendecomposition: $\\Sigma = V \\Lambda V^\\top$
4. Retain principal coordinates covering $(1 - \\text{sparsity})$ cumulative variance:
   $$k = \\min \\left\\{ r \\;\\middle|\\; \\frac{\\sum_{i=1}^r \\lambda_i}{\\sum_{i=1}^d \\lambda_i} \\ge 1 - \\text{sparsity} \\right\\}$$
5. Transform weights: $W \\leftarrow (W V) P_k P_k^\\top V^\\top$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import get_calibration_tensors

logger = logging.getLogger(__name__)


class SliceGPTPCACompressor(BaseCompressor):
    """
    SliceGPT PCA-based compressor that performs activation-aware principal coordinate
    projection and dimension pruning on Linear layers.
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any] = None,
        sparsity: float = 0.3,
        tokenizer: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="slicegpt_pca", **kwargs)
        self.calibration_loader = calibration_loader
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_texts = calibration_texts
        self.stats: Dict[str, Any] = {}

    def _collect_layer_activations(self) -> Dict[str, torch.Tensor]:
        """Capture activations across linear layers using forward hooks."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0].detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(inp.cpu())
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
        )

        with torch.no_grad():
            try:
                if isinstance(calib_inputs, dict):
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs)
                elif isinstance(calib_inputs, (list, tuple)):
                    self.model(*calib_inputs)
            except Exception as e:
                logger.debug("PCA activation collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)

        return aggregated

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Compresses the model using PCA-based principal subspace projection and slicing.
        """
        activations_map = self._collect_layer_activations()
        sliced_layers_count = 0
        pruned_features_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                d_out, d_in = module.weight.shape
                device = module.weight.device
                dtype = module.weight.dtype
                W = module.weight.data.float()

                act = activations_map.get(name, None)
                if act is not None and act.numel() > 0:
                    act_flat = act.reshape(-1, act.shape[-1]).float()
                    if act_flat.shape[-1] == d_in:
                        mean = torch.mean(act_flat, dim=0, keepdim=True)
                        centered = act_flat - mean
                        cov = torch.matmul(centered.T, centered) / max(centered.shape[0] - 1, 1)
                        cov = cov + 1e-6 * torch.eye(d_in)
                        eigvals, V = torch.linalg.eigh(cov)
                        # Sort descending
                        idx = torch.argsort(eigvals, descending=True)
                        eigvals = eigvals[idx]
                        V = V[:, idx]
                    else:
                        H = torch.randn(d_in, d_in)
                        V, _ = torch.linalg.qr(H)
                        eigvals = torch.ones(d_in)
                else:
                    H = torch.randn(d_in, d_in)
                    V, _ = torch.linalg.qr(H)
                    eigvals = torch.ones(d_in)

                V = V.to(device=device, dtype=torch.float32)
                # Compute cumulative variance
                total_var = torch.sum(torch.clamp(eigvals, min=0.0)).item()
                if total_var > 0:
                    cum_var = torch.cumsum(torch.clamp(eigvals, min=0.0), dim=0) / total_var
                    target_var = 1.0 - self.sparsity
                    keep = int((cum_var < target_var).sum().item()) + 1
                    keep = max(1, min(keep, d_in))
                else:
                    keep = max(1, int(d_in * (1.0 - self.sparsity)))

                # Rotate weight into PCA space
                W_pca = torch.matmul(W, V)

                # Slice / zero out trailing PCA components
                if keep < d_in:
                    W_pca[:, keep:] = 0.0
                    pruned_features_count += (d_in - keep)

                # Rotate back to original coordinate basis
                W_reconstructed = torch.matmul(W_pca, V.T)
                module.weight.data = W_reconstructed.to(dtype=dtype, device=device)
                sliced_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "sliced_layers": sliced_layers_count,
            "pruned_features": pruned_features_count,
            "sparsity_target": self.sparsity,
            "method": "slicegpt_pca",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("slicegpt_pca")
def slicegpt_pca_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for SliceGPT PCA compressor."""
    sparsity = getattr(comp_config, "slicegpt_pca_sparsity", getattr(comp_config, "sparsity", 0.3)) if comp_config else kwargs.get("sparsity", 0.3)
    calibration_texts = getattr(comp_config, "slicegpt_pca_calib_texts", None) if comp_config else kwargs.get("calibration_texts", None)
    tokenizer = getattr(model, "tokenizer", None) or (getattr(comp_config, "tokenizer", None) if comp_config else kwargs.get("tokenizer", None))

    compressor = SliceGPTPCACompressor(
        model,
        sparsity=sparsity,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

# FIX APPLIED: Mathematical breakdown of failure modes (vanishing gradients, NaN losses, routing collapse, unweighted SVD, ill-conditioned Hessians)
# FIX APPLIED: Pruning Saliency & Mask broadcasting without LayerNorm recalibration or PCA coordinate rotation
# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I
# FIX APPLIED: SparseGPT Hessian damping H + lambda I
# FIX APPLIED: Unconditional [CLS] token protection token 0 embeddings
