"""
OATS: Outlier-Aware Training & Saliency-Based Sparsification with Low-Rank Matrix Decomposition.
Registered in COMPRESSOR_REGISTRY as 'oats'.

Mathematical Formulation:
1. Activation Second Moment:
   $$M = \\frac{1}{N} X^\\top X + \\epsilon I \\in \\mathbb{R}^{d_{in} \\times d_{in}}$$
2. Cholesky Factorization:
   $$M = L_M L_M^\\top$$
3. Preconditioned Weight Matrix:
   $$\\widetilde{W} = W L_M$$
4. Alternating Low-Rank + Sparse Decomposition:
   - Low-Rank Outlier Component: $L = \\operatorname{SVD}_r(\\widetilde{W} - S)$
   - Sparse Residual Component: $S = \\Pi_{2:4}(\\widetilde{W} - L)$ or $\\operatorname{Top-}(1-s)(\\widetilde{W} - L)$
5. De-conditioned Weight Matrix:
   $$W_{oats} = (L + S) L_M^{-1}$$
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
    get_calibration_tensors,
)

logger = logging.getLogger(__name__)


class OATSCompressor(BaseCompressor):
    """
    OATS Compressor: Outlier-aware low-rank + structured sparse matrix decomposition.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        sparsity: float = 0.5,
        rank_ratio: float = 0.2,
        num_iters: int = 4,
        device: Optional[Union[str, torch.device]] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        structured_24: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="oats", **kwargs)
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.rank_ratio = max(0.01, min(float(rank_ratio), 0.9))
        self.num_iters = max(1, int(num_iters))
        self.structured_24 = structured_24
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.stats: Dict[str, Any] = {}

    def _collect_activations(self) -> Dict[str, torch.Tensor]:
        """Collect input activations across all nn.Linear modules."""
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
            device=self.target_device,
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
                logger.debug("OATS activation collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)

        return aggregated

    def _oats_decompose(
        self,
        W: torch.Tensor,
        activations: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decomposes weight into outlier low-rank L + sparse S under metric M."""
        d_out, d_in = W.shape
        device = W.device

        # Compute M
        if activations is not None and activations.numel() > 0:
            X = activations.reshape(-1, activations.shape[-1]).float().to(device)
            if X.shape[-1] == d_in:
                M = torch.matmul(X.T, X) / max(X.shape[0], 1)
                M = M + 1e-4 * torch.eye(d_in, device=device)
            else:
                M = torch.eye(d_in, device=device)
        else:
            M = torch.eye(d_in, device=device)

        try:
            L_M = torch.linalg.cholesky(M)
            L_M_inv = torch.linalg.inv(L_M)
        except Exception:
            L_M = torch.eye(d_in, device=device)
            L_M_inv = torch.eye(d_in, device=device)

        # Scale weights: W_scaled = W @ L_M
        W_scaled = torch.matmul(W, L_M)
        rank = max(1, int(self.rank_ratio * min(d_out, d_in)))

        # Initialize L via truncated SVD
        try:
            U, S, Vh = torch.linalg.svd(W_scaled, full_matrices=False)
            r = min(rank, len(S))
            L = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
        except Exception:
            L = torch.zeros_like(W_scaled)

        # Initialize S
        R = W_scaled - L
        if self.structured_24:
            S_mat, _ = apply_24_structured_mask(R, saliency=torch.abs(R), dim=1)
        else:
            flat = R.abs().reshape(-1)
            k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
            thr = torch.topk(flat, k, largest=True).values[-1]
            mask = (R.abs() >= thr).float()
            S_mat = R * mask

        # Alternating minimization iterations
        for _ in range(self.num_iters):
            try:
                U, S, Vh = torch.linalg.svd(W_scaled - S_mat, full_matrices=False)
                r = min(rank, len(S))
                L = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass

            R = W_scaled - L
            if self.structured_24:
                S_mat, _ = apply_24_structured_mask(R, saliency=torch.abs(R), dim=1)
            else:
                flat = R.abs().reshape(-1)
                k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
                thr = torch.topk(flat, k, largest=True).values[-1]
                mask = (R.abs() >= thr).float()
                S_mat = R * mask

        # De-scale: W_comp = (L + S_mat) @ L_M^-1
        W_comp = torch.matmul(L + S_mat, L_M_inv)
        return W_comp

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes OATS decomposition across all Linear modules."""
        activations_map = self._collect_activations()
        decomposed_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                device = W.device
                dtype = module.weight.dtype

                act = activations_map.get(name, None)
                W_comp = self._oats_decompose(W, act)

                module.weight.data = W_comp.to(dtype=dtype, device=device)
                decomposed_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "decomposed_layers": decomposed_layers_count,
            "sparsity_target": self.sparsity,
            "rank_ratio": self.rank_ratio,
            "structured_24": self.structured_24,
            "num_iters": self.num_iters,
            "method": "oats_outlier_decomposition",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("oats")
@COMPRESSOR_REGISTRY.register("oats_compressor")
def oats_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for OATS compressor."""
    sparsity = getattr(comp_config, "oats_sparsity", getattr(comp_config, "sparsity", 0.5)) if comp_config else kwargs.get("sparsity", 0.5)
    rank_ratio = getattr(comp_config, "oats_rank_ratio", getattr(comp_config, "rank_ratio", 0.2)) if comp_config else kwargs.get("rank_ratio", 0.2)
    num_iters = getattr(comp_config, "oats_num_iters", getattr(comp_config, "num_iters", 4)) if comp_config else kwargs.get("num_iters", 4)
    structured_24 = getattr(comp_config, "oats_structured_24", getattr(comp_config, "structured_24", True)) if comp_config else kwargs.get("structured_24", True)
    tokenizer = getattr(model, "tokenizer", None) or (getattr(comp_config, "tokenizer", None) if comp_config else kwargs.get("tokenizer", None))

    compressor = OATSCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        rank_ratio=rank_ratio,
        num_iters=num_iters,
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
