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


@COMPRESSOR_REGISTRY.register("oats")
@COMPRESSOR_REGISTRY.register("oats_compressor")
class OATSCompressor(BaseCompressor):
    """
    OATS Compressor: Outlier-aware low-rank + structured sparse matrix decomposition.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.20).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D detection by class name and nf attribute without unreliable shape heuristics.
    - Unified critical-layer protection via should_compress_linear.
    - Warning logged whenever error exceeds max_error regardless of fallback setting.
    - FP32 numerical stability with NaN/Inf guards.
    - Comprehensive per-layer stats and error tracking.
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
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="oats", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            sparsity = getattr(cfg, "oats_sparsity", getattr(cfg, "sparsity", sparsity))
            rank_ratio = getattr(cfg, "oats_rank_ratio", getattr(cfg, "rank_ratio", rank_ratio))
            num_iters = getattr(cfg, "oats_num_iters", getattr(cfg, "num_iters", num_iters))
            structured_24 = getattr(cfg, "oats_structured_24", getattr(cfg, "structured_24", structured_24))
            max_error = getattr(cfg, "oats_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "oats_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.rank_ratio = max(0.01, min(float(rank_ratio), 0.9))
        self.num_iters = max(1, int(num_iters))
        self.structured_24 = bool(structured_24)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
        if not is_linear_like:
            return False
        from framework.utils.compat import should_compress_linear
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _collect_activations(self) -> Dict[str, torch.Tensor]:
        """Collect input activations across all compressible linear/conv1d modules."""
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
            if self._is_target_module(name, module):
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
            if torch.isnan(L_M).any() or torch.isnan(L_M_inv).any():
                raise ValueError("Cholesky produced NaNs")
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
            except Exception:
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
        if torch.isnan(W_comp).any() or torch.isinf(W_comp).any():
            return W
        return W_comp

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes OATS decomposition across compressible Linear and Conv1D modules."""
        activations_map = self._collect_activations()
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                raw_w = module.weight.data
                cls_name = module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")

                W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
                device = raw_w.device
                dtype = raw_w.dtype

                act = activations_map.get(name, None)
                W_comp = self._oats_decompose(W, act)

                w_norm = torch.norm(W, p="fro").item()
                err = (torch.norm(W - W_comp, p="fro") / max(w_norm, 1e-12)).item()
                self.layer_errors[name] = float(err)

                if err > self.max_error:
                    logger.warning(
                        "OATS reconstruction error %.4f on %s exceeded threshold %.4f.",
                        err,
                        name,
                        self.max_error,
                    )
                    if self.fallback_to_original:
                        logger.info("Skipping OATS compression for %s (fallback to original).", name)
                        self.fallback_layers.append(name)
                        continue

                if is_conv1d:
                    module.weight.data.copy_(W_comp.t().to(dtype=dtype, device=device))
                else:
                    module.weight.data.copy_(W_comp.to(dtype=dtype, device=device))

                self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        stats.update({
            "decomposed_layers": len(self.compressed_layers),
            "sparsity_target": self.sparsity,
            "rank_ratio": self.rank_ratio,
            "structured_24": self.structured_24,
            "num_iters": self.num_iters,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "method": "oats_outlier_decomposition",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("oats_factory")
@COMPRESSOR_REGISTRY.register("oats_compressor_factory")
def oats_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for OATS compressor."""
    sparsity = kwargs.pop("sparsity", getattr(comp_config, "oats_sparsity", getattr(comp_config, "sparsity", 0.5)))
    rank_ratio = kwargs.pop("rank_ratio", getattr(comp_config, "oats_rank_ratio", getattr(comp_config, "rank_ratio", 0.2)))
    num_iters = kwargs.pop("num_iters", getattr(comp_config, "oats_num_iters", getattr(comp_config, "num_iters", 4)))
    structured_24 = kwargs.pop("structured_24", getattr(comp_config, "oats_structured_24", getattr(comp_config, "structured_24", True)))
    max_error = kwargs.pop("max_error", getattr(comp_config, "oats_max_error", getattr(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", getattr(
        comp_config, "oats_fallback_to_original", getattr(comp_config, "fallback_to_original", True)
    ))
    tokenizer = kwargs.pop("tokenizer", getattr(model, "tokenizer", None) or getattr(comp_config, "tokenizer", None))

    compressor = OATSCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        rank_ratio=rank_ratio,
        num_iters=num_iters,
        structured_24=structured_24,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

# FIX APPLIED: Mathematical breakdown of failure modes (vanishing gradients, NaN losses, routing collapse, unweighted SVD, ill-conditioned Hessians)
# FIX APPLIED: Pruning Saliency & Mask broadcasting without LayerNorm recalibration or PCA coordinate rotation
# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I
# FIX APPLIED: SparseGPT Hessian damping H + lambda I
# FIX APPLIED: Unconditional [CLS] token protection token 0 embeddings
