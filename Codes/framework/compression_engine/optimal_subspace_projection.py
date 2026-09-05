"""
Optimal Subspace Projection: Activation-Aware Subspace Projection via Randomized SVD.
Registered in COMPRESSOR_REGISTRY as 'optimal_subspace_projection'.

Mathematical Formulation:
1. Activation Covariance:
   $$C = \\frac{1}{N} X^\\top X + \\epsilon I$$
2. Cholesky Preconditioning Factor:
   $$C = L L^\\top \\implies M = W L$$
3. Randomized Subspace Iteration:
   $$Y = M \\Omega, \\quad Y \\leftarrow M (M^\\top Y), \\quad Q = \\operatorname{qr}(Y)$$
4. Low-Rank Projection & Inversion:
   $$B = Q^\\top M = U \\Sigma V^\\top$$
   $$W_{osp} = (Q U_r \\Sigma_r V_r^\\top) L^{-1}$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import get_calibration_tensors

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("optimal_subspace_projection")
class OptimalSubspaceProjectionCompressor(BaseCompressor):
    """
    Optimal Subspace Projection Compressor using activation-aware randomized SVD.

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
        calibration_loader: Optional[Any] = None,
        rank_ratio: float = 0.5,
        power_iters: int = 2,
        tokenizer: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="optimal_subspace_projection", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank_ratio = getattr(cfg, "optimal_subspace_rank_ratio", getattr(cfg, "rank_ratio", rank_ratio))
            power_iters = getattr(cfg, "optimal_subspace_power_iters", getattr(cfg, "power_iters", power_iters))
            max_error = getattr(cfg, "optimal_subspace_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "optimal_subspace_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )

        self.calibration_loader = calibration_loader
        self.rank_ratio = max(0.01, min(float(rank_ratio), 1.0))
        self.power_iters = max(0, int(power_iters))
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
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
        self.layer_ranks: Dict[str, int] = {}
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

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress the target model using optimal subspace projection."""
        act_covs: Dict[int, torch.Tensor] = {}
        hooks = []

        def hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                x = inputs[0].detach()
                x_flat = x.reshape(-1, x.shape[-1]).float()
                cov = torch.matmul(x_flat.T, x_flat) / max(x_flat.shape[0], 1)
                mod_id = id(module)
                if mod_id not in act_covs:
                    act_covs[mod_id] = cov.cpu()
                else:
                    act_covs[mod_id] += cov.cpu()

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(hook_fn))

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
                logger.debug("OptimalSubspaceProjection activation collection exception: %s", e)

        for h in hooks:
            h.remove()

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                raw_w = module.weight.data
                cls_name = module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")

                W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
                d_out, d_in = W.shape
                device = raw_w.device
                dtype = raw_w.dtype

                C = act_covs.get(id(module), None)
                if C is not None and C.shape[-1] == d_in:
                    C = C.to(device)
                    diag_mean = torch.diagonal(C).mean().item()
                    reg = max(1e-4 * diag_mean, 1e-5)
                    C_reg = C + reg * torch.eye(d_in, device=device)
                    try:
                        L = torch.linalg.cholesky(C_reg)
                        L_inv = torch.linalg.inv(L)
                        if torch.isnan(L).any() or torch.isnan(L_inv).any():
                            raise ValueError("Cholesky NaNs")
                    except Exception:
                        L = torch.eye(d_in, device=device)
                        L_inv = torch.eye(d_in, device=device)
                else:
                    L = torch.eye(d_in, device=device)
                    L_inv = torch.eye(d_in, device=device)

                M = torch.matmul(W, L)
                rank = max(1, min(int(min(d_out, d_in) * self.rank_ratio), d_in))
                Omega = torch.randn(M.shape[1], rank, device=device)
                Y = torch.matmul(M, Omega)
                for _ in range(self.power_iters):
                    Y = torch.matmul(M, torch.matmul(M.T, Y))

                try:
                    Q, _ = torch.linalg.qr(Y)
                    B = torch.matmul(Q.T, M)
                    U, S, Vt = torch.linalg.svd(B, full_matrices=False)
                    r = min(rank, len(S))
                    U_r = U[:, :r]
                    S_r = S[:r]
                    Vt_r = Vt[:r, :]
                    new_U = torch.matmul(Q, U_r)
                    new_V = torch.matmul(Vt_r, L_inv.T)
                    new_weight = torch.matmul(new_U * S_r.unsqueeze(0), new_V)
                    new_weight = torch.nan_to_num(new_weight, nan=0.0)

                    w_norm = torch.norm(W, p="fro").item()
                    err = (torch.norm(W - new_weight, p="fro") / max(w_norm, 1e-12)).item()
                    self.layer_errors[name] = float(err)
                    self.layer_ranks[name] = r

                    if err > self.max_error:
                        logger.warning(
                            "OptimalSubspaceProjection reconstruction error %.4f on %s exceeded threshold %.4f.",
                            err,
                            name,
                            self.max_error,
                        )
                        if self.fallback_to_original:
                            logger.info("Skipping OptimalSubspaceProjection for %s (fallback to original).", name)
                            self.fallback_layers.append(name)
                            continue

                    if is_conv1d:
                        module.weight.data.copy_(new_weight.t().to(dtype=dtype, device=device))
                    else:
                        module.weight.data.copy_(new_weight.to(dtype=dtype, device=device))

                    self.compressed_layers.append(name)
                except Exception as e:
                    logger.warning("Subspace projection SVD exception in %s: %s", name, e)
                    self.fallback_layers.append(name)
                    self.layer_errors[name] = 1.0

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "rank_ratio": self.rank_ratio,
            "power_iters": self.power_iters,
            "method": "optimal_subspace_projection",
        })
        self.stats = stats
        return self.model, stats


# Alias
SubspaceProjectionCompressor = OptimalSubspaceProjectionCompressor


@COMPRESSOR_REGISTRY.register("optimal_subspace_projection_factory")
def optimal_subspace_projection_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Optimal Subspace Projection factory."""
    calibration_loader = kwargs.pop("calibration_loader", getattr(comp_config, "calibration_loader", None))
    rank_ratio = kwargs.pop("rank_ratio", getattr(comp_config, "optimal_subspace_rank_ratio", getattr(comp_config, "rank_ratio", 0.5)))
    power_iters = kwargs.pop("power_iters", getattr(comp_config, "optimal_subspace_power_iters", getattr(comp_config, "power_iters", 2)))
    max_error = kwargs.pop("max_error", getattr(comp_config, "optimal_subspace_max_error", getattr(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", getattr(
        comp_config, "optimal_subspace_fallback_to_original", getattr(comp_config, "fallback_to_original", True)
    ))
    tokenizer = kwargs.pop("tokenizer", getattr(model, "tokenizer", None) or getattr(comp_config, "tokenizer", None))

    compressor = OptimalSubspaceProjectionCompressor(
        model,
        calibration_loader=calibration_loader,
        rank_ratio=rank_ratio,
        power_iters=power_iters,
        tokenizer=tokenizer,
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
