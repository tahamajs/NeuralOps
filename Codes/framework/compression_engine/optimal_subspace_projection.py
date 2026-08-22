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


class OptimalSubspaceProjectionCompressor(BaseCompressor):
    """
    Optimal Subspace Projection Compressor using activation-aware randomized SVD.
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
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="optimal_subspace_projection", **kwargs)
        self.calibration_loader = calibration_loader
        self.rank_ratio = max(0.01, min(float(rank_ratio), 1.0))
        self.power_iters = max(0, int(power_iters))
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.stats: Dict[str, Any] = {}

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
            if isinstance(module, nn.Linear) and not any(
                skip in name.lower() for skip in ["embeddings", "classifier", "lm_head"]
            ):
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

        compressed_layers_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and not any(
                skip in name.lower() for skip in ["embeddings", "classifier", "lm_head"]
            ):
                W = module.weight.data.clone().float()
                d_out, d_in = W.shape
                device = W.device
                dtype = module.weight.dtype

                C = act_covs.get(id(module), None)
                if C is not None and C.shape[-1] == d_in:
                    C = C.to(device)
                    diag_mean = torch.diagonal(C).mean().item()
                    reg = max(1e-4 * diag_mean, 1e-5)
                    C_reg = C + reg * torch.eye(d_in, device=device)
                    try:
                        L = torch.linalg.cholesky(C_reg)
                        L_inv = torch.linalg.inv(L)
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
                    module.weight.data = new_weight.to(dtype=dtype, device=device)
                    compressed_layers_count += 1
                except Exception as e:
                    logger.debug("Subspace projection SVD exception in %s: %s", name, e)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": compressed_layers_count,
            "rank_ratio": self.rank_ratio,
            "power_iters": self.power_iters,
            "method": "optimal_subspace_projection",
        })
        self.stats = stats
        return self.model, stats


# Alias
SubspaceProjectionCompressor = OptimalSubspaceProjectionCompressor


@COMPRESSOR_REGISTRY.register("optimal_subspace_projection")
def optimal_subspace_projection_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Optimal Subspace Projection factory."""
    calibration_loader = getattr(comp_config, "calibration_loader", kwargs.get("calibration_loader", None))
    rank_ratio = getattr(comp_config, "optimal_subspace_rank_ratio", getattr(comp_config, "rank_ratio", 0.5)) if comp_config else kwargs.get("rank_ratio", 0.5)
    power_iters = getattr(comp_config, "optimal_subspace_power_iters", getattr(comp_config, "power_iters", 2)) if comp_config else kwargs.get("power_iters", 2)
    tokenizer = getattr(model, "tokenizer", None) or (getattr(comp_config, "tokenizer", None) if comp_config else kwargs.get("tokenizer", None))

    compressor = OptimalSubspaceProjectionCompressor(
        model,
        calibration_loader=calibration_loader,
        rank_ratio=rank_ratio,
        power_iters=power_iters,
        tokenizer=tokenizer,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
