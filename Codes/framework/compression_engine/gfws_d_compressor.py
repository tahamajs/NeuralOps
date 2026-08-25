"""GFWSVD: Fisher-weighted SVD. Registered as 'gfws' and 'gfws_d'."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.compression_engine.pruning_utils import get_calibration_tensors
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class GFWSCompressor(BaseCompressor):
    """
    Generalized Fisher / Covariance-Weighted SVD Compressor.
    Whitens weights using empirical input activation covariance square-root Cov(X)^{1/2},
    performs low-rank truncated SVD, and de-whitens to restore optimal compressed weights.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        damping: float = 0.01,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="gfws", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "gfws_rank", getattr(cfg, "rank", rank))
            damping = getattr(cfg, "gfws_damping", getattr(cfg, "damping", damping))
        self.rank = int(rank)
        self.damping = float(damping)
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )

    def _collect_activation_covariances(self) -> Dict[str, torch.Tensor]:
        """Collect empirical input activation covariances across nn.Linear modules."""
        self.model.eval()
        running_cov: Dict[str, torch.Tensor] = {}
        sample_counts: Dict[str, int] = {}
        hooks = []

        def make_hook(layer_name: str) -> Any:
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any) -> None:
                if len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach().float()
                    if x.ndim >= 2:
                        flat_x = x.reshape(-1, x.shape[-1])
                    else:
                        flat_x = x.unsqueeze(0)
                    n_samples = flat_x.shape[0]
                    if n_samples > 0:
                        batch_cov = torch.matmul(flat_x.T, flat_x).cpu()
                        if layer_name not in running_cov:
                            running_cov[layer_name] = batch_cov
                            sample_counts[layer_name] = n_samples
                        else:
                            running_cov[layer_name] += batch_cov
                            sample_counts[layer_name] += n_samples

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
                logger.debug("GFWS covariance collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        covariances: Dict[str, torch.Tensor] = {}
        for name, cov_sum in running_cov.items():
            n = float(max(sample_counts.get(name, 1), 1))
            covariances[name] = cov_sum / n

        return covariances

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers via real activation covariance singular decomposition W * Cov(X)^{1/2}."""
        covariances = self._collect_activation_covariances()
        compressed_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                d_out, d_in = W.shape
                device = module.weight.device
                dtype = module.weight.dtype
                r = max(1, min(self.rank, d_out, d_in))

                cov = covariances.get(name, None)
                if cov is not None and cov.shape == (d_in, d_in):
                    try:
                        cov_dev = cov.to(device=device, dtype=torch.float32)
                        diag_mean = torch.diagonal(cov_dev).mean().item()
                        if not math.isfinite(diag_mean) or diag_mean <= 0.0:
                            diag_mean = 1.0
                        lambda_reg = max(float(self.damping) * diag_mean, 1e-6)
                        cov_reg = cov_dev + lambda_reg * torch.eye(d_in, device=device)

                        eigvals, eigvecs = torch.linalg.eigh(cov_reg)
                        eigvals_clamped = torch.clamp(eigvals, min=1e-6)

                        # Cov(X)^{1/2} and Cov(X)^{-1/2}
                        cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_clamped)) @ eigvecs.T
                        cov_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals_clamped)) @ eigvecs.T

                        # Whiten weights: M = W * Cov(X)^{1/2}
                        M = torch.matmul(W, cov_sqrt)
                        U, S, Vt = torch.linalg.svd(M, full_matrices=False)

                        # Low-rank approximation of M: U_r * S_r * Vt_r
                        M_r = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]

                        # De-whiten: W_approx = M_r * Cov(X)^{-1/2}
                        W_approx = torch.matmul(M_r, cov_inv_sqrt)
                        module.weight.data = W_approx.to(dtype=dtype, device=device)
                        compressed_count += 1
                        continue
                    except Exception as e:
                        logger.debug("GFWS covariance decomposition exception on layer %s: %s", name, e)

                # Fallback to standard SVD
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                W_approx = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
                module.weight.data = W_approx.to(dtype=dtype, device=device)
                compressed_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "rank": self.rank,
            "damping": self.damping,
            "compressed_layers": compressed_count,
            "method": "covariance_weighted_svd",
        })
        return self.model, stats


@COMPRESSOR_REGISTRY.register("gfws")
@COMPRESSOR_REGISTRY.register("gfws_d")
def gfws_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """GFWS factory."""
    rank = getattr(comp_config, "gfws_rank", getattr(comp_config, "rank", kwargs.get("rank", 64)))
    damping = getattr(comp_config, "gfws_damping", getattr(comp_config, "damping", kwargs.get("damping", 0.01)))
    compressor = GFWSCompressor(model, rank=rank, damping=damping, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
