"""
Adaptive Ranks SVD Compressor: dynamic rank selection per layer based on singular value energy spectra.

Dynamic cumulative singular value energy thresholding:
    $$\\frac{\\sum_{i=1}^r \\sigma_i^2}{\\sum_{j=1}^d \\sigma_j^2} \\ge \\tau \\quad (\\tau = 0.92)$$
with minimum rank floor:
    $$r_{\\min} = \\max(16, \\lfloor 0.15 \\cdot \\min(d_{in}, d_{out}) \\rfloor)$$
activation covariance whitening:
    $$M = W \\cdot \\operatorname{Cov}(X)^{1/2}$$
and symmetrical $\\sqrt{\\Sigma}$ singular value distribution:
    $$U_r = U_{:, :r} \\sqrt{\\Sigma_r}, \\quad V_r = (\\sqrt{\\Sigma_r} V^\\top_{:r, :}) \\cdot \\operatorname{Cov}(X)^{-1/2}$$
Registered as 'adaptive_ranks' and 'adaptive_svd'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.attentions.compressed_attention import CompressedAttentionLayer
from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.compression_engine.pruning_utils import get_calibration_tensors
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("adaptive_ranks")
@COMPRESSOR_REGISTRY.register("adaptive_svd")
class AdaptiveRanksCompressor(BaseCompressor):
    r"""
    Adaptive Ranks SVD Compressor.

    Dynamically determines ranks per layer according to cumulative singular value energy threshold:
        $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
    with minimum rank floor $r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{in}, d_{out}) \rfloor)$,
    activation covariance whitening, and symmetrically distributed singular values:
        $$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = (\sqrt{\Sigma_r} V^\top_{:r, :}) \cdot \operatorname{Cov}(X)^{-1/2}$$
    """

    def __init__(
        self,
        model: nn.Module,
        energy_threshold: float = 0.92,
        min_rank: int = 16,
        target_modules: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="adaptive_ranks", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            energy_threshold = getattr(cfg, "energy_threshold", energy_threshold)
            min_rank = getattr(cfg, "min_rank", getattr(cfg, "rank", min_rank))

        if "rank" in kwargs and kwargs["rank"] is not None:
            min_rank = kwargs["rank"]

        self.energy_threshold = float(energy_threshold)
        self.min_rank = max(16, int(min_rank))
        self.target_modules = target_modules
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, (SVDLinear, CompressedAttentionLayer)):
            return False
        if not isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            return False
        name_lower = name.lower()
        if any(skip in name_lower for skip in ["lm_head", "classifier", "head", "output_layer", "score"]):
            return False
        if isinstance(module, nn.Linear) and min(module.in_features, module.out_features) < 16:
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _compute_energy_rank(self, S: torch.Tensor, max_rank: int, min_rank_floor: Optional[int] = None) -> int:
        r"""
        Find smallest rank $r$ satisfying cumulative singular energy ratio $\ge$ threshold:
        $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
        constrained by minimum rank floor.
        """
        r_floor = min_rank_floor if min_rank_floor is not None else self.min_rank
        # Hard safety floor applies even to direct callers that do not pass the
        # compressor's precomputed floor.
        r_floor = max(self.min_rank, 16, int(r_floor))
        r_floor = max(1, min(r_floor, max_rank))
        s_squared = S ** 2
        total_energy = s_squared.sum()
        if total_energy <= 1e-12:
            return r_floor

        cum_energy = torch.cumsum(s_squared, dim=0) / total_energy
        idx = torch.where(cum_energy >= self.energy_threshold)[0]
        r = int(idx[0].item()) + 1 if len(idx) > 0 else max_rank
        return max(1, min(max(r, r_floor), max_rank))

    def _collect_activation_covariances(self) -> Dict[str, torch.Tensor]:
        """Collect empirical input activation covariances across target nn.Linear modules."""
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
            if isinstance(module, nn.Linear) and self._is_target_module(name, module):
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
                logger.debug("AdaptiveRanks covariance collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        covariances: Dict[str, torch.Tensor] = {}
        for name, cov_sum in running_cov.items():
            n = float(max(sample_counts.get(name, 1), 1))
            covariances[name] = cov_sum / n

        return covariances

    def compress_linear(self, layer: nn.Linear, cov: Optional[torch.Tensor] = None) -> SVDLinear:
        """Decompose a single linear layer via energy-thresholded SVD with covariance whitening."""
        out_f = layer.out_features
        in_f = layer.in_features
        min_dim = min(out_f, in_f)
        device = layer.weight.device
        dtype = layer.weight.dtype

        # Enforce minimum rank floor: r_min = max(16, floor(0.15 * min(d_in, d_out)))
        r_floor = max(self.min_rank, 16, int(0.15 * min_dim))
        r_floor = max(1, min(r_floor, min_dim))

        W = layer.weight.data.float()

        if cov is not None and cov.shape == (in_f, in_f):
            try:
                cov_dev = cov.to(device=device, dtype=torch.float32)
                diag_mean = torch.diagonal(cov_dev).mean().item()
                if not math.isfinite(diag_mean) or diag_mean <= 0.0:
                    diag_mean = 1.0
                lambda_reg = max(1e-4 * diag_mean, 1e-6)
                cov_reg = cov_dev + lambda_reg * torch.eye(in_f, device=device)

                eigvals, eigvecs = torch.linalg.eigh(cov_reg)
                eigvals_clamped = torch.clamp(eigvals, min=1e-6)

                # Cov(X)^{1/2} and Cov(X)^{-1/2}
                cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_clamped)) @ eigvecs.T
                cov_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals_clamped)) @ eigvecs.T

                # Whiten weights: M = W * Cov(X)^{1/2}
                M = torch.matmul(W, cov_sqrt)
                U, S, Vt = torch.linalg.svd(M, full_matrices=False)

                r = self._compute_energy_rank(S, min_dim, min_rank_floor=r_floor)
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))

                # Symmetric singular value distribution: U_r = U * sqrt(S), V_r = (sqrt(S) * V^T) * Cov(X)^{-1/2}
                U_r = U[:, :r] * sqrt_S.unsqueeze(0)
                V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]) @ cov_inv_sqrt
                # Symmetrically normalize factor magnitudes to prevent scale disparity
                norm_u = torch.norm(U_r, p="fro").clamp_min(1e-8)
                norm_v = torch.norm(V_r, p="fro").clamp_min(1e-8)
                scale = torch.sqrt(norm_u / norm_v)
                U_r = (U_r / scale).to(dtype=dtype, device=device)
                V_r = (V_r * scale).to(dtype=dtype, device=device)

                svd_layer = SVDLinear(
                    in_features=in_f,
                    out_features=out_f,
                    rank=r,
                    bias=layer.bias is not None,
                    dtype=dtype,
                    device=device,
                )
                svd_layer.set_low_rank_factors(U_r, V_r)
                if layer.bias is not None:
                    svd_layer.bias.data.copy_(layer.bias.data)

                return svd_layer
            except Exception as e:
                logger.debug("Covariance whitening exception on linear layer: %s", e)

        # Fallback to direct SVD with symmetric distribution and rank floor
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        r = self._compute_energy_rank(S, min_dim, min_rank_floor=r_floor)
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=layer.bias is not None,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(U_r, V_r)
        if layer.bias is not None:
            svd_layer.bias.data.copy_(layer.bias.data)

        return svd_layer

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress target layers with activation covariance whitening."""
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("AdaptiveRanksCompressor: No compressible layers found.")
            return self.model, self.get_compression_stats()

        covariances = self._collect_activation_covariances()

        for name, module in target_layers:
            if isinstance(module, nn.MultiheadAttention):
                try:
                    embed_dim = module.embed_dim
                    r_floor = max(self.min_rank, 16, int(0.15 * embed_dim))
                    r = max(1, min(r_floor, embed_dim))
                    if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
                        _, S, _ = torch.linalg.svd(module.in_proj_weight.data.float(), full_matrices=False)
                        r = self._compute_energy_rank(S, min(embed_dim, 3 * embed_dim), min_rank_floor=r_floor)
                    compressed = CompressedAttentionLayer(
                        original_layer=module,
                        rank_qkv=r,
                        rank_out=r,
                    )
                    replace_module(self.model, name, compressed)
                except Exception as e:
                    logger.warning("Failed to compress attention layer %s: %s", name, e)
            elif isinstance(module, nn.Linear):
                cov = covariances.get(name, None)
                svd_layer = self.compress_linear(module, cov=cov)
                replace_module(self.model, name, svd_layer)

        stats = self.get_compression_stats()
        return self.model, stats


# Backward compatibility alias
AdaptiveTransformerCompressor = AdaptiveRanksCompressor


def adaptive_ranks_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Adaptive ranks factory function."""
    energy_threshold = getattr(comp_config, "energy_threshold", kwargs.pop("energy_threshold", 0.92))
    min_rank = getattr(comp_config, "min_rank", kwargs.pop("min_rank", kwargs.pop("rank", 16)))
    compressor = AdaptiveRanksCompressor(
        model, energy_threshold=energy_threshold, min_rank=min_rank, **kwargs
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


adaptive_svd_factory = adaptive_ranks_factory

__all__ = [
    "AdaptiveRanksCompressor",
    "AdaptiveTransformerCompressor",
    "adaptive_ranks_factory",
    "adaptive_svd_factory",
]

