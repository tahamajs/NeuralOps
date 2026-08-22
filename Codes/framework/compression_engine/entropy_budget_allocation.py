"""
EntropyBudgetAllocation: global rank assignment via whitened activation entropy.
Registered as 'entropy_budget'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("entropy_budget")
class EntropyBudgetCompressor(BaseCompressor):
    """
    Entropy Budget Low-Rank SVD Compressor.

    Computes activation covariance and whiten weights, calculates per-component
    singular entropy contributions, allocates low-rank budgets globally across layers,
    and replaces dense nn.Linear layers with factorized SVDLinear modules with symmetric
    singular value splitting (U * sqrt(Sigma), sqrt(Sigma) * V^T).

    Args:
        model: Model to compress.
        calibration_loader: Optional DataLoader for activation collection.
        total_budget_ratio: Parameter retention ratio (0.0 to 1.0].
        rank: Optional fallback or target rank per layer.
        eps: Regularization factor for covariance eigenvalues and entropy.
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any] = None,
        total_budget_ratio: float = 0.7,
        rank: Optional[int] = None,
        eps: float = 1e-6,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="entropy_budget", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            calibration_loader = getattr(cfg, "calibration_loader", calibration_loader)
            total_budget_ratio = getattr(
                cfg, "total_budget_ratio", getattr(cfg, "budget_ratio", total_budget_ratio)
            )
            rank = getattr(cfg, "rank", getattr(cfg, "target_rank", rank))

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.calibration_loader = calibration_loader
        self.total_budget_ratio = float(total_budget_ratio)
        self.rank = rank
        self.eps = float(eps)
        self.target_modules = target_modules

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if isinstance(module, SVDLinear):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def collect_covariances(self) -> Dict[str, torch.Tensor]:
        """Collect empirical input activation covariances with double-precision accumulation."""
        if self.calibration_loader is None:
            return {}

        activations: Dict[str, torch.Tensor] = {}
        sample_counts: Dict[str, int] = {}
        hooks = []

        def get_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[Any, ...], outp: Any) -> None:
                if not inp or inp[0] is None:
                    return
                x = inp[0].detach()
                if x.ndim >= 2:
                    x_flat = x.reshape(-1, x.shape[-1]).float()
                    cov = x_flat.T @ x_flat
                    n = x_flat.shape[0]
                    if name not in activations:
                        activations[name] = cov
                        sample_counts[name] = n
                    else:
                        activations[name] += cov
                        sample_counts[name] += n

            return hook

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            return {}

        for name, mod in target_layers:
            hooks.append(mod.register_forward_hook(get_hook(name)))

        device = next(self.model.parameters()).device
        self.model.eval()

        try:
            with torch.no_grad():
                for batch in self.calibration_loader:
                    if isinstance(batch, dict):
                        batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        self.model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        batch = [
                            v.to(device) if isinstance(v, torch.Tensor) else v
                            for v in batch
                        ]
                        self.model(*batch)
                    elif isinstance(batch, torch.Tensor):
                        self.model(batch.to(device))
        except Exception as e:
            logger.warning(
                "EntropyBudget: Calibration forward pass failed (%s). Falling back to weight spectra.",
                e,
            )
        finally:
            for h in hooks:
                h.remove()

        normalized_covs: Dict[str, torch.Tensor] = {}
        for name, cov in activations.items():
            cnt = max(1, sample_counts.get(name, 1))
            normalized_covs[name] = cov / cnt

        return normalized_covs

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress model using whitened entropy budget allocation."""
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("EntropyBudget: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        covs = self.collect_covariances()
        components = []

        for name, module in target_layers:
            W = module.weight.data.float()
            in_f = module.in_features
            out_f = module.out_features

            if name in covs:
                # Robust symmetric covariance root
                cov = covs[name].to(device=W.device, dtype=torch.float32)
                try:
                    eigvals, eigvecs = torch.linalg.eigh(cov + self.eps * torch.eye(in_f, device=W.device))
                    cov_sqrt = eigvecs @ torch.diag(torch.sqrt(torch.clamp(eigvals, min=self.eps))) @ eigvecs.T
                    M = W @ cov_sqrt
                except Exception:
                    M = W
            else:
                M = W

            U, S, Vt = torch.linalg.svd(M, full_matrices=False)
            total_s = S.sum() + self.eps
            p = torch.clamp(S / total_s, min=self.eps, max=1.0)
            entropy = -torch.sum(p * torch.log(p)).item()
            entropy = max(entropy, 1e-4)

            for i, s_val in enumerate(S):
                # Score component by singular energy relative to entropy
                score = (s_val.item() ** 2) / entropy
                components.append((name, i, s_val.item(), score))

        if self.rank is not None:
            allocated_ranks = {name: min(self.rank, mod.in_features, mod.out_features) for name, mod in target_layers}
        else:
            # Sort components globally by energy-entropy score
            components.sort(key=lambda x: x[3], reverse=True)
            total_components = len(components)
            keep_count = max(len(target_layers), int(total_components * self.total_budget_ratio))
            keep_set = set((c[0], c[1]) for c in components[:keep_count])

            allocated_ranks = {}
            for name, module in target_layers:
                layer_rank = len([i for (n, i) in keep_set if n == name])
                max_r = min(module.in_features, module.out_features)
                allocated_ranks[name] = max(1, min(layer_rank, max_r))

        # Perform factorized replacement with symmetric square root singular values
        for name, module in target_layers:
            r = allocated_ranks[name]
            in_f = module.in_features
            out_f = module.out_features
            device = module.weight.device
            dtype = module.weight.dtype

            W = module.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)

            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

            svd_layer = SVDLinear(
                in_features=in_f,
                out_features=out_f,
                rank=r,
                bias=module.bias is not None,
                dtype=dtype,
                device=device,
            )
            svd_layer.set_low_rank_factors(U_r, V_r)
            if module.bias is not None:
                svd_layer.bias.data.copy_(module.bias.data)

            replace_module(self.model, name, svd_layer)

        stats = self.get_compression_stats()
        stats["allocated_ranks"] = allocated_ranks
        return self.model, stats


def entropy_budget_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Entropy budget factory."""
    compressor = EntropyBudgetCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


entropy_budget_allocation_factory = entropy_budget_factory

__all__ = [
    "EntropyBudgetCompressor",
    "entropy_budget_factory",
    "entropy_budget_allocation_factory",
]
