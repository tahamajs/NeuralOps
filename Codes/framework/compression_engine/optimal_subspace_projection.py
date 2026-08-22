"""Activation‑Aware Subspace Projection via randomized SVD. Registered as 'subspace_projection'."""
from typing import Any

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class SubspaceProjectionCompressor(BaseCompressor):
    def __init__(
        self, model, calibration_loader, rank_ratio=0.5, power_iters=2, **kwargs
    ):
        super().__init__(model, name="subspace_projection", **kwargs)
        self.calibration_loader = calibration_loader
        self.rank_ratio = rank_ratio
        self.power_iters = power_iters

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        act_covs = {}

        def hook_fn(module, input, output) -> None:
            """Hook fn."""
            x = input[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).float()
            cov = x_flat.T @ x_flat / x_flat.shape[0]
            if id(module) not in act_covs:
                act_covs[id(module)] = cov
            else:
                act_covs[id(module)] += cov

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and not any(
                skip in name for skip in ["embeddings", "classifier"]
            ):
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            for batch in self.calibration_loader:
                batch = {
                    k: v.to(next(self.model.parameters()).device)
                    for k, v in batch.items()
                }
                self.model(**batch)

        for h in hooks:
            h.remove()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and not any(
                skip in name for skip in ["embeddings", "classifier"]
            ):
                W = module.weight.data.float()
                C = act_covs[id(module)]
                L = torch.linalg.cholesky(C)
                M = W @ L
                rank = max(1, int(min(W.shape) * self.rank_ratio))
                Omega = torch.randn(M.shape[1], rank, device=W.device)
                Y = M @ Omega
                for _ in range(self.power_iters):
                    Y = M @ (M.T @ Y)
                Q, _ = torch.linalg.qr(Y)
                B = Q.T @ M
                U, S, Vt = torch.linalg.svd(B, full_matrices=False)
                r = min(rank, len(S))
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]
                new_U = Q @ U_r
                new_V = (
                    Vt_r
                    @ torch.linalg.solve(L.T, torch.eye(L.shape[0], device=W.device)).T
                )
                new_weight = new_U @ torch.diag(S_r) @ new_V
                module.weight.data = new_weight.to(dtype=module.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("subspace_projection")
def subspace_projection_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Subspace projection factory."""
    calibration_loader = getattr(comp_config, "calibration_loader", kwargs.get("calibration_loader", None))
    if calibration_loader is None:
        # Dummy calibration loader for standalone tests if not provided
        dummy_input = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10)
        }
        calibration_loader = [dummy_input]
    rank_ratio = getattr(comp_config, "rank_ratio", kwargs.get("rank_ratio", 0.5))
    power_iters = getattr(comp_config, "power_iters", kwargs.get("power_iters", 2))
    compressor = SubspaceProjectionCompressor(
        model,
        calibration_loader=calibration_loader,
        rank_ratio=rank_ratio,
        power_iters=power_iters
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
