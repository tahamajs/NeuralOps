"""AdaSVD: Adaptive SVD with per‑layer importance and iterative refinement."""
from typing import Any

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class AdaSVDCompressor(BaseCompressor):
    """
    Compress a model by assigning SVD ranks proportional to each layer's
    importance (nuclear norm) and then iteratively refining U and V.

    Args:
        model: PyTorch model.
        target_param_ratio: fraction of parameters to **remove** (0-1).
        num_iters: number of alternating refinement steps.
    """

    def __init__(self, model, target_param_ratio=0.3, num_iters=5, **kwargs):
        super().__init__(model, name="adasvd", **kwargs)
        self.target_param_ratio = target_param_ratio
        self.num_iters = num_iters

    @staticmethod
    def _weight_importance(W) -> Any:
        """ weight importance."""
        _, S, _ = np.linalg.svd(W, full_matrices=False)
        return np.sum(S)

    def compress(self):
        # 1. Compute importance per layer
        """Compress the target module using this strategy."""
        layer_importances = {}
        total_importance = 0.0
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                imp = self._weight_importance(module.weight.data.float().cpu().numpy())
                layer_importances[name] = imp
                total_importance += imp

        if total_importance == 0.0:
            self._compressed_params = self._count_params(self.model)
            return self.model, self.get_compression_stats()

        # 2. Compute total original parameters
        sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # 3. Assign ranks proportionally, keeping target_params after compression
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in layer_importances:
                frac = layer_importances[name] / total_importance
                max_rank = min(module.weight.shape[0], module.weight.shape[1])
                # Determine rank so that total saved params ≈ target_param_ratio * total_params
                # Simple heuristic: rank = max_rank * (1 - target_param_ratio) * frac
                rank = max(1, int(max_rank * (1.0 - self.target_param_ratio) * frac))
                rank = min(rank, max_rank)

                W = module.weight.data.float().cpu().numpy()
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                
                # Initialize A and B for ALS: W ≈ A @ B
                sqrt_S = np.sqrt(S[:rank])
                A = U[:, :rank] * sqrt_S[np.newaxis, :]
                B = sqrt_S[:, np.newaxis] * Vt[:rank, :]

                # Alternating Least Squares (ALS) refinement
                for _ in range(self.num_iters):
                    # Update A: minimize ||W - A @ B||_F^2
                    # A = W @ B^T @ (B @ B^T + eps * I)^-1
                    B_Bt = B @ B.T
                    A = W @ B.T @ np.linalg.inv(B_Bt + 1e-6 * np.eye(rank))
                    
                    # Update B: minimize ||W - A @ B||_F^2
                    # B = (A^T @ A + eps * I)^-1 @ A^T @ W
                    At_A = A.T @ A
                    B = np.linalg.inv(At_A + 1e-6 * np.eye(rank)) @ A.T @ W
                
                # After refinement, final approximation
                W_approx = A @ B
                
                module.weight.data = torch.tensor(
                    W_approx, dtype=module.weight.dtype, device=module.weight.device
                )

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("adasvd")
def adasvd_factory(model, comp_config) -> Any:
    """Adasvd factory."""
    target_param_ratio = getattr(comp_config, "adasvd_target_ratio", 0.3)
    num_iters = getattr(comp_config, "adasvd_iters", 5)
    compressor = AdaSVDCompressor(
        model, target_param_ratio=target_param_ratio, num_iters=num_iters
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
