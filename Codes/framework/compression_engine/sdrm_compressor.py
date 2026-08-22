"""
SDRM: rank assignment proportional to parameter budget for edge inference.
Registered as 'sdrm'.
"""
from typing import Any

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class SDRMCompressor(BaseCompressor):
    """
    SDRM: Scaled Dimension Rank Modulation.
    Assigns layer rank proportional to parameter budget for edge inference.
    """

    def __init__(self, model: torch.nn.Module, total_param_ratio: float = 0.7, **kwargs: Any) -> None:
        super().__init__(model, name="sdrm", **kwargs)
        self.total_param_ratio = float(total_param_ratio)

    def compress(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Compress linear layers using budget-proportional rank assignment."""
        layer_params = {}
        total_params = 0
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                n, m = module.weight.shape
                params = n * m
                layer_params[name] = (module, params, n, m)
                total_params += params

        for name, (module, p, n, m) in layer_params.items():
            allowed = p * self.total_param_ratio
            rank = max(1, int(allowed / max(n + m, 1)))
            rank = min(rank, min(n, m))
            W = module.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = max(1, min(rank, len(S)))
            W_approx = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
            module.weight.data = W_approx.to(
                dtype=module.weight.dtype, device=module.weight.device
            )
        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["total_param_ratio"] = self.total_param_ratio
        return self.model, stats


@COMPRESSOR_REGISTRY.register("sdrm")
def sdrm_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """SDRM compressor factory."""
    total_param_ratio = kwargs.pop(
        "total_param_ratio",
        kwargs.pop("ratio", getattr(comp_config, "total_param_ratio", 0.7) if comp_config is not None else 0.7),
    )
    compressor = SDRMCompressor(model, total_param_ratio=total_param_ratio, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
