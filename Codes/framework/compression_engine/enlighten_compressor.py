"""ENLighten: low‑rank SVD + 2:4 structured sparsity (photonic‑aligned). Registered as 'enlighten'."""
from typing import Any

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class ENLightenCompressor(BaseCompressor):
    def __init__(self, model, rank=64, sparsity=0.5, **kwargs):
        super().__init__(model, name="enlighten", **kwargs)
        self.rank = rank
        self.sparsity = sparsity

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                r = min(self.rank, len(S))
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]
                L = U_r @ np.diag(S_r) @ Vt_r
                residual = W - L
                # 2:4 sparsity on residual
                residual_sp = np.zeros_like(residual)
                for i in range(residual.shape[0]):
                    for j in range(0, residual.shape[1], 4):
                        sub = residual[i, j : j + 4]
                        idx = np.argsort(np.abs(sub))[-2:]
                        mask = np.zeros(4)
                        mask[idx] = 1
                        residual_sp[i, j : j + 4] = sub * mask
                W_new = L + residual_sp
                module.weight.data = torch.tensor(
                    W_new, dtype=module.weight.dtype, device=module.weight.device
                )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("enlighten")
def enlighten_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Enlighten factory."""
    rank = getattr(comp_config, "enlighten_rank", kwargs.get("rank", 64))
    sparsity = getattr(comp_config, "enlighten_sparsity", kwargs.get("sparsity", 0.5))
    compressor = ENLightenCompressor(model, rank=rank, sparsity=sparsity, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
