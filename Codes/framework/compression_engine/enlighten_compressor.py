"""ENLighten: low‑rank SVD + 2:4 structured sparsity (photonic‑aligned). Registered as 'enlighten'."""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class ENLightenCompressor(BaseCompressor):
    """
    ENLighten: Low-rank SVD + 2:4 structured sparsity (photonic-aligned).
    Decomposes linear layers into low-rank representations plus 2:4 structured sparse residuals.
    """

    def __init__(self, model: torch.nn.Module, rank: int = 64, sparsity: float = 0.5, **kwargs: Any) -> None:
        super().__init__(model, name="enlighten", **kwargs)
        self.rank = int(rank)
        self.sparsity = float(sparsity)

    def compress(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Compress linear layers using low-rank decomposition + 2:4 structured sparse residuals."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = max(1, min(self.rank, len(S), min(W.shape)))
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]
                L = (U_r * S_r.unsqueeze(0)) @ Vt_r
                residual = W - L

                # 2:4 structured sparsity along rows (groups of 4)
                out_f, in_f = residual.shape
                residual_sp = torch.zeros_like(residual)
                for j in range(0, in_f, 4):
                    end_j = min(j + 4, in_f)
                    sub = residual[:, j:end_j]
                    chunk_len = end_j - j
                    k_keep = min(2, chunk_len)
                    if k_keep > 0:
                        _, top_idx = torch.topk(sub.abs(), k_keep, dim=1)
                        mask = torch.zeros_like(sub)
                        mask.scatter_(1, top_idx, 1.0)
                        residual_sp[:, j:end_j] = sub * mask

                W_new = L + residual_sp
                module.weight.data = W_new.to(
                    dtype=module.weight.dtype, device=module.weight.device
                )

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["rank"] = self.rank
        stats["sparsity"] = self.sparsity
        return self.model, stats


@COMPRESSOR_REGISTRY.register("enlighten")
def enlighten_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Enlighten compressor factory function."""
    rank = kwargs.get("rank", getattr(comp_config, "enlighten_rank", 64) if comp_config is not None else 64)
    sparsity = kwargs.get("sparsity", getattr(comp_config, "enlighten_sparsity", 0.5) if comp_config is not None else 0.5)
    compressor = ENLightenCompressor(model, rank=rank, sparsity=sparsity, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
