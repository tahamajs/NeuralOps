"""SLiM: one‑shot quantisation, 2:4 sparsity, low‑rank correction. Registered as 'slim'."""
from typing import Any

import numpy as np
import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class SLiMCompressor(BaseCompressor):
    def __init__(self, model, bits=4, sparsity=0.5, rank=8, **kwargs):
        super().__init__(model, name="slim", **kwargs)
        self.bits = bits
        self.sparsity = sparsity
        self.rank = rank

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float().cpu().numpy()
                min_val, max_val = W.min(), W.max()
                if max_val == min_val:
                    max_val = min_val + 1e-6
                # 1. Quantize
                scale = (max_val - min_val) / (2**self.bits - 1)
                W_q = np.round((W - min_val) / scale) * scale + min_val
                # 2. 2:4 structured sparsity
                W_sp = W_q.copy()
                for i in range(W_q.shape[0]):
                    row = W_q[i]
                    for j in range(0, len(row), 4):
                        sub = row[j : j + 4]
                        idx = np.argsort(np.abs(sub))[-2:]
                        mask = np.zeros(4)
                        mask[idx] = 1
                        W_sp[i, j : j + 4] = sub * mask
                # 3. Low‑rank correction
                residual = W - W_sp
                U, S, Vt = np.linalg.svd(residual, full_matrices=False)
                r = min(self.rank, len(S))
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]
                correction = U_r @ np.diag(S_r) @ Vt_r
                W_final = W_sp + correction
                module.weight.data = torch.tensor(
                    W_final, dtype=module.weight.dtype, device=module.weight.device
                )
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("slim")
def slim_factory(model, comp_config) -> Any:
    """Slim factory."""
    bits = getattr(comp_config, "slim_bits", 4)
    sparsity = getattr(comp_config, "slim_sparsity", 0.5)
    rank = getattr(comp_config, "slim_rank", 8)
    compressor = SLiMCompressor(model, bits=bits, sparsity=sparsity, rank=rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
