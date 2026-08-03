"""UniQL: Joint low‑rank + quantisation using SVD and uniform quant. Registered as 'uniql'."""

import torch
from framework.core.base import BaseCompressor


class UniQLCompressor(BaseCompressor):
    def __init__(self, model, rank=64, bits=8, **kwargs):
        super().__init__(model, name="uniql", **kwargs)
        self.rank = rank
        self.bits = bits

    @staticmethod
    def _quantize(tensor, bits) -> Any:
        """ quantize."""
        if bits >= 32:
            return tensor.float()
        t = tensor.float()
        min_val, max_val = t.min(), t.max()
        if max_val == min_val:
            return t
        scale = (max_val - min_val) / (2**bits - 1)
        quant = torch.round((t - min_val) / scale) * scale + min_val
        return quant

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(self.rank, min(W.shape))
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]
                # Quantize factors
                U_q = self._quantize(U_r, self.bits)
                S_q = self._quantize(S_r, self.bits)
                Vt_q = self._quantize(Vt_r, self.bits)
                W_new = U_q @ torch.diag(S_q) @ Vt_q
                module.weight.data = W_new.to(dtype=module.weight.dtype)
                if module.bias is not None:
                    module.bias.data = self._quantize(module.bias.data, self.bits)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
