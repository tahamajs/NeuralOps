"""UniQL: Joint low‑rank + quantisation using SVD and uniform quant. Registered as 'uniql'."""
from typing import Any

import torch
from framework.core.base import BaseCompressor


class UniQLCompressor(BaseCompressor):
    def __init__(self, model, rank=64, bits=8, **kwargs):
        super().__init__(model, name="uniql", **kwargs)
        self.rank = rank
        self.bits = bits

    @staticmethod
    def _quantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Uniform min-max quantization for FP16/FP32 stability."""
        if bits >= 32:
            return tensor.float()
        t = tensor.float()
        min_val, max_val = t.min(), t.max()
        if max_val == min_val:
            return t
        qmax = float(max((2**bits) - 1, 1))
        scale = torch.clamp((max_val - min_val) / qmax, min=1e-8)
        quant = torch.clamp(torch.round((t - min_val) / scale), 0.0, qmax) * scale + min_val
        return quant

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = max(1, min(self.rank, min(W.shape), len(S)))
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]
                # Quantize factors
                U_q = self._quantize(U_r, self.bits)
                S_q = self._quantize(S_r, self.bits)
                Vt_q = self._quantize(Vt_r, self.bits)
                W_new = (U_q * S_q.unsqueeze(0)) @ Vt_q
                module.weight.data = W_new.to(dtype=module.weight.dtype, device=module.weight.device)
                if module.bias is not None:
                    b_q = self._quantize(module.bias.data, self.bits)
                    module.bias.data = b_q.to(dtype=module.bias.dtype, device=module.bias.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


from framework.core.registry import COMPRESSOR_REGISTRY


@COMPRESSOR_REGISTRY.register("uniql")
def uniql_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """UniQL factory."""
    rank = kwargs.get("rank", getattr(comp_config, "uniql_rank", 64) if comp_config is not None else 64)
    bits = kwargs.get("bits", getattr(comp_config, "uniql_bits", 8) if comp_config is not None else 8)
    compressor = UniQLCompressor(model, rank=rank, bits=bits, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
