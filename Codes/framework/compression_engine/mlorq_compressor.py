"""MLoRQ: Multi‑Layer optimal rank & quantisation under a total memory budget. Registered as 'mlorq'."""
from typing import Any

import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class MLoRQCompressor(BaseCompressor):
    def __init__(
        self, model, total_memory_budget_mb=500, bits_candidates=[2, 4, 8], **kwargs
    ):
        super().__init__(model, name="mlorq", **kwargs)
        self.total_budget = total_memory_budget_mb * 8 * 1024 * 1024  # bits
        self.bits_candidates = bits_candidates

    @staticmethod
    def _param_size(shape, bits) -> Any:
        """ param size."""
        return shape[0] * shape[1] * bits

    @staticmethod
    def _quantize(tensor, bits) -> Any:
        """ quantize."""
        if bits >= 32:
            return tensor.float()
        t = tensor.float()
        min_v, max_v = t.min(), t.max()
        if max_v == min_v:
            return t
        scale = (max_v - min_v) / (2**bits - 1)
        return torch.round((t - min_v) / scale) * scale + min_v

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data
                out_f, in_f = W.shape
                layers.append(
                    {"name": name, "module": module, "shape": (out_f, in_f), "W": W}
                )

        total_cost = 0
        for info in layers:
            best = None
            max_rank = min(info["shape"])
            for bits in sorted(self.bits_candidates, reverse=True):
                for rank in range(max_rank, 0, -1):
                    mem = (info["shape"][0] * rank + rank * info["shape"][1]) * bits
                    if total_cost + mem <= self.total_budget:
                        best = (rank, bits, mem)
                        break
                if best is not None:
                    break
            if best is None:
                best = (1, self.bits_candidates[-1], 0)
            r, b, mem = best
            total_cost += mem
            # Compress layer
            W_np = info["W"].cpu().float()
            U, S, Vt = torch.linalg.svd(W_np, full_matrices=False)
            U_r = U[:, :r]
            S_r = S[:r]
            Vt_r = Vt[:r, :]
            U_q = self._quantize(U_r, b).to(info["W"].device)
            S_q = self._quantize(S_r, b).to(info["W"].device)
            Vt_q = self._quantize(Vt_r, b).to(info["W"].device)
            W_new = U_q @ torch.diag(S_q) @ Vt_q
            info["module"].weight.data = W_new.to(dtype=info["W"].dtype)
            if info["module"].bias is not None:
                info["module"].bias.data = self._quantize(info["module"].bias.data, b)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("mlorq")
def mlorq_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Mlorq factory."""
    total_budget_mb = getattr(comp_config, "mlorq_budget_mb", kwargs.get("total_memory_budget_mb", 500))
    bits_candidates = getattr(comp_config, "mlorq_bits_candidates", kwargs.get("bits_candidates", [2, 4, 8]))
    compressor = MLoRQCompressor(
        model, total_memory_budget_mb=total_budget_mb, bits_candidates=bits_candidates, **kwargs
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
