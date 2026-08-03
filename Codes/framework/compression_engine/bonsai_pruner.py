"""Bonsai: Iterative Magnitude Pruning (IMP). Registered as 'bonsai'."""

import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class BonsaiPruner(BaseCompressor):
    def __init__(self, model, sparsity=0.5, steps=10, **kwargs):
        super().__init__(model, name="bonsai", **kwargs)
        self.sparsity = sparsity
        self.steps = steps

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.float()
                final_mask = torch.ones_like(W)
                for s in range(self.steps):
                    # Progressive sparsity
                    sp = self.sparsity * (s + 1) / self.steps
                    flat = W.abs().view(-1)
                    k = int(flat.numel() * (1 - sp))
                    thr = torch.topk(flat, k, largest=True).values[-1]
                    mask = (W.abs() >= thr).float()
                    W = W * mask
                    final_mask = mask  # last mask
                module.weight.data = (W * final_mask).to(dtype=module.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("bonsai")
def bonsai_factory(model, comp_config) -> Any:
    """Bonsai factory."""
    sparsity = getattr(comp_config, "bonsai_sparsity", 0.5)
    steps = getattr(comp_config, "bonsai_steps", 10)
    compressor = BonsaiPruner(model, sparsity=sparsity, steps=steps)
    compressed_model, _ = compressor.compress()
    return compressed_model
