import torch

from .base import COMPRESSOR_REGISTRY, BaseCompressor


@COMPRESSOR_REGISTRY.register("low_rank")
class LowRankCompressor(BaseCompressor):
    def apply(self, model, rank=64, **kwargs):
        # Replace Linear layers with low‑rank SVD approximations (simplified)
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                U, S, V = torch.svd(module.weight.data)
                U = U[:, :rank] @ torch.diag(S[:rank])
                V = V[:, :rank].t()
                new_linear = torch.nn.Linear(V.shape[1], U.shape[0])
                new_linear.weight.data = U @ V
                setattr(model, name, new_linear)
        return model
