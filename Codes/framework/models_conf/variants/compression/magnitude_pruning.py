from typing import Any
import torch.nn.utils.prune as prune

from .base import COMPRESSOR_REGISTRY, BaseCompressor


@COMPRESSOR_REGISTRY.register("magnitude_pruning")
class MagnitudePruning(BaseCompressor):
    def apply(self, model, amount=0.3, **kwargs) -> Any:
        """Execute the apply operation and return results."""
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                prune.l1_unstructured(module, name="weight", amount=amount)
        return model
