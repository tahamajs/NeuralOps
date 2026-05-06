import torch.nn.utils.prune as prune
from .base import BaseCompressor, COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("magnitude_pruning")
class MagnitudePruning(BaseCompressor):
    def apply(self, model, amount=0.3, **kwargs):
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                prune.l1_unstructured(module, name='weight', amount=amount)
        return model