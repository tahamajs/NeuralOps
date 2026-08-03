from typing import Any
from .base import QUANTIZER_REGISTRY, BaseQuantizer


@QUANTIZER_REGISTRY.register("torchao")
class TorchAOQuantizer(BaseQuantizer):
    def apply(self, model, method="int4_weight_only", **kwargs) -> Any:
        """Execute the apply operation and return results."""
        try:
            from torchao.quantization import quantize_

            quantize_(model, method)
        except ImportError:
            raise ImportError("Install torchao for TorchAO quantization")
        return model
