from .base import BaseQuantizer, QUANTIZER_REGISTRY

@QUANTIZER_REGISTRY.register("torchao")
class TorchAOQuantizer(BaseQuantizer):
    def apply(self, model, method="int4_weight_only", **kwargs):
        try:
            from torchao.quantization import quantize_
            quantize_(model, method)
        except ImportError:
            raise ImportError("Install torchao for TorchAO quantization")
        return model