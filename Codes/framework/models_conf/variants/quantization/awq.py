import torch
from .base import BaseQuantizer, QUANTIZER_REGISTRY

@QUANTIZER_REGISTRY.register("awq")
class AWQQuantizer(BaseQuantizer):
    def apply(self, model, bits=4, **kwargs):
        try:
            from transformers import AwqConfig
        except ImportError:
            raise ImportError("Install transformers[awq] for AWQ support")
        quant_config = AwqConfig(bits=bits)
        model.config.quantization_config = quant_config
        return model