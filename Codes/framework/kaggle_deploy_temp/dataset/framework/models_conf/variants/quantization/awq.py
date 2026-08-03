from .base import QUANTIZER_REGISTRY, BaseQuantizer


@QUANTIZER_REGISTRY.register("awq")
class AWQQuantizer(BaseQuantizer):
    def apply(self, model, bits=4, **kwargs) -> Any:
        """Execute the apply operation and return results."""
        try:
            from transformers import AwqConfig
        except ImportError:
            raise ImportError("Install transformers[awq] for AWQ support")
        quant_config = AwqConfig(bits=bits)
        model.config.quantization_config = quant_config
        return model
