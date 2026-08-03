from .base import QUANTIZER_REGISTRY, BaseQuantizer


@QUANTIZER_REGISTRY.register("gptq")
class GPTQQuantizer(BaseQuantizer):
    def apply(self, model, bits=4, dataset="c4", **kwargs) -> Any:
        """Execute the apply operation and return results."""
        try:
            from transformers import GPTQConfig
        except ImportError:
            raise ImportError("Install transformers[gptq] for GPTQ support")
        quant_config = GPTQConfig(bits=bits, dataset=dataset)
        model.config.quantization_config = quant_config
        return model
