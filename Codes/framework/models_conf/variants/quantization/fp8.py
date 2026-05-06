from .base import BaseQuantizer, QUANTIZER_REGISTRY

@QUANTIZER_REGISTRY.register("fp8")
class FP8Quantizer(BaseQuantizer):
    def apply(self, model, **kwargs):
        try:
            import transformer_engine.pytorch as te
            # Wrap model with FP8 – simplified placeholder
            model = te.fp8_autocast(enabled=True)(model)
        except ImportError:
            raise ImportError("Install TransformerEngine for FP8 support")
        return model