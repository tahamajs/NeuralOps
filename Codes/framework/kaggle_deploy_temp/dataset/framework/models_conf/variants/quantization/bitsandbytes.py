import torch
from transformers import BitsAndBytesConfig

from .base import QUANTIZER_REGISTRY, BaseQuantizer


@QUANTIZER_REGISTRY.register("bitsandbytes")
class BitsAndBytesQuantizer(BaseQuantizer):
    def apply(
        self, model, load_in_4bit=True, bnb_4bit_compute_dtype="float16", **kwargs
    ) -> Any:
        """Execute the apply operation and return results."""
        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
        )
        # The model must be reloaded with this config; this instance works on an already loaded model
        # by replacing its config (simplification). In practice, you'd reload.
        if hasattr(model, "config"):
            model.config.quantization_config = quant_config
        return model
