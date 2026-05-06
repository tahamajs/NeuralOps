import torch
from transformers import BitsAndBytesConfig
from .base import BaseQuantizer, QUANTIZER_REGISTRY

@QUANTIZER_REGISTRY.register("bitsandbytes")
class BitsAndBytesQuantizer(BaseQuantizer):
    def apply(self, model, load_in_4bit=True, bnb_4bit_compute_dtype="float16", **kwargs):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype))
        # The model must be reloaded with this config; this instance works on an already loaded model
        # by replacing its config (simplification). In practice, you'd reload.
        if hasattr(model, 'config'):
            model.config.quantization_config = quant_config
        return model