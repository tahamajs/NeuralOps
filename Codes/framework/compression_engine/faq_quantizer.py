"""FAQ (Future‑Aware Quantization): adjusts quantisation scale using next layer activations. Registered as 'faq'."""

import torch
import numpy as np
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class FAQQuantizer(BaseCompressor):
    def __init__(self, model, tokenizer, bits=8, calibration_texts=None, **kwargs):
        super().__init__(model, name="faq", **kwargs)
        self.tokenizer = tokenizer
        self.bits = bits
        self.calibration_texts = calibration_texts or ["The quick brown fox jumps over the lazy dog"]

    def compress(self):
        self.model.eval()
        activations = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        inputs = self.tokenizer(self.calibration_texts, return_tensors="pt", padding=True, truncation=True)
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in activations:
                act = activations[name].view(-1).float()
                min_a, max_a = act.min(), act.max()
                if max_a == min_a:
                    continue
                scale = (max_a - min_a) / (2**self.bits - 1)
                W = module.weight.data.float()
                quant = torch.round((W - min_a) / scale) * scale + min_a
                module.weight.data = quant.to(dtype=module.weight.dtype)
                if module.bias is not None:
                    module.bias.data = torch.round((module.bias.data - min_a) / scale) * scale + min_a
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    


@COMPRESSOR_REGISTRY.register("faq")
def faq_factory(model, comp_config):
    bits = getattr(comp_config, 'faq_bits', 8)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("faq requires tokenizer attached as model.tokenizer")
    compressor = FAQQuantizer(model, tokenizer, bits)
    compressed_model, _ = compressor.compress()
    return compressed_model
