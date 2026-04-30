import torch
import numpy as np

class FAQQuantizer:
    """
    Future‑Aware Quantization: uses next layer's activations to adjust quantization scale.
    """
    def __init__(self, model, tokenizer, bits=8, calibration_texts=None):
        self.model = model
        self.tokenizer = tokenizer
        self.bits = bits
        self.calibration_texts = calibration_texts or ["Quant"]

    def compress(self):
        self.model.eval()
        # Capture activations layer by layer
        activations = {}
        hooks = []
        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))
        inputs = self.tokenizer(self.calibration_texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            self.model(**inputs)
        for hook in hooks:
            hook.remove()

        # Use next layer's activation range to shift quantization of current layer (simplified)
        for name, module in self.model.named_modules():
            if name in activations:
                act_next = activations[name].view(-1).cpu().numpy()
                min_val, max_val = np.min(act_next), np.max(act_next)
                scale = (max_val - min_val) / (2**self.bits - 1)
                # Quantize weights of previous layer (here same layer for simplicity)
                W = module.weight.data
                q_W = torch.round((W - min_val) / scale) * scale + min_val
                module.weight.data = q_W.to(dtype=module.weight.dtype)
        return self.model