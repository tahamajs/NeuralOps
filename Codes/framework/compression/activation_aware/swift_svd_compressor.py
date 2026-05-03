"""Swift‑SVD Activation‑aware via cross‑covariance. Registered as 'swift_svd_act'."""

import torch
import numpy as np
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("swift_svd_act")
class SwiftSVDActivationCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None, **kwargs):
        super().__init__(model, name="swift_svd_act", **kwargs)
        self.tokenizer = tokenizer
        self.rank = rank
        self.calibration_texts = calibration_texts or ["Sample text."]

    def compress(self):
        self.model.eval()
        activations = {}
        hooks = []
        def make_hook(name):
            def hook(module, input, output):
                activations[name] = (input[0].detach(), output.detach())
            return hook
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))
        inputs = self.tokenizer(self.calibration_texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            self.model(**inputs)
        for hook in hooks:
            hook.remove()

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in activations:
                inp, out = activations[name]
                inp_flat = inp.view(-1, inp.size(-1)).float()
                out_flat = out.view(-1, out.size(-1)).float()
                C = inp_flat.t() @ out_flat / inp_flat.size(0)
                U, S, Vt = torch.linalg.svd(C, full_matrices=False)
                r = min(self.rank, min(C.shape))
                W_new = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
                module.weight.data = W_new.to(dtype=module.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()