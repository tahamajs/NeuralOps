"""COMPOT: orthogonal dictionary learning from calibration. Registered as 'compot'."""

import torch
import numpy as np
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("compot")
class COMPOTCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None, **kwargs):
        super().__init__(model, name="compot", **kwargs)
        self.tokenizer = tokenizer
        self.rank = rank
        self.calibration_texts = calibration_texts or ["Example text for calibration."]

    def compress(self):
        self.model.eval()
        layer_data = {}
        hooks = []
        def make_hook(name):
            def hook(module, input, output):
                layer_data[name] = {'input': input[0].detach(), 'output': output.detach()}
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
            if isinstance(module, torch.nn.Linear) and name in layer_data:
                inp = layer_data[name]['input'].view(-1, module.in_features).float().cpu().numpy()
                out = layer_data[name]['output'].view(-1, module.out_features).float().cpu().numpy()
                Q, _ = np.linalg.qr(inp)
                D = out @ Q
                U, S, Vt = np.linalg.svd(D, full_matrices=False)
                r = min(self.rank, len(S), min(D.shape))
                D_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
                W_new = D_approx @ Q.T
                module.weight.data = torch.tensor(W_new, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()