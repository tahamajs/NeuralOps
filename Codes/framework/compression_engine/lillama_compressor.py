"""LiLLaMA: local distillation via cross‑covariance. Registered as 'lillama'."""

import torch
import numpy as np
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class LiLLaMACompressor(BaseCompressor):
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None, **kwargs):
        super().__init__(model, name="lillama", **kwargs)
        self.tokenizer = tokenizer
        self.rank = rank
        self.calibration_texts = calibration_texts or ["Distill"]

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
                inp_np = inp.view(-1, inp.size(-1)).float().cpu().numpy()
                out_np = out.view(-1, out.size(-1)).float().cpu().numpy()
                C = inp_np.T @ out_np / len(inp_np)
                U, S, Vt = np.linalg.svd(C, full_matrices=False)
                r = min(self.rank, min(C.shape), len(S))
                W_new = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
                module.weight.data = torch.tensor(W_new, dtype=module.weight.dtype, device=module.weight.device)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()




@COMPRESSOR_REGISTRY.register("lillama")
def lillama_factory(model, comp_config):
    rank = getattr(comp_config, 'lillama_rank', 64)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("lillama requires tokenizer attached as model.tokenizer")
    compressor = LiLLaMACompressor(model, tokenizer, rank)
    compressed_model, _ = compressor.compress()
    return compressed_model

