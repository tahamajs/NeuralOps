"""Mamba-Shedder: sensitivity-based structured pruning. Registered as 'mamba_shedder'."""

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@COMPRESSOR_REGISTRY.register("mamba_shedder")
class MambaShedderCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, sparsity=0.2, device="cuda", **kwargs):
        super().__init__(model, name="mamba_shedder", **kwargs)
        self.tokenizer = tokenizer
        self.sparsity = sparsity
        self.device = device

    def _collect_activations(self, calib_loader):
        self.model.eval()
        activations = {}
        hooks = []

        def hook_fn(module, input, output, name):
            if isinstance(module, nn.Linear):
                activations[name] = input[0].detach().clone()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                hooks.append(hook)

        with torch.no_grad():
            for batch in calib_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                _ = self.model(**batch)

        for hook in hooks:
            hook.remove()
        return activations

    def _compute_sensitivity(self, activations):
        sensitivities = {}
        for name, act in activations.items():
            act_flat = act.view(-1, act.shape[-1])
            fisher = (act_flat ** 2).mean(dim=0)
            sensitivities[name] = fisher.sqrt()
        return sensitivities

    def compress(self, calib_loader):
        print(f"✂️ Applying Mamba-Shedder (sparsity={self.sparsity})...")
        activations = self._collect_activations(calib_loader)
        sensitivities = self._compute_sensitivity(activations)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in sensitivities:
                weight = module.weight.data
                sens = sensitivities[name]
                n_keep = int(weight.shape[1] * (1 - self.sparsity))
                keep_idx = torch.argsort(sens, descending=True)[:n_keep]
                module.weight.data = weight[:, keep_idx]
                if module.bias is not None:
                    module.bias.data = module.bias.data[keep_idx]
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()