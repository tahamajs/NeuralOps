"""Mamba-Shedder: sensitivity-based structured pruning. Registered as 'mamba_shedder'."""

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class MambaShedderCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, sparsity=0.2, device="cuda", **kwargs):
        super().__init__(model, name="mamba_shedder", **kwargs)
        self.tokenizer = tokenizer
        self.sparsity = sparsity
        self.device = device

    def _collect_activations(self):
        """Use a small set of generated texts to collect input activations."""
        self.model.eval()
        # Create a simple calibration set from a few sentences
        calib_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subfield of artificial intelligence.",
            "Transformers have revolutionized natural language processing.",
        ]
        inputs = self.tokenizer(
            calib_texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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
            _ = self.model(**inputs)

        for hook in hooks:
            hook.remove()
        return activations

    def _compute_sensitivity(self, activations):
        sensitivities = {}
        for name, act in activations.items():
            act_flat = act.view(-1, act.shape[-1])
            fisher = (act_flat**2).mean(dim=0)
            sensitivities[name] = fisher.sqrt()
        return sensitivities

    def compress(self):
        print(f"✂️ Applying Mamba-Shedder (sparsity={self.sparsity})...")
        activations = self._collect_activations()
        sensitivities = self._compute_sensitivity(activations)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in sensitivities:
                weight = module.weight.data
                sens = sensitivities[name]
                n_keep = int(weight.shape[1] * (1 - self.sparsity))
                keep_idx = torch.argsort(sens, descending=True)[:n_keep]

                # Soft pruning: zero out unimportant columns
                mask = torch.zeros(weight.shape[1], device=weight.device)
                mask[keep_idx] = 1.0
                module.weight.data = weight * mask.unsqueeze(0)  # broadcast over rows
                # Bias remains unchanged

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("mamba_shedder")
def mamba_shedder_factory(model, comp_config):
    sparsity = getattr(comp_config, "mamba_shedder_sparsity", 0.2)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("mamba_shedder requires tokenizer attached as model.tokenizer")
    compressor = MambaShedderCompressor(model, tokenizer, sparsity)
    compressed_model, _ = compressor.compress()
    return compressed_model
