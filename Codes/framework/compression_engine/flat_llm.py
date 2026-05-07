"""FLAT-LLM: Activation‑aware dimension pruning via PCA. Registered as 'flat_llm'."""

import torch
import torch.nn as nn
import numpy as np
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


@COMPRESSOR_REGISTRY.register("flat_llm")
class FLATLLMCompressor(BaseCompressor):
    """
    Prunes output dimensions of linear layers based on activation variance.
    
    Uses PCA on the output activations collected from a forward pass with
    calibration data, then retains only the dimensions with the highest
    variance.

    Args:
        model: PyTorch model.
        tokenizer: HuggingFace tokenizer for calibration texts.
        keep_ratio: fraction of output dimensions to keep (default 0.7).
        calibration_texts: list of input strings for calibration.
    """
    def __init__(self, model, tokenizer, keep_ratio=0.7, calibration_texts=None, **kwargs):
        super().__init__(model, name="flat_llm", **kwargs)
        self.tokenizer = tokenizer
        self.keep_ratio = keep_ratio
        self.calibration_texts = calibration_texts or [
            "The quick brown fox jumps over the lazy dog."
        ]

    def compress(self):
        self.model.eval()
        activations = {}
        hooks = []

        def hook_fn(module, input, output, name):
            activations[name] = output.detach().cpu()

        # Register hooks on all Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o, n=name: hook_fn(m, i, o, n)
                    )
                )

        # Forward pass with calibration data
        inputs = self.tokenizer(
            self.calibration_texts, return_tensors="pt", padding=True, truncation=True
        )
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        # For each linear layer, prune output dimensions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in activations:
                act = activations[name]
                # Flatten to 2D (batch*seq, out_features)
                act_flat = act.view(-1, act.size(-1)).float().numpy()
                # Covariance matrix of activations
                cov = np.cov(act_flat.T)
                eigenvalues, _ = np.linalg.eigh(cov)
                total_dims = len(eigenvalues)
                keep_dims = max(1, int(total_dims * self.keep_ratio))
                # Keep dimensions with largest eigenvalues
                idx = np.argsort(eigenvalues)[::-1][:keep_dims]
                # Prune weight and bias
                W = module.weight.data
                module.weight.data = W[idx, :]
                if module.bias is not None:
                    module.bias.data = module.bias.data[idx]

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()