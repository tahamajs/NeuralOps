"""FLAT-LLM: Activation‑aware dimension pruning via PCA. Registered as 'flat_llm'."""

import numpy as np
import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class FLATLLMCompressor(BaseCompressor):
    def __init__(
        self, model, tokenizer, keep_ratio=0.7, calibration_texts=None, **kwargs
    ):
        super().__init__(model, name="flat_llm", **kwargs)
        self.tokenizer = tokenizer
        self.keep_ratio = keep_ratio
        self.calibration_texts = calibration_texts or [
            "The quick brown fox jumps over the lazy dog."
        ]

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        self.model.eval()
        activations = {}
        hooks = []

        def hook_fn(module, input, output, name) -> None:
            """Hook fn."""
            activations[name] = output.detach().cpu()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o, n=name: hook_fn(m, i, o, n)
                    )
                )

        inputs = self.tokenizer(
            self.calibration_texts, return_tensors="pt", padding=True, truncation=True
        )
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in activations:
                act = activations[name]
                act_flat = act.view(-1, act.size(-1)).float().numpy()
                cov = np.cov(act_flat.T)
                eigenvalues, _ = np.linalg.eigh(cov)
                total_dims = len(eigenvalues)
                keep_dims = max(1, int(total_dims * self.keep_ratio))
                # Indices of dimensions to keep (largest eigenvalues)
                idx = np.argsort(eigenvalues)[::-1][:keep_dims]

                # Soft pruning: zero out unimportant rows (output neurons)
                mask = torch.zeros(total_dims, device=module.weight.device)
                mask[idx] = 1.0
                module.weight.data = module.weight.data * mask.unsqueeze(
                    1
                )  # broadcast over input dim
                if module.bias is not None:
                    module.bias.data = (
                        module.bias.data * mask
                    )  # keep only the biases of kept neurons

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("flat_llm")
def flat_llm_factory(model, comp_config) -> Any:
    """Flat llm factory."""
    keep_ratio = getattr(comp_config, "flat_llm_keep_ratio", 0.7)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("flat_llm requires tokenizer attached as model.tokenizer")
    compressor = FLATLLMCompressor(model, tokenizer, keep_ratio=keep_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model
