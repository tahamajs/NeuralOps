"""SliceGPT PCA‑based slicing compressor. Registered as 'slicegpt_pca'."""

import torch
import torch.nn as nn
import numpy as np
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class SliceGPTPCACompressor(BaseCompressor):
    def __init__(self, model, calibration_loader, sparsity=0.3, **kwargs):
        super().__init__(model, name="slicegpt_pca", **kwargs)
        self.calibration_loader = calibration_loader
        self.sparsity = sparsity

    def compress(self):
        activations = []
        def hook(module, input, output):
            activations.append(output.detach().cpu().numpy())
        handle = None
        # Attach to first layer of encoder if exists
        for name, module in self.model.named_modules():
            if 'encoder.layer.0' in name:
                handle = module.register_forward_hook(hook)
                break
        if handle is None:
            # fallback: attach to first Linear
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    handle = module.register_forward_hook(hook)
                    break
        if handle is None:
            raise RuntimeError("Could not attach hook")
        with torch.no_grad():
            for batch in self.calibration_loader:
                self.model(**{k: v.to(next(self.model.parameters()).device) for k, v in batch.items()})
        handle.remove()
        activations = np.concatenate(activations, axis=0).reshape(-1, activations[0].shape[-1])
        from sklearn.decomposition import PCA
        pca = PCA(n_components=activations.shape[-1])
        pca.fit(activations)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        keep = np.searchsorted(cumsum, 1 - self.sparsity) + 1
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data[:, :keep]
                if module.bias is not None:
                    module.bias.data = module.bias.data[:keep]
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    



    