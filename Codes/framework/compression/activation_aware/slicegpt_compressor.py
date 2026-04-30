# framework/compression/slicegpt_compressor.py
import torch
import torch.nn as nn
import numpy as np

class SliceGPTCompressor:
    """
    SliceGPT: PCA-based row/column slicing.
    """
    def __init__(self, model, calibration_loader, sparsity=0.3):
        self.model = model
        self.calibration_loader = calibration_loader
        self.sparsity = sparsity

    def compress(self):
        # Collect activations from first layer
        activations = []
        hooks = []
        def hook(module, input, output):
            activations.append(output.detach().cpu().numpy())
        hook_handle = self.model.encoder.layer[0].register_forward_hook(hook)
        with torch.no_grad():
            for batch in self.calibration_loader:
                _ = self.model(**batch)
        hook_handle.remove()
        activations = np.concatenate(activations, axis=0)
        # PCA on activations
        from sklearn.decomposition import PCA
        pca = PCA(n_components=activations.shape[-1])
        pca.fit(activations.reshape(-1, activations.shape[-1]))
        explained_variance = pca.explained_variance_ratio_
        # Keep dimensions that explain (1 - sparsity) of variance
        cumsum = np.cumsum(explained_variance)
        keep = np.where(cumsum >= 1 - self.sparsity)[0][0] + 1
        # Apply slicing to all linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data[:, :keep]
                if module.bias is not None:
                    module.bias.data = module.bias.data[:keep]
        return self.model