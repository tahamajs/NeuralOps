import torch
import numpy as np
from sklearn.decomposition import PCA

class KVTCompressor:
    def __init__(self, model, bits=4, keep_eig=0.95):
        self.model = model
        self.bits = bits
        self.keep_eig = keep_eig

    def compress(self):
        # Only works if the model has a KV cache (conceptual)
        # We'll compress all Linear layers as a proxy
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                # PCA on weight rows
                pca = PCA(n_components=self.keep_eig)
                W_pca = pca.fit_transform(W.T).T
                # Quantize PCA‐transformed weights
                min_v, max_v = W_pca.min(), W_pca.max()
                scale = (max_v - min_v) / (2**self.bits - 1)
                W_q = np.round((W_pca - min_v) / scale) * scale + min_v
                # Inverse transform (approximate)
                W_approx = W_q @ pca.components_  # rough
                module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        return self.model