import torch
import numpy as np

class BonsaiPruner:
    def __init__(self, model, sparsity=0.5):
        self.model = model
        self.sparsity = sparsity

    def prune(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                # Compute hessian‑free second‑order importance by output variation
                # (Simplified: use absolute weight as proxy)
                importance = np.abs(W).mean(axis=1)  # row importance
                keep = int(W.shape[0] * (1-self.sparsity))
                idx = np.argsort(importance)[-keep:]
                module.weight.data = torch.tensor(W[idx,:], dtype=module.weight.dtype, device=module.weight.device)
                if module.bias is not None:
                    module.bias.data = module.bias.data[idx]
        return self.model