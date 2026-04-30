import torch
import numpy as np

class ProxSparsePruner:
    """
    ProxSparse: regularized learning of semi‑structured sparsity with global feedback.
    Post‑training: we approximate via iterative soft‑thresholding.
    """
    def __init__(self, model, sparsity=0.5, lambda_prox=0.01, steps=5):
        self.model = model
        self.sparsity = sparsity
        self.lambda_prox = lambda_prox
        self.steps = steps

    def prune(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                # Proximal operator for sparsity
                for _ in range(self.steps):
                    threshold = np.percentile(np.abs(W), 100 * (1 - self.sparsity))
                    W_prox = np.where(np.abs(W) > threshold, W - np.sign(W)*threshold, 0)
                    # global feedback: adjust towards original
                    W = W - self.lambda_prox * (W - W_prox)
                module.weight.data = torch.tensor(W_prox, dtype=module.weight.dtype, device=module.weight.device)
        return self.model