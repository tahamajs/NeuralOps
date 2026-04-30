import torch
import numpy as np

class ASTPruner:
    """
    AST: retraining framework for semi‑structured sparse LLMs.
    For post‑training, we just apply 2:4 pattern.
    """
    def __init__(self, model, block_size=4):
        self.model = model
        self.block_size = block_size

    def prune(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                N, M = W.shape
                W_ast = np.zeros_like(W)
                for i in range(N):
                    for j in range(0, M, self.block_size):
                        sub = W[i, j:j+self.block_size]
                        idx = np.argsort(np.abs(sub))[-2:]  # keep top 2
                        mask = np.zeros(self.block_size)
                        mask[idx] = 1
                        W_ast[i, j:j+self.block_size] = sub * mask
                module.weight.data = torch.tensor(W_ast, dtype=module.weight.dtype, device=module.weight.device)
        return self.model