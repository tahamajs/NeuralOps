import torch
import numpy as np
from .cur_compressor import CURCompressor  # reuse column selection

class CURMoECompressor:
    """
    Apply CUR decomposition to MoE expert layers.
    """
    def __init__(self, model, rank=64):
        self.model = model
        self.rank = rank

    def compress(self):
        for name, module in self.model.named_modules():
            # Identify MoE layers (e.g., modules with `router` or expert lists)
            if hasattr(module, 'experts'):
                for expert in module.experts:
                    cur = CURCompressor(None, self.rank)  # we will manually process weights
                    for param_name, param in expert.named_parameters():
                        if 'weight' in param_name and param.dim() == 2:
                            W = param.data.cpu().numpy()
                            # Use column selection as in CUR
                            col_norms = np.linalg.norm(W, axis=0)
                            col_probs = col_norms / col_norms.sum()
                            selected = np.random.choice(W.shape[1], size=self.rank, replace=False, p=col_probs)
                            C = W[:, selected]
                            R = np.linalg.lstsq(C, W, rcond=None)[0]
                            new_W = C @ R
                            param.data = torch.tensor(new_W, dtype=param.dtype, device=param.device)
        return self.model