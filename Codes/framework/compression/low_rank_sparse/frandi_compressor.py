import torch
import numpy as np

class FRanDICompressor:
    def __init__(self, model):
        self.model = model

    def compress(self):
        # Regress features from original model to generate synthetic calibration data
        # For simplicity, use random noise and optimize to match activations
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data
                # Use random input to compute output
                x = torch.randn(1, 10, W.shape[1], device=W.device)  # batch=1, seq=10
                with torch.no_grad():
                    out = module(x)
                # Fit a low‑rank approximation to minimize output error
                # Solve: min_{W', rank r} ||W'x - out||
                # For brevity, just do SVD of W
                U, S, V = torch.svd(W.float())
                rank = 8
                W_approx = U[:,:rank] @ torch.diag(S[:rank]) @ V[:,:rank].T
                module.weight.data = W_approx.to(dtype=W.dtype)
        return self.model