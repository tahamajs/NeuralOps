import torch
import numpy as np

class UniQLCompressor:
    """
    Joint low‑rank + quantization: apply SVD then quantize the factors.
    """
    def __init__(self, model, rank=64, bits=8):
        self.model = model
        self.rank = rank
        self.bits = bits

    def _quantize(self, tensor):
        # Simple uniform quantization
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / (2**self.bits - 1)
        quant = torch.round((tensor - min_val) / scale) * scale + min_val
        return quant

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                U_r = U[:, :self.rank]
                S_r = S[:self.rank]
                Vt_r = Vt[:self.rank, :]
                # Quantize the factors
                U_q = self._quantize(torch.tensor(U_r))
                S_q = self._quantize(torch.tensor(S_r))
                Vt_q = self._quantize(torch.tensor(Vt_r))
                W_q = U_q @ torch.diag(S_q) @ Vt_q
                module.weight.data = W_q.to(dtype=module.weight.dtype, device=module.weight.device)
        return self.model