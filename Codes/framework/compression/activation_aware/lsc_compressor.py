import torch
import numpy as np

class LSCCompressor:
    def __init__(self, model, rank=32):
        self.model = model
        self.rank = rank

    def compress(self):
        # Use Fourier transform on flattened weight, truncate high‑frequency
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                # 2D FFT
                W_fft = np.fft.fft2(W)
                # Keep only top‑rank coefficients
                magnitude = np.abs(W_fft)
                indices = np.dstack(np.unravel_index(np.argsort(-magnitude.ravel()), magnitude.shape))
                keep = indices[0, :self.rank*self.rank]  # rough
                mask = np.zeros_like(W_fft)
                for idx in keep:
                    mask[idx[0], idx[1]] = 1
                W_comp = np.fft.ifft2(W_fft * mask).real
                module.weight.data = torch.tensor(W_comp, dtype=module.weight.dtype, device=module.weight.device)
        return self.model