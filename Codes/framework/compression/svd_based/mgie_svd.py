# framework/compression/mgie_svd.py
import torch
import torch.nn as nn

class MGIESVDCompressor:
    """
    MGIE-SVD: Entropy-driven rank selection.
    """
    def compress_layer(self, W):
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        # Gaussian information entropy of singular values
        p = S / S.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-8))
        rank = max(1, int(entropy * len(S) / (torch.log(torch.tensor(len(S))) + 1e-8)))
        return U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]

    def compress(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = self.compress_layer(module.weight.data)
        return model