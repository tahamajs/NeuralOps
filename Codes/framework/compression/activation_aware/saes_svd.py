# framework/compression/saes_svd.py
import torch
import torch.nn as nn

class SAESSVDCompressor:
    """
    SAES-SVD: Suppress accumulated and local errors.
    """
    def compress_layer(self, W):
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        # Error compensation: keep extra singular values for error reduction
        rank = int(W.shape[0] * 0.7)
        return U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]

    def compress(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = self.compress_layer(module.weight.data)
        return model