import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes

class COMPOTCompressor:
    """
    COMPOT: Use calibration data to learn a sparse factorization with orthogonal dictionaries.
    Simplified: decompose weight as W = D @ A, where D is learned dictionary (orthogonal).
    """
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None):
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        self.calibration_texts = calibration_texts or ["Example text for calibration."]

    def compress(self):
        self.model.eval()
        # 1. Collect inputs and outputs of each linear layer
        layer_data = {}
        hooks = []
        def make_hook(name):
            def hook(module, input, output):
                layer_data[name] = {'input': input[0].detach(), 'output': output.detach()}
            return hook
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))
        inputs = self.tokenizer(self.calibration_texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            self.model(**inputs)
        for hook in hooks:
            hook.remove()

        for name, module in self.model.named_modules():
            if name in layer_data:
                inp = layer_data[name]['input']  # (batch*seq, in_features)
                out = layer_data[name]['output']  # (batch*seq, out_features)
                # Compute orthonormal basis for input via QR
                Q, _ = np.linalg.qr(inp.cpu().numpy())
                # Solve for dictionary D: out = D @ Q^T, so D = out @ Q
                D = out.cpu().numpy() @ Q
                # Truncate to rank (keep top components)
                U, S, Vt = np.linalg.svd(D, full_matrices=False)
                D_approx = U[:, :self.rank] @ np.diag(S[:self.rank]) @ Vt[:self.rank, :]
                # New weight: W_new = D_approx @ Q.T (but Q is orthonormal, so Q.T = Q^-1)
                W_new = D_approx @ Q.T
                module.weight.data = torch.tensor(W_new, dtype=module.weight.dtype, device=module.weight.device)
        return self.model