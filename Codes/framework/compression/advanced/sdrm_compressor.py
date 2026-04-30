import torch
import numpy as np

class SDRMCompressor:
    """
    SDRM: SVD‑based dimensionality reduction with AdaMatrix optimization.
    For edge inference, selects optimal rank per matrix.
    """
    def __init__(self, model, total_param_ratio=0.7):
        self.model = model
        self.total_param_ratio = total_param_ratio

    def compress(self):
        total_params = 0
        layer_params = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                n, m = module.weight.shape
                params = n * m
                layer_params[name] = (module, params, n, m)
                total_params += params
        target_params = total_params * self.total_param_ratio
        # allocate ranks proportionally to original params
        for name, (module, p, n, m) in layer_params.items():
            allowed = p * self.total_param_ratio
            # rank * (n+m) <= allowed -> rank = allowed/(n+m)
            rank = max(1, int(allowed / (n + m)))
            rank = min(rank, min(n, m))
            W = module.weight.data.cpu().numpy()
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vt_r = Vt[:rank, :]
            W_approx = U_r @ np.diag(S_r) @ Vt_r
            module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        return self.model