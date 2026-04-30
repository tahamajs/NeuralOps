import torch
import numpy as np

class AdaSVDCompressor:
    """
    Adaptive SVD: compute per‑layer importance and assign rank accordingly,
    then compensate truncation error by alternating U/V updates.
    """
    def __init__(self, model, target_param_ratio=0.3, num_iters=5):
        self.model = model
        self.target_ratio = target_param_ratio
        self.num_iters = num_iters

    def _weight_importance(self, W):
        # Use nuclear norm or sum of singular values
        _, S, _ = np.linalg.svd(W, full_matrices=False)
        return np.sum(S)

    def compress(self):
        total_importance = 0
        layer_importances = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                imp = self._weight_importance(module.weight.data.cpu().numpy())
                layer_importances[name] = imp
                total_importance += imp

        # Assign ranks proportional to importance
        total_params = sum(p.numel() for p in self.model.parameters())
        target_params = total_params * (1 - self.target_ratio)
        for name, module in self.model.named_modules():
            if name in layer_importances:
                frac = layer_importances[name] / total_importance
                max_rank = min(module.weight.shape[0], module.weight.shape[1])
                rank = max(1, int(max_rank * frac * (1 - self.target_ratio)))
                W = module.weight.data.cpu().numpy()
                # Initial SVD
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                U_r = U[:, :rank]
                S_r = S[:rank]
                Vt_r = Vt[:rank, :]
                W_approx = U_r @ np.diag(S_r) @ Vt_r
                # Iteratively refine U,V
                for _ in range(self.num_iters):
                    # Fix V, update U
                    U_r = W @ Vt_r.T @ np.linalg.inv(np.diag(S_r) + 1e-6 * np.eye(rank))
                    # Fix U, update V
                    Vt_r = np.linalg.inv(U_r.T @ U_r + 1e-6 * np.eye(rank)) @ U_r.T @ W
                    # Recompute S as diag of singular values after update? simpler: recompute full approximation
                    W_approx = U_r @ np.diag(S_r) @ Vt_r
                module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        return self.model