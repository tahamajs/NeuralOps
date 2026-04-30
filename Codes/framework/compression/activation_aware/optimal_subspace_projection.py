# framework/compression/optimal_subspace_projection.py
import torch
import torch.nn as nn

class ActivationAwareSubspaceCompressor:
    def __init__(self, model, calibration_loader, rank_ratio=0.5, power_iters=2):
        self.model = model
        self.calibration_loader = calibration_loader
        self.rank_ratio = rank_ratio         # target rank / original dimension
        self.power_iters = power_iters

    def compress(self):
        # 1. Collect activation covariances
        act_covs = {}
        def hook_fn(module, input, output):
            x = input[0].detach()   # (B, L, D)
            # Flatten across batch and sequence
            x_flat = x.reshape(-1, x.shape[-1])
            cov = x_flat.T @ x_flat / x_flat.shape[0]
            if id(module) not in act_covs:
                act_covs[id(module)] = cov
            else:
                act_covs[id(module)] += cov

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and not any(skip in name for skip in ['embeddings', 'classifier']):
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            for batch in self.calibration_loader:
                batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
                self.model(**batch)

        for h in hooks:
            h.remove()

        # 2. Compress each linear layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and not any(skip in name for skip in ['embeddings', 'classifier']):
                W = module.weight.data
                C = act_covs[id(module)]
                # Cholesky factor
                L = torch.linalg.cholesky(C)
                M = W @ L

                # Randomized subspace iteration
                rank = max(1, int(W.shape[0] * self.rank_ratio))
                Ω = torch.randn(M.shape[1], rank, device=W.device)
                Y = M @ Ω
                for _ in range(self.power_iters):
                    Y = M @ (M.T @ Y)
                Q, _ = torch.linalg.qr(Y)
                B = Q.T @ M
                U_r, S_r, Vt_r = torch.linalg.svd(B, full_matrices=False)
                U_r = U_r[:, :rank]
                S_r = S_r[:rank]
                Vt_r = Vt_r[:rank, :]

                # Fused low‑rank layer
                new_U = Q @ U_r
                new_V = Vt_r @ torch.linalg.solve(L.T, torch.eye(L.shape[0], device=W.device)).T
                new_weight = new_U @ torch.diag(S_r) @ new_V

                module.weight.data = new_weight
                # Keep bias unchanged
        return self.model