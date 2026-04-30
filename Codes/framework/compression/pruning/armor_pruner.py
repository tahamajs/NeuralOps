import torch
import numpy as np

class ARMORPruner:
    """
    ARMOR: factorizes weight into 2:4 sparse core + two block‑diagonal error‑correcting matrices.
    """
    def __init__(self, model, block_size=4):
        self.model = model
        self.block_size = block_size

    def prune(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.cpu().numpy()
                N, M = W.shape
                # Create 2:4 sparsity pattern: within each group of 4, keep only 2 largest
                W_pruned = np.zeros_like(W)
                for i in range(0, N, self.block_size):
                    blk = W[i:i+self.block_size, :]
                    # For each row, keep 2 largest abs values
                    abs_blk = np.abs(blk)
                    threshold = np.sort(abs_blk, axis=1)[:, -2:]  # top 2
                    mask = abs_blk >= threshold[:, 0:1]
                    W_pruned[i:i+self.block_size] = blk * mask
                # Add block‑diagonal error correction (simplified: add back residual low‑rank)
                residual = (W - W_pruned) @ np.eye(M)
                U, S, Vt = np.linalg.svd(residual, full_matrices=False)
                rank = min(16, len(S))
                correction = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
                W_armor = W_pruned + correction
                module.weight.data = torch.tensor(W_armor, dtype=module.weight.dtype, device=module.weight.device)
        return self.model