# framework/compression/cur_moe.py
import torch
import torch.nn as nn

class CURMoECompressor:
    """
    CUR-MoE: Interpretable CUR decomposition for MoE experts.
    """
    def cur_compress(self, W, rank):
        # Column selection (random for simplicity)
        col_idx = torch.randperm(W.shape[1])[:rank]
        row_idx = torch.randperm(W.shape[0])[:rank]
        C = W[:, col_idx]
        R = W[row_idx, :]
        U = torch.linalg.lstsq(C, W @ torch.linalg.pinv(R)).solution
        return C @ U @ R

    def compress(self, model):
        for name, module in model.named_modules():
            if 'expert' in name and isinstance(module, nn.Linear):
                module.weight.data = self.cur_compress(module.weight.data, rank=32)
        return model