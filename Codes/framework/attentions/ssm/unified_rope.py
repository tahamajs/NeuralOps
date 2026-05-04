import torch.nn as nn

class UnifiedRoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, x):
        return x