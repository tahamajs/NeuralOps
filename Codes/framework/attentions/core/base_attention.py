import torch.nn as nn

class BaseAttention(nn.Module):
    def __init__(self, config, rank=None):
        super().__init__()
        self.config = config
        self.rank = rank
