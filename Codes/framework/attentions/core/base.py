"""Base class for custom attention modules."""
import torch.nn as nn

class BaseAttention(nn.Module):
    def __init__(self, config, rank=None):
        super().__init__()
        self.config = config
        self.rank = rank

    def get_ortho_loss(self):
        return 0.0
