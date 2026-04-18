from torch import nn

class BaseAttention(nn.Module):
    def __init__(self, config, rank=None):
        super().__init__()
        self.config = config
        self.rank = rank

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        raise NotImplementedError