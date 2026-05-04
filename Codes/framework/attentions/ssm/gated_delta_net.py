import torch.nn as nn

class GatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model, use_spectral_norm=False):
        super().__init__()
        self.net = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.net(x)