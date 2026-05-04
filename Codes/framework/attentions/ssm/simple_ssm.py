import torch.nn as nn

class SimpleSSM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.linear(x)