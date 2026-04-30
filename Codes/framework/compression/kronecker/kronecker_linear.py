# framework/compression/kronecker_linear.py
import torch
import torch.nn as nn

class KroneckerLinear(nn.Module):
    """
    Kronecker product decomposition of weight matrix.
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        # Factorize in_features = a*b, out_features = c*d
        # Here we use simple square root
        a = int(in_features ** 0.5)
        b = in_features // a
        c = int(out_features ** 0.5)
        d = out_features // c
        self.A = nn.Parameter(torch.randn(c, a))
        self.B = nn.Parameter(torch.randn(d, b))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Kronecker product: A ⊗ B
        W = torch.kron(self.A, self.B)
        return x @ W.T + self.bias