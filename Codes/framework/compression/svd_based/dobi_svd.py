"""
Dobi‑SVD: Differentiable SVD for LLM Compression.
Paper: https://arxiv.org/abs/2502.02723
Code: https://github.com/wangqinsi1/Dobi-SVD
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DobiSVDLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_rank_ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank_ratio = nn.Parameter(torch.tensor(init_rank_ratio))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        effective_rank = max(1, int(self.rank_ratio * min(self.in_features, self.out_features)))
        U, S, Vt = torch.linalg.svd(self.weight, full_matrices=False)
        U_r = U[:, :effective_rank]
        S_r = S[:effective_rank]
        Vt_r = Vt[:effective_rank, :]
        W_compressed = U_r @ torch.diag(S_r) @ Vt_r
        return F.linear(x, W_compressed, self.bias)