"""Dobi-SVD: Differentiable SVD with learnable rank ratio. Registered as 'dobi'."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY

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
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        max_rank = min(self.in_features, self.out_features)
        effective_rank = max(1, int(self.rank_ratio * max_rank))
        U, S, Vt = torch.linalg.svd(self.weight, full_matrices=False)
        U_r = U[:, :effective_rank]
        S_r = S[:effective_rank]
        Vt_r = Vt[:effective_rank, :]
        W_compressed = U_r @ torch.diag(S_r) @ Vt_r
        return F.linear(x, W_compressed, self.bias)


@COMPRESSOR_REGISTRY.register("dobi")
class DobiSVDCompressor(BaseCompressor):
    def __init__(self, model, init_rank_ratio=0.5, **kwargs):
        super().__init__(model, name="dobi", **kwargs)
        self.init_rank_ratio = init_rank_ratio

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                new_layer = DobiSVDLinearLayer(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    init_rank_ratio=self.init_rank_ratio
                )
                parent = self._get_parent(name)
                if parent is not None:
                    setattr(parent, name.split('.')[-1], new_layer)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _get_parent(self, full_name):
        parts = full_name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
        return parent