import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

class HGRN2OuterBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W_f = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, state=None, **kwargs):
        B, L, D = x.shape
        if state is None:
            state = torch.zeros(B, D, D, device=x.device)
        outputs = []
        for t in range(L):
            xt = x[:, t]
            f = torch.sigmoid(self.W_f(xt))
            k = self.W_k(xt)
            v = self.W_v(xt)
            state = f.unsqueeze(-1) * state + (1 - f.unsqueeze(-1)) * (k.unsqueeze(-1) @ v.unsqueeze(1))
            yt = (state @ xt.unsqueeze(-1)).squeeze(-1)
            outputs.append(yt)
        out = torch.stack(outputs, dim=1)
        return self.out_proj(out), None

@ATTENTION_REGISTRY.register("hgrn2_outer")
def inject_hgrn2_outer(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    dim = model_config.hidden_size
    new_block = HGRN2OuterBlock(dim)
    layer.attention = new_block
    return new_block
