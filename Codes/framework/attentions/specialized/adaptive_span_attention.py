import torch, torch.nn as nn, torch.nn.functional as F
from framework.attentions.core.registry import ATTENTION_REGISTRY

class AdaptiveSpanAttention(nn.Module):
    def __init__(self, config, max_span=1024,
                 comp_config=None, layer_idx=None, total_layers=None, old_attn=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.span_raw = nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, x, attention_mask=None, **kwargs):
        B, L, D = x.shape; H = self.num_heads; d = self.head_dim
        q = self.q_proj(x).view(B, L, H, d).permute(0,2,1,3)
        k = self.k_proj(x).view(B, L, H, d).permute(0,2,1,3)
        v = self.v_proj(x).view(B, L, H, d).permute(0,2,1,3)
        attn = torch.matmul(q, k.transpose(-2,-1)) * (d**-0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(B, L, H*d)
        return self.out_proj(out)

@ATTENTION_REGISTRY.register("adaptive_span")
def inject_adaptive_span(layer, config, comp_config, layer_idx, total_layers, old_attn):
    new_attn = AdaptiveSpanAttention(config, comp_config=comp_config, layer_idx=layer_idx, total_layers=total_layers, old_attn=old_attn)
    if old_attn is not None:
        new_attn.q_proj.weight.data.copy_(old_attn.q_lin.weight.data)
        new_attn.q_proj.bias.data.copy_(old_attn.q_lin.bias.data)
        new_attn.k_proj.weight.data.copy_(old_attn.k_lin.weight.data)
        new_attn.k_proj.bias.data.copy_(old_attn.k_lin.bias.data)
        new_attn.v_proj.weight.data.copy_(old_attn.v_lin.weight.data)
        new_attn.v_proj.bias.data.copy_(old_attn.v_lin.bias.data)
        new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
        new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
    layer.attention = new_attn
    return new_attn
