import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

class TokenCompressedAttention(nn.Module):
    def __init__(self, config, comp_config, layer_idx, num_layers, old_attn):
        super().__init__()
        self.attn = old_attn

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return self.attn(hidden_states, attention_mask=attention_mask, **kwargs)

@ATTENTION_REGISTRY.register("token_compressed")
def inject_token_compressed(layer, config, comp_config, layer_idx, num_layers, old_attn):
    layer.attention = TokenCompressedAttention(config, comp_config, layer_idx, num_layers, old_attn)

