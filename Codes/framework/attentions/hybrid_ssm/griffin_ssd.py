import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_attention import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

def ssd_kernel(q, k, v, state=None):
    """
    Simplified SSD kernel using block-decomposition to mimic masked attention.
    For production, use the Triton‑implemented SSD kernel.
    """
    # Here we just implement a causal linear attention equivalent
    B, L, D = q.shape
    scale = D ** -0.5
    q = q * scale
    attn = torch.bmm(q, k.transpose(1,2))   # (B, L, L)
    mask = torch.tril(torch.ones(L, L, device=q.device))
    attn = attn.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return torch.bmm(attn, v)

@ATTENTION_REGISTRY.register("griffin_ssd")
class GriffinSSDBlock(BaseAttention):
    """
    Griffin‑SSD block: replaces RG‑LRU with an SSD‑based recurrence
    and maintains a local sliding window attention.
    """
    def __init__(self, config, rank=64, window_size=256):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.window_size = window_size

        # Local attention
        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)

        # SSD recurrent core
        self.ssd_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape

        # Local window attention
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        # causal sliding window: mask past tokens beyond window
        window_mask = torch.triu(torch.ones(L, L, device=x.device), -self.window_size).bool()
        scores = torch.bmm(q, k.transpose(1,2)) / (D ** 0.5)
        scores = scores.masked_fill(window_mask.unsqueeze(0), float('-inf'))
        local_out = torch.bmm(F.softmax(scores, dim=-1), v)

        # SSD‑recurrent global context
        global_out = ssd_kernel(self.ssd_proj(x), k, v)   # same K,V for simplicity
        out = self.out_proj(local_out + global_out)
        return (out, None)