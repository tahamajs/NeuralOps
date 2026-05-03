import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

def ssd_kernel(q, k, v, chunk_size=64):
    """
    Causal linear attention with chunked scan (SSD simplified).
    q, k, v: (B, L, D), all projected identically for demo.
    """
    B, L, D = q.shape
    # For causal linear attention: output_t = sum_{j<=t} q_t k_j v_j / sum_{j<=t} q_t k_j
    # We'll implement a chunked parallel version for speed.
    # Here a simple recurrent loop for clarity; replace with Triton SSD for production.
    h = torch.zeros(B, D, device=q.device, dtype=q.dtype)
    norm = torch.zeros(B, 1, device=q.device, dtype=q.dtype)
    outputs = []
    for t in range(L):
        kt = k[:, t]        # (B, D)
        vt = v[:, t]
        qt = q[:, t]
        h = h + kt.unsqueeze(-1) * vt.unsqueeze(1)    # (B, D, D)
        norm = norm + kt.sum(dim=1, keepdim=True)      # (B, 1)
        out_t = (qt.unsqueeze(1) @ h).squeeze(1)       # (B, D)
        # Normalize (optional, for linear attention stability)
        # out_t = out_t / (norm + 1e-8)
        outputs.append(out_t)
    return torch.stack(outputs, dim=1)


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

        # Local sliding window attention
        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)

        # SSD recurrent projection
        self.ssd_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape

        # Local window attention (causal, fixed window)
        q_local = self.W_q(x)
        k_local = self.W_k(x)
        v_local = self.W_v(x)
        scores = torch.bmm(q_local, k_local.transpose(1,2)) / (D**0.5)
        idx = torch.arange(L, device=x.device)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        window = (idx.unsqueeze(1) - idx.unsqueeze(0)) < self.window_size
        local_mask = ~(causal & window)
        scores = scores.masked_fill(local_mask.unsqueeze(0), float('-inf'))
        local_out = torch.bmm(F.softmax(scores, dim=-1), v_local)

        # Global linear attention via SSD
        q_global = self.ssd_proj(x)
        global_out = ssd_kernel(q_global, self.W_k(x), self.W_v(x))

        out = self.out_proj(local_out + global_out)
        return (out, None)