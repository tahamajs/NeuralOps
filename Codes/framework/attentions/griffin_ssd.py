import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def ssd_kernel(q, k, v):
    """Causal linear attention (SSD‑style simplified). q,k,v: (B, L, D)."""
    B, L, D = q.shape
    h = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)
    outputs = []
    for t in range(L):
        kt = k[:, t]
        vt = v[:, t]
        qt = q[:, t]
        h = h + kt.unsqueeze(-1) @ vt.unsqueeze(1)  # outer product
        out_t = (qt.unsqueeze(1) @ h).squeeze(1)  # (B, D)
        outputs.append(out_t)
    return torch.stack(outputs, dim=1)


@ATTENTION_REGISTRY.register("griffin_ssd")
class GriffinSSDBlock(BaseAttention):
    """Local sliding window + global SSD recurrence."""

    def __init__(self, config, rank=64, window_size=256):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.window_size = window_size

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.ssd_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, attention_mask=None, mask=None, **kwargs):
        if attention_mask is None and mask is not None:
            attention_mask = mask
        B, L, D = x.shape

        # Local sliding window (causal)
        q_local = self.W_q(x)
        k_local = self.W_k(x)
        v_local = self.W_v(x)
        scores = torch.bmm(q_local, k_local.transpose(1, 2)) / (D**0.5)
        idx = torch.arange(L, device=x.device)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        window = (idx.unsqueeze(1) - idx.unsqueeze(0)) < self.window_size
        local_mask = ~(causal & window)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = attention_mask.unsqueeze(1) == 0  # (B, 1, L)
                scores = scores.masked_fill(pad_mask, float("-1e9"))
        scores = scores.masked_fill(local_mask.unsqueeze(0), float("-1e9"))
        local_out = torch.bmm(F.softmax(scores, dim=-1), v_local)

        # Global linear attention
        q_global = self.ssd_proj(x)
        k_global = self.W_k(x)
        v_global = self.W_v(x)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                m = attention_mask.unsqueeze(-1)
                k_global = k_global * m
                v_global = v_global * m
        global_out = ssd_kernel(q_global, k_global, v_global)

        out = self.out_proj(local_out + global_out)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                out = out * attention_mask.unsqueeze(-1)
        return out, None

