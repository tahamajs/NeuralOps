import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("hla")
class HLAAttention(BaseAttention):
    """
    Hadamard Linear Attention (HLA): Efficient linear attention using element‑wise
    Hadamard products and cumulative sums to reduce to O(N) complexity.
    """
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.dim = config.hidden_size

        self.W_q = nn.Linear(self.dim, self.dim)
        self.W_k = nn.Linear(self.dim, self.dim)
        self.W_v = nn.Linear(self.dim, self.dim)
        self.out = nn.Linear(self.dim, self.dim)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.W_q(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        k = self.W_k(x).view(B, L, H, d).transpose(1, 2)
        v = self.W_v(x).view(B, L, H, d).transpose(1, 2)

        # Apply non‑negativity activation (like ReLU) to Q and K for linear attention
        q_lin = F.relu(q)
        k_lin = F.relu(k)

        # Linear attention: output = (Q_lin * (K_lin^T * V)) / sum
        # We compute K^T V using einsum
        kv = torch.einsum('bhnd,bhne->bhnde', k_lin, v)  # (B, H, d, d)
        # Cumulative summing over time? We'll do a simple recurrent linear attention without cumsum for now.
        # Instead, use standard linear attention (global) – O(L) via matrix multiplication
        # For causal, we need causal masking, but we omit here for brevity.

        # Actually HLA uses Hadamard product and recurrence: out_t = q_t * sum_{i<=t} (k_i * v_i)
        # We'll implement a simple recurring scan:
        h = torch.zeros(B, H, d, d, device=q.device)
        outputs = []
        for t in range(L):
            qt = q_lin[:, :, t]    # (B, H, d)
            kt = k_lin[:, :, t]
            vt = v[:, :, t]
            # Update hidden state: h = h + kt * vt^T (outer product)
            h = h + torch.einsum('bhk,bhv->bhkv', kt, vt)
            # Output: qt @ h
            out_t = torch.einsum('bhd,bhdk->bhk', qt, h)
            outputs.append(out_t)
        out_seq = torch.stack(outputs, dim=2)  # (B, H, L, d)
        out_seq = out_seq.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out_seq), None