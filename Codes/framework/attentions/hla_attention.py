# framework/attentions/specialized/hla_attention.py
import torch
import torch.nn as nn
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hla")
class HLAAttention(BaseAttention):
    """
    Hadamard Linear Attention (HLA).
    Paper: https://arxiv.org/abs/2602.12128
    """

    def __init__(self, config, rank: int = None):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x, attention_mask=None, **kwargs):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, Dh)
        k = self.k_proj(x).view(B, L, H, Dh)
        v = self.v_proj(x).view(B, L, H, Dh)

        q = q.permute(0, 2, 1, 3)  # (B, H, L, Dh)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
            elif attention_mask.dim() == 4:
                mask = (attention_mask > -1).float().permute(0, 2, 3, 1)
            else:
                mask = attention_mask
            k = k * mask
            v = v * mask

        # Recurrent linear attention with Hadamard product
        h = torch.zeros(B, H, Dh, Dh, device=x.device)
        outputs = []
        for t in range(L):
            qt = q[:, :, t, :]  # (B, H, Dh)
            kt = k[:, :, t, :]  # (B, H, Dh)
            vt = v[:, :, t, :]  # (B, H, Dh)
            h = h + torch.einsum("bhd,bhe->bhde", kt, vt)
            out_t = torch.einsum("bhd,bhde->bhe", qt, h)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=2)  # (B, H, L, Dh)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                out = out * attention_mask.unsqueeze(-1)
        return out, None

