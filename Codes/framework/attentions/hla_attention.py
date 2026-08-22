from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hla")
class HLAAttention(BaseAttention):
    r"""
    Hadamard Linear Attention (HLA).

    Mathematical Mechanism:
        Maintains an associative causal matrix state $H_t \in \mathbb{R}^{B \times H \times d_k \times d_k}$
        across sequence steps:
        $$\begin{aligned}
        Q &= X W_Q, \quad K = X W_K, \quad V = X W_V \\
        H_t &= H_{t-1} + K_t \otimes V_t = H_{t-1} + K_t^\top V_t \\
        O_t &= Q_t H_t = \sum_{j=1}^{t} (Q_t K_j^\top) V_j \\
        \text{Output} &= \text{Concat}(O_1, \dots, O_H) W_{\text{out}}
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot L \cdot d_k^2)$ — linear $\mathcal{O}(L)$ in sequence length.
        - Space: $\mathcal{O}(B \cdot H \cdot d_k^2)$ recurrent memory footprint.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.hidden_size: int = getattr(config, "hidden_size", 768)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        self.head_dim: int = self.hidden_size // self.num_heads if self.num_heads > 0 else 64

        self.q_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Causal linear attention forward pass.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional padding mask `[B, L]` or additive mask `[B, 1, L, L]`.
            head_mask: Optional head masking tensor.
            output_attentions: Unused flag for interface compliance.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, None)` with `output` of shape `[B, L, D]`.
        """
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, Dh)
        k = self.k_proj(hidden_states).view(B, L, H, Dh)
        v = self.v_proj(hidden_states).view(B, L, H, Dh)

        q = q.permute(0, 2, 1, 3)  # (B, H, L, Dh)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask = (attention_mask[:, 0, 0, :] > -10000.0).float()
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, 0, :]
            else:
                mask = attention_mask.float()
            
            mask = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
            k = k * mask
            v = v * mask

        # Recurrent linear attention with outer product accumulation
        h = torch.zeros(B, H, Dh, Dh, device=hidden_states.device, dtype=hidden_states.dtype)
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
