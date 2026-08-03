"""
Nyströmformer Attention – Nyström‑based low‑rank kernel approximation.
Paper: https://arxiv.org/abs/2102.03902
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("nystrom")
class NystromAttention(nn.Module):
    r"""Nyströmformer: Linear Complexity Self-Attention via Nyström Landmark Approximation (Xiong et al., 2021).

    Approximates standard $\mathcal{O}(N^2)$ attention using $m \ll N$ landmark tokens:
    $$\hat{A} = \hat{F}_1 \hat{F}_2^+ \hat{F}_3$$
    $$\hat{F}_1 = \text{softmax}\left(\frac{Q \tilde{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times m}, \quad \hat{F}_2 = \text{softmax}\left(\frac{\tilde{Q} \tilde{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{m \times m}, \quad \hat{F}_3 = \text{softmax}\left(\frac{\tilde{Q} K^\top}{\sqrt{d_k}} + M\right) \in \mathbb{R}^{m \times N}$$
    $$\text{Attention}_{\text{Nyström}}(Q, K, V) = \hat{F}_1 \left( \hat{F}_2^+ \left( \hat{F}_3 V \right) \right)$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot (N \cdot m \cdot d_k + m^3))$
        - Memory: $\mathcal{O}(B \cdot H \cdot (N \cdot m + m \cdot d_k))$ vs $\mathcal{O}(B \cdot H \cdot N^2)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any,
        num_landmarks: Optional[int] = None,
        pinv_iterations: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim

        self.num_landmarks: int = getattr(
            config,
            "num_landmarks",
            num_landmarks if num_landmarks is not None else getattr(config, "num_landmarks", 64),
        )
        self.pinv_iterations: int = pinv_iterations

        self.query: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, config.hidden_size)
        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def _iterative_pinv(self, A: torch.Tensor, iters: int = 6) -> torch.Tensor:
        r"""Compute Moore-Penrose pseudo-inverse using iterative Schulz updates."""
        # A: (B, H, m, m)
        # Compute ||A||_1 and ||A||_inf
        norm_1 = A.abs().sum(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
        norm_inf = A.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
        alpha = 1.0 / (norm_1 * norm_inf + 1e-7)

        # V_0 = alpha * A^T
        V = alpha * A.transpose(-2, -1)
        I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)

        for _ in range(iters):
            # V_{k+1} = V_k (2I - A V_k)
            V = torch.matmul(V, 2.0 * I - torch.matmul(A, V))
        return V

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for true O(N) linear Nyströmformer attention."""
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q: torch.Tensor = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
        K: torch.Tensor = self.key(hidden_states).view(B, N, H, d).transpose(1, 2)
        V: torch.Tensor = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)

        m = min(self.num_landmarks, N)

        # Landmark extraction via adaptive average pooling over sequence dimension
        # Q: (B, H, N, d) -> transpose to (B*H, d, N) for pooling
        Q_flat = Q.reshape(B * H, N, d).transpose(1, 2)
        K_flat = K.reshape(B * H, N, d).transpose(1, 2)

        Q_landmarks = F.adaptive_avg_pool1d(Q_flat, m).transpose(1, 2).reshape(B, H, m, d)
        K_landmarks = F.adaptive_avg_pool1d(K_flat, m).transpose(1, 2).reshape(B, H, m, d)

        scale = d**-0.5
        # Kernel 1: Q @ K_landmarks^T -> (B, H, N, m)
        F1 = F.softmax(torch.matmul(Q, K_landmarks.transpose(-2, -1)) * scale, dim=-1)

        # Kernel 2: Q_landmarks @ K_landmarks^T -> (B, H, m, m)
        F2 = F.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-2, -1)) * scale, dim=-1)

        # Kernel 3: Q_landmarks @ K^T -> (B, H, m, N)
        scores3 = torch.matmul(Q_landmarks, K.transpose(-2, -1)) * scale

        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # (B, N)
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        scores3 = scores3.masked_fill(~mask_expanded, -10000.0)
        F3 = F.softmax(scores3, dim=-1)

        # Compute F2^+
        F2_pinv = self._iterative_pinv(F2, iters=self.pinv_iterations)

        # Associative linear multiplication: O(N * m * d)
        # step1 = F3 @ V: (B, H, m, d)
        step1 = torch.matmul(F3, V)
        # step2 = F2_pinv @ step1: (B, H, m, d)
        step2 = torch.matmul(F2_pinv, step1)
        # context = F1 @ step2: (B, H, N, d)
        context = torch.matmul(F1, step2)

        context = self.dropout(context)
        if head_mask is not None:
            context = context * head_mask

        # Mask padding queries
        context = context * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out: torch.Tensor = self.out_proj(context)

        return out, None
