"""
Nyströmformer: linear‑time attention via Nyström approximation.
Paper: https://arxiv.org/abs/2102.03902
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def _compute_moore_penrose(A: torch.Tensor, iters: int = 6) -> torch.Tensor:
    r"""Compute the Moore-Penrose pseudoinverse using iterative Newton-Schulz method:
    $$X_{k+1} = X_k (2I - A X_k)$$
    where $X_0 = A^\top / \|A\|_F^2$.
    """
    norm_A: torch.Tensor = torch.linalg.norm(A, dim=(-2, -1), keepdim=True)
    X: torch.Tensor = A.transpose(-2, -1) / (norm_A + 1e-8)
    I: torch.Tensor = torch.eye(A.size(-1), device=A.device).unsqueeze(0).unsqueeze(0)
    for _ in range(iters):
        X = torch.matmul(X, 2 * I - torch.matmul(A, X))
    return X


@ATTENTION_REGISTRY.register("nystrom")
class NystromAttention(BaseAttention):
    r"""Nyströmformer Linear Attention Mechanism.

    Approximates the full $N \times N$ attention matrix $S = \text{softmax}(Q K^\top / \sqrt{d_k})$
    using the Nyström matrix decomposition with $m \ll N$ landmark points:
    $$\tilde{S} = F_1 F_2^+ F_3$$
    where $F_1 = \text{softmax}(Q \tilde{K}^\top / \sqrt{d_k}) \in \mathbb{R}^{N \times m}$,
    $F_2 = \text{softmax}(\tilde{Q} \tilde{K}^\top / \sqrt{d_k}) \in \mathbb{R}^{m \times m}$, and
    $F_3 = \text{softmax}(\tilde{Q} K^\top / \sqrt{d_k}) \in \mathbb{R}^{m \times N}$, with $F_2^+$
    evaluated via the iterative Newton-Schulz algorithm.

    Computational Complexity:
        - Time Complexity: $\mathcal{O}(B \cdot H \cdot (N \cdot m \cdot d_k + m^3 \cdot \text{iters}))$
        - Memory Complexity: $\mathcal{O}(B \cdot H \cdot N \cdot m)$ vs $\mathcal{O}(B \cdot H \cdot N^2)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, N])$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 64,
        pinv_iterations: int = 6,
        residual: bool = True,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        if rank is None:
            rank = 64
        super().__init__(config, rank)
        self.pinv_iterations: int = pinv_iterations
        self.residual: bool = residual

        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rank: int = rank

        self.query: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Copy weights and biases from pretrained multi-head attention module."""
        with torch.no_grad():
            self.query.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.query.bias.copy_(src.q_lin.bias)
            self.key.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.key.bias.copy_(src.k_lin.bias)
            self.value.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.value.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def _reshape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using Nyström low-rank attention factorisation.

        Args:
            hidden_states (torch.Tensor): Input tensor `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Binary or additive mask.
            head_mask (Optional[torch.Tensor]): Head mask tensor.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Reconstructed output and optional attention matrix.
        """
        batch_size, seq_len, _ = hidden_states.shape
        Q: torch.Tensor = self._reshape(self.query(hidden_states), batch_size)
        K: torch.Tensor = self._reshape(self.key(hidden_states), batch_size)
        V: torch.Tensor = self._reshape(self.value(hidden_states), batch_size)

        num_landmarks: int = self.rank
        if seq_len % num_landmarks == 0:
            seg = seq_len // num_landmarks
            Q_land: torch.Tensor = Q.reshape(
                batch_size, self.num_heads, num_landmarks, seg, self.head_dim
            ).mean(dim=3)
            K_land: torch.Tensor = K.reshape(
                batch_size, self.num_heads, num_landmarks, seg, self.head_dim
            ).mean(dim=3)
        else:
            idx = torch.linspace(
                0,
                seq_len - 1,
                num_landmarks,
                dtype=torch.long,
                device=hidden_states.device,
            )
            Q_land = Q[:, :, idx, :]
            K_land = K[:, :, idx, :]

        scale: float = 1.0 / (self.head_dim**0.5)
        F1: torch.Tensor = torch.matmul(Q, K_land.transpose(-2, -1)) * scale
        F2: torch.Tensor = torch.matmul(Q_land, K_land.transpose(-2, -1)) * scale
        F3: torch.Tensor = torch.matmul(Q_land, K.transpose(-2, -1)) * scale

        F2_pinv: torch.Tensor = _compute_moore_penrose(F2, self.pinv_iterations)
        attn: torch.Tensor = torch.matmul(F1, torch.matmul(F2_pinv, F3))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(dtype=attn.dtype)
                attn = attn + (1.0 - mask) * -1e9
            else:
                attn = attn + attention_mask

        probs: torch.Tensor = F.softmax(attn, dim=-1)
        probs = self.dropout(probs)

        context: torch.Tensor = torch.matmul(probs, V)
        if self.residual:
            context = context + V
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        out: torch.Tensor = self.out_proj(context)

        if output_attentions:
            return out, attn
        return out, None


def inject_nystrom(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> NystromAttention:
    """Injection helper for Nyström attention."""
    rank: int = getattr(comp_config, "rank", 64)
    residual: bool = getattr(comp_config, "residual", True)
    new_attn: NystromAttention = NystromAttention(
        config=model_config,
        rank=rank,
        residual=residual,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
