from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("svd")
class SVDLinearAttention(nn.Module):
    r"""Low-Rank SVD-Compressed Attention Mechanism.

    Reduces quadratic sequence complexity $\mathcal{O}(N^2)$ to linear in sequence length $\mathcal{O}(N \cdot r)$
    by projecting Key and Value representations into the top-$r$ left-singular subspace of the sequence representation:
    $$X \approx U_r \Sigma_r V_r^\top$$
    $$K_{\text{proj}} = U_r^\top K \in \mathbb{R}^{B \times H \times r \times d_k}, \quad V_{\text{proj}} = U_r^\top V \in \mathbb{R}^{B \times H \times r \times d_k}$$
    $$\text{Attention}_{\text{SVD}}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\text{proj}}^\top}{\sqrt{d_k}} + M\right) V_{\text{proj}}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot (N \cdot r \cdot d_k + N \cdot d^2))$
        - Memory: $\mathcal{O}(B \cdot (N \cdot r + r \cdot d_k))$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 64,
        dropout: Optional[float] = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rank: Optional[int] = rank
        self.query: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        dropout_val: float = (
            dropout if dropout is not None else getattr(config, "dropout", 0.1)
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_val)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape and transpose input tensor for multi-head attention scoring.

        Args:
            x (torch.Tensor): Tensor of shape `[B, N, all_head_size]`.

        Returns:
            torch.Tensor: Transposed tensor of shape `[B, num_heads, N, head_dim]`.
        """
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of SVD-compressed multi-head attention.

        Args:
            hidden_states (torch.Tensor): Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Mask tensor of shape `[Batch, 1, SeqLen, SeqLen]`.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Tuple containing output tensor `[Batch, SeqLen, HiddenDim]` and `None`.
        """
        batch_size, seq_len, _ = hidden_states.shape
        Q: torch.Tensor = self.transpose_for_scores(self.query(hidden_states))
        K: torch.Tensor = self.transpose_for_scores(self.key(hidden_states))
        V: torch.Tensor = self.transpose_for_scores(self.value(hidden_states))

        if self.rank is None or seq_len <= self.rank:
            scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) / (
                self.head_dim**0.5
            )
            if attention_mask is not None:
                scores = scores + attention_mask
            probs: torch.Tensor = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            context: torch.Tensor = torch.matmul(probs, V)
        else:
            hidden_cpu = hidden_states.cpu()
            U_r = []
            for b in range(batch_size):
                U, S, _ = torch.svd_lowrank(hidden_cpu[b], q=self.rank, niter=2)
                U_r.append(U[:, : self.rank])
            U_r_t = torch.stack(U_r, dim=0).to(hidden_states.device)
            U_r_t = U_r_t.unsqueeze(1)  # (B, 1, seq_len, rank)
            K_proj: torch.Tensor = torch.matmul(U_r_t.transpose(-2, -1), K)
            V_proj: torch.Tensor = torch.matmul(U_r_t.transpose(-2, -1), V)
            scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / (
                self.head_dim**0.5
            )
            if attention_mask is not None:
                scores = scores + attention_mask.unsqueeze(-1)
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            context = torch.matmul(probs, V_proj)

        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        return (context, None)

    def get_ortho_loss(self) -> torch.Tensor:
        """Compute orthogonality regularization loss for projection weights.

        Returns:
            torch.Tensor: Scalar zero tensor.
        """
        return torch.tensor(0.0, device=self.query.weight.device)
