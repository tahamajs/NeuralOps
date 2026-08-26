from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("svd")
class SVDLinearAttention(nn.Module):
    r"""Low-Rank SVD-Compressed Sequence Attention Mechanism.

    Reduces quadratic sequence complexity $\mathcal{O}(N^2)$ to linear in sequence length $\mathcal{O}(N \cdot r)$
    by projecting Key and Value sequence dimensions into the top-$r$ left-singular subspace of the sequence representation:
    $$X_{\text{valid}} = X \odot M_{\text{valid}}$$
    $$X \approx U_r \Sigma_r V_r^\top, \quad U_r \in \mathbb{R}^{B \times N \times r}$$
    $$K_{\text{proj}} = U_r^\top K \in \mathbb{R}^{B \times H \times r \times d_k}, \quad V_{\text{proj}} = U_r^\top V \in \mathbb{R}^{B \times H \times r \times d_k}$$
    $$\text{Attention}_{\text{SVD}}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\text{proj}}^\top}{\sqrt{d_k}}\right) V_{\text{proj}}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot (N \cdot r \cdot d_k + N \cdot d^2))$
        - Memory: $\mathcal{O}(B \cdot (N \cdot r + r \cdot d_k))$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, r])$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 64,
        dropout: Optional[float] = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rank: Optional[int] = rank if rank is not None else 64

        self.query: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, config.hidden_size)
        dropout_val: float = (
            dropout if dropout is not None else getattr(config, "dropout", 0.1)
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_val)
        self._last_u_r: Optional[torch.Tensor] = None

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape and transpose input tensor for multi-head attention scoring."""
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of SVD-compressed multi-head attention on device."""
        # Removed undefined variable references
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q: torch.Tensor = self.transpose_for_scores(self.query(hidden_states))
        K: torch.Tensor = self.transpose_for_scores(self.key(hidden_states))
        V: torch.Tensor = self.transpose_for_scores(self.value(hidden_states))

        effective_rank = min(self.rank, seq_len) if self.rank is not None else seq_len

        if self.rank is None or seq_len <= self.rank:
            scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) / (
                self.head_dim**0.5
            )
            scores = apply_attention_mask(scores, attention_mask)
            probs: torch.Tensor = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            if head_mask is not None:
                probs = probs * head_mask
            context: torch.Tensor = torch.matmul(probs, V)
            attn_ret = probs if output_attentions else None
        else:
            # Extract valid mask to prevent padding from biasing SVD subspace
            valid_mask = get_token_valid_mask(attention_mask, batch_size, seq_len, device)
            masked_h = hidden_states * valid_mask.unsqueeze(-1).to(dtype)

            # Compute orthonormal subspace using randomized QR (Halko et al., 2011) inside no_grad to prevent QR autograd instability
            with torch.no_grad():
                U_list = []
                for b in range(batch_size):
                    h_b_float = masked_h[b].float()
                    try:
                        D_dim = h_b_float.size(1)
                        omega = torch.randn(D_dim, effective_rank, device=device, dtype=torch.float32)
                        Y = torch.matmul(h_b_float, omega)
                        Q_mat, _ = torch.linalg.qr(Y)
                        U_r_b = Q_mat[:, :effective_rank].to(device=device, dtype=dtype)
                    except Exception:
                        h_b_cpu = h_b_float.cpu().double()
                        D_dim = h_b_cpu.size(1)
                        omega = torch.randn(D_dim, effective_rank, device="cpu", dtype=torch.float64)
                        Y = torch.matmul(h_b_cpu, omega)
                        Q_mat, _ = torch.linalg.qr(Y)
                        U_r_b = Q_mat[:, :effective_rank].to(device=device, dtype=dtype)
                    U_list.append(U_r_b)

                U_r = torch.stack(U_list, dim=0)  # (B, seq_len, r)
            self._last_u_r = U_r

            U_r_t = U_r.unsqueeze(1)  # (B, 1, seq_len, r)
            K_proj: torch.Tensor = torch.matmul(U_r_t.transpose(-2, -1), K)  # (B, H, r, d)
            V_proj: torch.Tensor = torch.matmul(U_r_t.transpose(-2, -1), V)  # (B, H, r, d)

            scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / (self.head_dim**0.5)  # (B, H, N, r)
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)

            if head_mask is not None:
                probs = probs * head_mask

            context = torch.matmul(probs, V_proj)  # (B, H, N, d)

            # Mask out padding query positions
            context = context * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
            attn_ret = probs if output_attentions else None

        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        out = self.out_proj(context)
        return (out, attn_ret)

    def get_ortho_loss(self) -> torch.Tensor:
        """Compute orthogonality regularization loss for left-singular vectors."""
        if self._last_u_r is None:
            return torch.tensor(0.0, device=self.query.weight.device)
        # U_r: (B, N, r) -> U_r^T @ U_r should be close to I_r
        U = self._last_u_r
        r = U.size(-1)
        I = torch.eye(r, device=U.device, dtype=U.dtype).unsqueeze(0)
        gram = torch.matmul(U.transpose(-2, -1), U)
        return F.mse_loss(gram, I.expand_as(gram))
