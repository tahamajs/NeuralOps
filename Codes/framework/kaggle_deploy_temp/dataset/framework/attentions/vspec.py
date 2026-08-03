r"""
VSpec++ Attention: Variational Spectral Token-Adaptive Attention.

Combines SVD-initialized dynamic low-rank projection with input-dependent
variational spectral gating, learned per-example head importance scaling,
and top-k token routing for adaptive sparse attention computation.

Mathematical Mechanism:
    Given input $X \in \mathbb{R}^{B \times N \times D}$:
    1. **Top-$k$ Token Routing:**
       $$s_i = \sigma(\text{MLP}(x_i)), \quad \mathcal{I} = \text{top-}k(\{s_i\}_{i=1}^N)$$
       $$\widetilde{X} = \text{gather}(X, \mathcal{I}) \in \mathbb{R}^{B \times k \times D}$$
    2. **Variational Spectral Gating:**
       For each projection weight $W \approx U \text{diag}(\Sigma) V^T$ (truncated SVD, rank $r$):
       $$\mu, \log\sigma = \text{MLP}(\bar{X}), \quad g = \sigma(\mu + \epsilon \exp(\log\sigma)) \in (0,1)^r$$
       $$Z = \widetilde{X} V^T_{:r}, \quad Y = Z \odot g, \quad \text{proj}(\widetilde{X}) = Y U_{:, :r}^T$$
    3. **Head-Gated Scaled Dot-Product Attention:**
       $$Q, K, V = \text{proj}_Q(\widetilde{X}), \text{proj}_K(\widetilde{X}), \text{proj}_V(\widetilde{X})$$
       $$h_g = \sigma(\text{MLP}(\bar{X})) \in \mathbb{R}^{B \times H}$$
       $$A = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d}} \odot h_g\right) V$$

Computational Complexity:
    - Token routing: $\mathcal{O}(B \cdot N \cdot D)$.
    - SVD-gated projections: $\mathcal{O}(B \cdot k \cdot D \cdot r)$.
    - Attention: $\mathcal{O}(B \cdot H \cdot k^2 \cdot d)$.
    - Total: $\mathcal{O}(B \cdot (N D + k^2 H d + k D r))$ with $k \ll N$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class VariationalSpectralGate(nn.Module):
    r"""Input-dependent variational gate over truncated singular values.

    Args:
        hidden_dim: Input feature dimension $D$.
        rank: Number of singular values to gate ($r$).
    """

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, rank * 2)
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute variational gate using reparameterization trick.

        Args:
            x: Input sequence `[B, N, D]`.

        Returns:
            Tuple of (gate, mu, log_sigma) each `[B, r]`.
        """
        pooled = x.mean(dim=1)  # (B, D)
        stats = self.net(pooled)  # (B, rank*2)
        mu, log_sigma = stats.chunk(2, dim=-1)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(log_sigma)
        g = torch.sigmoid(z)  # (B, rank)
        return g, mu, log_sigma


class HeadGate(nn.Module):
    r"""Learned per-example head importance scalars.

    Args:
        hidden_dim: Input dimension $D$.
        n_heads: Number of attention heads $H$.
    """

    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, n_heads)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute per-head importance gates `[B, H]`."""
        pooled = x.mean(dim=1)
        return torch.sigmoid(self.net(pooled))


class TokenRouter(nn.Module):
    r"""Top-$k$ token selector based on learned saliency scores.

    Args:
        hidden_dim: Input dimension $D$.
        keep_ratio: Fraction of tokens to retain.
        min_tokens: Minimum number of tokens to retain.
    """

    def __init__(
        self,
        hidden_dim: int,
        keep_ratio: float = 0.5,
        min_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.keep_ratio: float = keep_ratio
        self.min_tokens: int = min_tokens

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Select top-$k$ tokens by learned saliency.

        Args:
            x: Input sequence `[B, N, D]`.
            valid_mask: Optional boolean mask `[B, N]`.

        Returns:
            Tuple of (routed_tokens `[B, k, D]`, indices `[B, k]`).
        """
        B, N, D = x.shape
        scores = torch.sigmoid(self.net(x).squeeze(-1))  # (B, N)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, -1e9)
        keep = max(self.min_tokens, int(N * self.keep_ratio))
        idx = scores.topk(keep, dim=1).indices
        routed = torch.stack([x[b, idx[b]] for b in range(B)])
        return routed, idx


class DynamicLowRankProjector(nn.Module):
    r"""SVD-initialized low-rank projector with dynamic variational gate scaling.

    Args:
        W: Source weight matrix `[d_out, d_in]`.
        rank: Truncation rank $r$.
    """

    def __init__(self, W: torch.Tensor, rank: int) -> None:
        super().__init__()
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        self.register_buffer("U", U[:, :rank])
        self.register_buffer("Vh", Vh[:rank, :])
        self.S: nn.Parameter = nn.Parameter(S[:rank].clone())
        self.rank: int = rank

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        r"""Project input via gated truncated SVD.

        Args:
            x: Input `[B, N, d_in]`.
            gate: Singular value scaling gates `[B, r]`.

        Returns:
            Projected output `[B, N, d_out]`.
        """
        S_eff = self.S.unsqueeze(0) * gate  # (B, rank)
        mid = torch.matmul(x, self.Vh.t())  # (B, N, rank)
        mid = mid * S_eff.unsqueeze(1)  # (B, N, rank)
        out = torch.matmul(mid, self.U.t())  # (B, N, d_out)
        return out


@ATTENTION_REGISTRY.register("vspec")
class VSpecAttention(BaseAttention):
    r"""VSpec++ dynamic variational spectral attention layer.

    Args:
        config: Transformer configuration with `hidden_size`, `num_attention_heads`, and `dropout`.
        rank: SVD truncation and spectral gating rank $r$.
        pretrained_attn: Pretrained attention layer for weight initialization (required).
    """

    def __init__(
        self,
        config: Any,
        rank: int = 32,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(config, rank)
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.hidden_size // self.num_heads

        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for VSpecAttention.")

        self.Pq: DynamicLowRankProjector = DynamicLowRankProjector(
            pretrained_attn.q_lin.weight.data, rank
        )
        self.Pk: DynamicLowRankProjector = DynamicLowRankProjector(
            pretrained_attn.k_lin.weight.data, rank
        )
        self.Pv: DynamicLowRankProjector = DynamicLowRankProjector(
            pretrained_attn.v_lin.weight.data, rank
        )
        self.Po: DynamicLowRankProjector = DynamicLowRankProjector(
            pretrained_attn.out_lin.weight.data, rank
        )

        self.sg_q: VariationalSpectralGate = VariationalSpectralGate(
            self.hidden_size, rank
        )
        self.sg_k: VariationalSpectralGate = VariationalSpectralGate(
            self.hidden_size, rank
        )
        self.sg_v: VariationalSpectralGate = VariationalSpectralGate(
            self.hidden_size, rank
        )

        self.head_gate: HeadGate = HeadGate(self.hidden_size, self.num_heads)
        self.router: TokenRouter = TokenRouter(self.hidden_size)
        dropout_p: float = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute VSpec++ variational spectral attention.

        Args:
            hidden_states: Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional mask `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional per-head gate.
            output_attentions: Return explicit attention weights if True.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of:
                - output: Attended representation `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax matrix or None.
        """
        B, N, D = hidden_states.shape

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid: Optional[torch.Tensor] = attention_mask.bool().to(
                    hidden_states.device
                )
            else:
                valid = attention_mask.squeeze(1).squeeze(1).bool()
        else:
            valid = None

        routed, idx = self.router(hidden_states, valid)  # (B, k, D)

        gq, _, _ = self.sg_q(hidden_states)
        gk, _, _ = self.sg_k(hidden_states)
        gv, _, _ = self.sg_v(hidden_states)

        Q = self.Pq(routed, gq)
        K = self.Pk(routed, gk)
        V = self.Pv(routed, gv)

        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        hg = self.head_gate(hidden_states).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim**0.5)
        scores = scores * hg

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, -1, D)

        out = self.Po(context, gq)

        # Reverse routing: scatter back to original positions
        output = hidden_states.clone()
        for b in range(B):
            output[b, idx[b]] = out[b]

        if output_attentions:
            return output, attn_weights
        return output, None


def inject_vspec(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> VSpecAttention:
    r"""Bridge helper injecting VSpecAttention into a transformer layer."""
    rank: int = getattr(comp_config, "vspec_rank", 32)
    new_attn = VSpecAttention(
        config=model_config, rank=rank, pretrained_attn=old_attn
    )
    layer.attention = new_attn
    return new_attn
