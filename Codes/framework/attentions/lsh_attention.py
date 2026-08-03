"""
Locality‑Sensitive Hashing (LSH) Attention – Reformer style.
Paper: https://arxiv.org/abs/2001.04451
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("lsh")
class LSHAttention(nn.Module):
    r"""Locality-Sensitive Hashing (LSH) Attention (Reformer).

    Replaces quadratic pairwise comparison with random projection angular hashing:
    $$h(x) = \arg\max \left( [x R; -x R] \right) \quad \text{or} \quad \text{sign}(x R)$$
    where $R \in \mathbb{R}^{d_k \times b}$ is a random Gaussian matrix. Tokens are sorted into discrete
    hash buckets $\mathcal{B}_k$, restricting softmax attention within bucket clusters across $n_r$ hash rounds:
    $$\text{Attention}_{\text{LSH}}(Q, K, V) = \frac{1}{n_r} \sum_{r=1}^{n_r} \text{Softmax}\left(\frac{Q_{\mathcal{B}_k}^{(r)} (K_{\mathcal{B}_k}^{(r)})^\top}{\sqrt{d_k}}\right) V_{\mathcal{B}_k}^{(r)}$$

    Computational Complexity:
        - Time: $\mathcal{O}(n_r \cdot B \cdot H \cdot N \log N \cdot d_k)$
        - Memory: $\mathcal{O}(n_r \cdot B \cdot H \cdot N)$ vs $\mathcal{O}(B \cdot H \cdot N^2)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any,
        rounds: int = 4,
        bucket_size: int = 64,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.rounds: int = rounds
        self.bucket_size: int = bucket_size

        self.q_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Copy weights and biases from pretrained multi-head attention module."""
        with torch.no_grad():
            self.q_proj.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.q_proj.bias.copy_(src.q_lin.bias)
            self.k_proj.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.k_proj.bias.copy_(src.k_lin.bias)
            self.v_proj.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.v_proj.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def _hash(self, vectors: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        """Angular LSH: sign of dot product with random hyperplanes."""
        norm: torch.Tensor = F.normalize(vectors, p=2, dim=-1)
        dots: torch.Tensor = torch.matmul(norm, proj)
        bits: torch.Tensor = (dots > 0).to(torch.long)
        multipliers: torch.Tensor = 2 ** torch.arange(
            self.bucket_size, device=vectors.device, dtype=torch.long
        )
        return (bits * multipliers).sum(dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for multi-round locality-sensitive hashing attention.

        Args:
            hidden_states (torch.Tensor): Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Mask tensor.
            head_mask (Optional[torch.Tensor]): Head mask.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output features and `None`.
        """
        B, L, D = hidden_states.shape
        H = self.num_heads
        d = self.head_dim

        Q: torch.Tensor = (
            self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        )
        K: torch.Tensor = (
            self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        )
        V: torch.Tensor = (
            self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        )

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = attention_mask.bool().to(Q.device)
            else:
                valid = attention_mask.squeeze(1).squeeze(1).bool()
        else:
            valid = None

        out: torch.Tensor = torch.zeros_like(V)
        scale: float = d**-0.5

        for r in range(self.rounds):
            proj: torch.Tensor = torch.randn(d, self.bucket_size, device=Q.device)
            proj = F.normalize(proj, p=2, dim=0)

            q_hash: torch.Tensor = self._hash(Q, proj)
            k_hash: torch.Tensor = self._hash(K, proj)

            for b in range(B):
                for h in range(H):
                    qh: torch.Tensor = q_hash[b, h]
                    kh: torch.Tensor = k_hash[b, h]
                    qh_u: torch.Tensor = qh.unique()
                    kh_u: torch.Tensor = kh.unique()
                    unique_codes: torch.Tensor = qh_u[torch.isin(qh_u, kh_u)]
                    if len(unique_codes) == 0:
                        continue
                    for code in unique_codes:
                        q_mask = qh == code
                        k_mask = kh == code
                        if valid is not None and not q_mask[valid[b]].any():
                            continue
                        Qs: torch.Tensor = Q[b, h][q_mask]
                        Ks: torch.Tensor = K[b, h][k_mask]
                        Vs: torch.Tensor = V[b, h][k_mask]
                        if valid is not None:
                            kv_valid = valid[b][k_mask].unsqueeze(-1).to(Ks.dtype)
                            Ks = Ks * kv_valid
                            Vs = Vs * kv_valid
                        scores: torch.Tensor = torch.matmul(Qs, Ks.T) * scale
                        attn: torch.Tensor = F.softmax(scores, dim=-1)
                        attn = self.dropout(attn)
                        ctx: torch.Tensor = torch.matmul(attn, Vs)
                        out[b, h][q_mask] += ctx

        out = out / self.rounds
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return out, None


def inject_lsh(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> LSHAttention:
    """Injection helper for Reformer LSH attention."""
    rounds: int = getattr(comp_config, "rounds", 4)
    bucket_size: int = getattr(comp_config, "bucket_size", 64)
    new_attn: LSHAttention = LSHAttention(
        config=model_config,
        rounds=rounds,
        bucket_size=bucket_size,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
