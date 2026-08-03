r"""
PHANTOM: Parameter-Hashed AttentioN with Token-wise Orthogonal Mapping.

Combines CUR-matrix factorized linear projections based on empirical statistical leverage scores
with multi-round angular Locality-Sensitive Hashing (LSH) sparse attention and rectified Softpick activation.

Mathematical Mechanism:
    1. **CUR Matrix Factorization:**
       For pretrained weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, statistical leverage scores are:
       $$\ell_{\text{row}}(i) = \frac{\|U_{i,:}\|_2^2}{\sum_{i'} \|U_{i',:}\|_2^2}, \quad \ell_{\text{col}}(j) = \frac{\|V_{:,j}\|_2^2}{\sum_{j'} \|V_{:,j'}\|_2^2}$$
       Top-$r$ rows $R \in \mathbb{R}^{r \times d_{\text{in}}}$ and columns $C \in \mathbb{R}^{d_{\text{out}} \times r}$ form intersection $W_{\cap} \in \mathbb{R}^{r \times r}$:
       $$W \approx C W_{\cap}^{\dagger} R \implies \text{Down}(x) = x R^T, \quad \text{Up}(z) = z (C W_{\cap}^{\dagger})^T + b$$
    2. **Angular LSH Attention:**
       For each hashing round $m \in \{1, \dots, M\}$ with random Gaussian hyperplanes $P \in \mathbb{R}^{d \times B_s}$:
       $$h(q) = \sum_{k=0}^{B_s-1} 2^k \cdot \mathbb{I}(\langle \bar{q}, P_{:,k} \rangle > 0)$$
       Tokens are partitioned into discrete angular hash buckets, restricting attention computation:
       $$A_{ij}^{(m)} = \operatorname{Softpick}\left(\frac{q_i k_j^T}{\sqrt{d}}\right) \cdot \mathbb{I}(h(q_i) = h(k_j))$$

Computational Complexity:
    - Projection complexity per token: $\mathcal{O}(d_{\text{in}} \cdot r + r \cdot d_{\text{out}}) \ll \mathcal{O}(d_{\text{in}} \cdot d_{\text{out}})$.
    - Attention complexity: $\mathcal{O}(M \cdot B \cdot H \cdot \frac{N^2}{2^{B_s}} \cdot d)$ with expected bucket size reduction.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class PHANTOMConfig:
    r"""Hyperparameter configuration for PHANTOM attention.

    Attributes:
        cur_rank: Selected rank $r$ for CUR factorized linear projections.
        lsh_rounds: Number of distinct hashing rounds $M$.
        lsh_bucket_size: Number of binary hash hyperplane bits ($B_s$).
        softpick_enabled: Use rectified Softpick activation.
        softpick_temperature: Scaling temperature $\tau$ for Softpick.
        dropout: Attention dropout probability.
    """

    cur_rank: int = 16
    lsh_rounds: int = 4
    lsh_bucket_size: int = 32
    softpick_enabled: bool = True
    softpick_temperature: float = 1.0
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Softpick activation (sparse-friendly rectified softmax)
# ---------------------------------------------------------------------------
def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    r"""Compute rectified softmax avoiding artificial sink tokens.

    Args:
        x: Input logit tensor.
        dim: Normalization dimension.
        temperature: Temperature factor $\tau$.

    Returns:
        Sparse normalized probability tensor.
    """
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)


# ---------------------------------------------------------------------------
# CUR-based factorised linear layer
# ---------------------------------------------------------------------------
class FactorisedLinear_CUR(nn.Module):
    r"""Linear projection compressed via deterministic leverage-score CUR matrix decomposition.

    Args:
        pretrained_weight: Source 2D weight matrix `[OutDim, InDim]`.
        pretrained_bias: Optional source bias vector `[OutDim]`.
        rank: Target compression rank $r$.
    """

    def __init__(
        self,
        pretrained_weight: torch.Tensor,
        pretrained_bias: Optional[torch.Tensor],
        rank: int,
    ) -> None:
        super().__init__()
        out_dim, in_dim = pretrained_weight.shape
        self.rank: int = rank

        with torch.no_grad():
            W = pretrained_weight.float()
            U_full, S_full, Vt_full = torch.linalg.svd(W, full_matrices=False)
            col_leverage = (Vt_full**2).sum(dim=0)
            col_leverage /= col_leverage.sum()
            row_leverage = (U_full**2).sum(dim=1)
            row_leverage /= row_leverage.sum()

            _, col_idx = torch.topk(col_leverage, k=rank, largest=True)
            _, row_idx = torch.topk(row_leverage, k=rank, largest=True)

            C = W[:, col_idx]
            R = W[row_idx, :]
            W_intersection = W[row_idx][:, col_idx]
            U_int = torch.linalg.pinv(W_intersection.float())

            self.down = nn.Linear(in_dim, rank, bias=False)
            self.up = nn.Linear(rank, out_dim, bias=(pretrained_bias is not None))

            self.down.weight.data = R.float()
            self.up.weight.data = (C @ U_int).float()

            if pretrained_bias is not None:
                self.up.bias.data = pretrained_bias.float()
            else:
                self.up.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Project input through CUR factorized down- and up-projections."""
        return self.up(self.down(x))


# ---------------------------------------------------------------------------
# LSH attention (angular hash)
# ---------------------------------------------------------------------------
def lsh_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rounds: int,
    bucket_size: int,
    softpick_temp: float = 1.0,
    dropout: Optional[nn.Module] = None,
    token_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Execute multi-round angular Locality-Sensitive Hashing attention.

    Args:
        Q: Query tensor `[Batch, Heads, SeqLen, HeadDim]`.
        K: Key tensor `[Batch, Heads, SeqLen, HeadDim]`.
        V: Value tensor `[Batch, Heads, SeqLen, HeadDim]`.
        rounds: Number of independent hashing rounds.
        bucket_size: Bit width for bucket generation.
        softpick_temp: Softpick temperature parameter.
        dropout: Optional dropout layer module.
        token_valid: Optional boolean mask `[Batch, SeqLen]` of non-padded positions.

    Returns:
        Attended tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.
    """
    B, H, N, D = Q.shape
    scale = D**-0.5

    Q_norm = F.normalize(Q, p=2, dim=-1)
    K_norm = F.normalize(K, p=2, dim=-1)

    out = torch.zeros_like(V)
    bit_shifts = torch.tensor(
        [2**i for i in range(bucket_size)], device=Q.device
    ).view(1, 1, 1, -1)

    for _ in range(rounds):
        hash_proj = torch.randn(D, bucket_size, device=Q.device, dtype=Q.dtype)
        hash_proj = F.normalize(hash_proj, p=2, dim=0)

        hash_code_q = (
            (torch.matmul(Q_norm, hash_proj) > 0).int().mul(bit_shifts).sum(dim=-1)
        )
        hash_code_k = (
            (torch.matmul(K_norm, hash_proj) > 0).int().mul(bit_shifts).sum(dim=-1)
        )

        for b in range(B):
            for h in range(H):
                q_h = Q[b, h]
                k_h = K[b, h]
                v_h = V[b, h]
                if token_valid is not None:
                    valid = token_valid[b]
                    q_h = q_h[valid]
                    k_h = k_h[valid]
                    v_h = v_h[valid]
                    hq = hash_code_q[b, h][valid]
                    hk = hash_code_k[b, h][valid]
                    if q_h.size(0) == 0:
                        continue
                else:
                    valid = None
                    hq = hash_code_q[b, h]
                    hk = hash_code_k[b, h]

                sorted_q_idx = torch.argsort(hq)
                q_sorted = q_h[sorted_q_idx]

                buckets: Dict[int, List[int]] = {}
                for i, code in enumerate(hk):
                    buckets.setdefault(int(code.item()), []).append(i)

                attn_ctx = torch.zeros_like(q_sorted)
                for i, q_i in enumerate(q_sorted):
                    code_val = int(hq[sorted_q_idx[i]].item())
                    key_indices = buckets.get(code_val, [])
                    if not key_indices:
                        continue
                    k_bucket = k_h[key_indices]
                    v_bucket = v_h[key_indices]

                    scores = (q_i * scale) @ k_bucket.T
                    attn_w = softpick(
                        scores.unsqueeze(0), dim=-1, temperature=softpick_temp
                    ).squeeze(0)
                    if dropout is not None:
                        attn_w = dropout(attn_w)
                    attn_ctx[i] = attn_w @ v_bucket

                unsorted_idx = torch.argsort(sorted_q_idx)
                attn_unsorted = attn_ctx[unsorted_idx]
                if valid is not None:
                    out[b, h][valid] = attn_unsorted
                else:
                    out[b, h] = attn_unsorted

    return out / float(max(1, rounds))


# ---------------------------------------------------------------------------
# Main PHANTOM Attention module
# ---------------------------------------------------------------------------
@ATTENTION_REGISTRY.register("phantom_module")
class PHANTOMAttention(nn.Module):
    r"""PHANTOM Attention module with CUR factorized projections and multi-round LSH attention.

    Args:
        model_config: Transformer configuration object.
        phantom_cfg: PHANTOMConfig instance.
        pretrained_attn: Pretrained attention module for CUR parameter initialization.
        bert_mode: Flag indicating bidirectional sequence mode.
    """

    def __init__(
        self,
        model_config: Any,
        phantom_cfg: PHANTOMConfig,
        pretrained_attn: Optional[Any] = None,
        bert_mode: bool = False,
    ) -> None:
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("PHANTOM requires pretrained_attn for CUR init")
        self.num_heads: int = model_config.num_attention_heads
        self.head_dim: int = model_config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = model_config.hidden_size
        self.cfg: PHANTOMConfig = phantom_cfg

        self.q_proj = FactorisedLinear_CUR(
            pretrained_attn.q_lin.weight.data,
            pretrained_attn.q_lin.bias.data,
            phantom_cfg.cur_rank,
        )
        self.k_proj = FactorisedLinear_CUR(
            pretrained_attn.k_lin.weight.data,
            pretrained_attn.k_lin.bias.data,
            phantom_cfg.cur_rank,
        )
        self.v_proj = FactorisedLinear_CUR(
            pretrained_attn.v_lin.weight.data,
            pretrained_attn.v_lin.bias.data,
            phantom_cfg.cur_rank,
        )
        self.out_proj = FactorisedLinear_CUR(
            pretrained_attn.out_lin.weight.data,
            pretrained_attn.out_lin.bias.data,
            phantom_cfg.cur_rank,
        )

        self.dropout = nn.Dropout(phantom_cfg.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute PHANTOM attention.

        Args:
            hidden_states: Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Attention padding mask.
            head_mask: Optional head pruning mask.
            output_attentions: Return explicit attention matrices.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: None.
        """
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)

        mask = (
            attention_mask
            if attention_mask is not None
            else kwargs.get("attn_mask", None)
        )
        token_valid = None
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -1e9
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -1e9
            elif mask.dim() == 3:
                token_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                token_valid = mask.bool()

        context = lsh_attention(
            Q,
            K,
            V,
            rounds=self.cfg.lsh_rounds,
            bucket_size=self.cfg.lsh_bucket_size,
            softpick_temp=(
                self.cfg.softpick_temperature if self.cfg.softpick_enabled else 1.0
            ),
            dropout=self.dropout if self.cfg.softpick_enabled else None,
            token_valid=token_valid,
        )

        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        output = self.out_proj(context)
        return (output, None)

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Compute orthogonal loss regularizer."""
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


# ------------------------------------------------------------
# Framework injection & registration
# ------------------------------------------------------------
@ATTENTION_REGISTRY.register("phantom")
def inject_phantom(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> PHANTOMAttention:
    r"""Inject PHANTOM attention into a target transformer layer."""
    cfg = PHANTOMConfig(
        cur_rank=getattr(comp_config, "cur_rank", 16),
        lsh_rounds=getattr(comp_config, "lsh_rounds", 4),
        lsh_bucket_size=getattr(comp_config, "lsh_bucket_size", 32),
        softpick_enabled=True,
        dropout=getattr(comp_config, "dropout", 0.1),
    )
    new_attn = PHANTOMAttention(model_config, cfg, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
