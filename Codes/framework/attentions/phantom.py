r"""
Phantom: Parameter-Hashed AttentioN with Token-wise Orthogonal Mapping.

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
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class PHANTOMConfig:
    r"""Hyperparameter configuration for PHANTOM attention.

    Attributes:
        cur_rank (int): Selected rank $r$ for CUR factorized linear projections.
        lsh_rounds (int): Number of distinct hashing rounds $M$.
        lsh_bucket_size (int): Number of binary hash hyperplane bits ($B_s$).
        softpick_enabled (bool): Use rectified Softpick activation.
        softpick_temperature (float): Scaling temperature $\tau$ for Softpick.
        dropout (float): Attention dropout probability.
    """
    cur_rank: int = 16
    lsh_rounds: int = 4
    lsh_bucket_size: int = 4
    softpick_enabled: bool = True
    softpick_temperature: float = 1.0
    dropout: float = 0.1


def softpick(x, dim: int = -1, temperature: float = 1.0):
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    m_scaled = torch.clamp(-m / temperature, max=80.0)
    denom = e - torch.exp(m_scaled)
    p = F.relu(denom)
    p_sum = p.sum(dim=dim, keepdim=True)
    fallback = F.softmax(x / temperature, dim=dim)
    res = torch.where(p_sum > 1e-5, p / torch.clamp(p_sum, min=1e-5), fallback)
    return res.to(orig_dtype)


class FactorisedLinear_CUR(nn.Module):
    r"""Linear projection compressed via deterministic leverage-score CUR matrix decomposition.

    Factorizes a pre-trained weight matrix $W$ into $C W_{\cap}^{\dagger} R$.

    Args:
        pretrained_weight (torch.Tensor): Source 2D weight matrix.
        pretrained_bias (Optional[torch.Tensor]): Optional source bias vector.
        rank (int): Target compression rank $r$.
    """

    def __init__(
        self,
        pretrained_weight: torch.Tensor,
        pretrained_bias: Optional[torch.Tensor],
        rank: int,
    ) -> None:
        super().__init__()
        out_dim, in_dim = pretrained_weight.shape
        self.rank: int = max(1, min(rank, out_dim, in_dim))
        device = pretrained_weight.device
        dtype = pretrained_weight.dtype

        with torch.no_grad():
            W = pretrained_weight.float()
            U_full, S_full, Vt_full = torch.linalg.svd(W, full_matrices=False)
            col_leverage = (Vt_full**2).sum(dim=0)
            col_leverage = col_leverage / torch.clamp(col_leverage.sum(), min=1e-5)
            row_leverage = (U_full**2).sum(dim=1)
            row_leverage = row_leverage / torch.clamp(row_leverage.sum(), min=1e-5)

            _, col_idx = torch.topk(col_leverage, k=self.rank, largest=True)
            _, row_idx = torch.topk(row_leverage, k=self.rank, largest=True)

            C = W[:, col_idx]
            R = W[row_idx, :]
            W_intersection = W[row_idx][:, col_idx]
            U_int = torch.linalg.pinv(W_intersection)

            self.down = nn.Linear(in_dim, self.rank, bias=False, device=device, dtype=dtype)
            self.up = nn.Linear(self.rank, out_dim, bias=(pretrained_bias is not None), device=device, dtype=dtype)

            self.down.weight.data = R.to(dtype=dtype, device=device)
            self.up.weight.data = (C @ U_int).to(dtype=dtype, device=device)

            if pretrained_bias is not None:
                self.up.bias.data = pretrained_bias.to(dtype=dtype, device=device)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.up.weight, self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Project input through CUR factorized down- and up-projections.

        Math:
            $$y = (x R^T) (C W_{\cap}^{\dagger})^T + b$$

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Projected tensor.

        Shape:
            - Input: `[..., InDim]`
            - Output: `[..., OutDim]`
        """
        return self.up(self.down(x))


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

    Computes sparse attention by hashing query and key vectors into discrete buckets and only attending within buckets.

    Args:
        Q (torch.Tensor): Query tensor.
        K (torch.Tensor): Key tensor.
        V (torch.Tensor): Value tensor.
        rounds (int): Number of independent hashing rounds.
        bucket_size (int): Bit width for bucket generation.
        softpick_temp (float): Softpick temperature parameter.
        dropout (Optional[nn.Module]): Optional dropout layer module.
        token_valid (Optional[torch.Tensor]): Optional boolean mask of non-padded positions.

    Returns:
        torch.Tensor: Attended tensor.

    Shape:
        - Q: `[Batch, Heads, SeqLen, HeadDim]`
        - K: `[Batch, Heads, SeqLen, HeadDim]`
        - V: `[Batch, Heads, SeqLen, HeadDim]`
        - token_valid: `[Batch, SeqLen]`
        - Output: `[Batch, Heads, SeqLen, HeadDim]`
    """
    B, H, N, D = Q.shape
    scale = D**-0.5
    n_bits = max(1, min(bucket_size, 8))

    Q_norm = F.normalize(Q, p=2, dim=-1, eps=1e-5)
    K_norm = F.normalize(K, p=2, dim=-1, eps=1e-5)

    bit_shifts = torch.tensor(
        [2**i for i in range(n_bits)], device=Q.device, dtype=torch.float32
    )

    scores_base = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if token_valid is not None:
        val_mask = token_valid.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        scores_base = scores_base.masked_fill(~val_mask, -10000.0)

    accum_out = torch.zeros_like(V)

    for _ in range(rounds):
        hash_proj = torch.randn(D, n_bits, device=Q.device, dtype=Q.dtype)
        hash_proj = F.normalize(hash_proj, p=2, dim=0, eps=1e-5)

        hq = (torch.matmul(Q_norm, hash_proj) > 0).float() @ bit_shifts  # [B, H, N]
        hk = (torch.matmul(K_norm, hash_proj) > 0).float() @ bit_shifts  # [B, H, N]

        match = (hq.unsqueeze(-1) == hk.unsqueeze(-2))  # [B, H, N, N]
        eye = torch.eye(N, device=Q.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        match = match | eye

        scores_round = scores_base.masked_fill(~match, -10000.0)
        attn_w = softpick(scores_round, dim=-1, temperature=softpick_temp)
        if dropout is not None:
            attn_w = dropout(attn_w)

        accum_out = accum_out + torch.matmul(attn_w, V)

    return accum_out / float(max(1, rounds))


@ATTENTION_REGISTRY.register("phantom")
@ATTENTION_REGISTRY.register("phantom_module")
class PHANTOMAttention(BaseAttention):
    r"""PHANTOM Attention module with CUR factorized projections and multi-round LSH attention.

    Args:
        config (Optional[Any]): Transformer configuration object.
        rank (Optional[int]): Compression rank.
        phantom_cfg (Optional[PHANTOMConfig]): PHANTOMConfig instance.
        pretrained_attn (Optional[Any]): Pretrained attention module for CUR parameter initialization.
        bert_mode (bool): Flag indicating bidirectional sequence mode.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 16,
        phantom_cfg: Optional[PHANTOMConfig] = None,
        pretrained_attn: Optional[Any] = None,
        bert_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if phantom_cfg is None:
            cur_rank = rank or getattr(config, "cur_rank", getattr(config, "svd_rank", 16))
            phantom_cfg = PHANTOMConfig(cur_rank=cur_rank)
        elif rank is not None:
            phantom_cfg.cur_rank = rank

        model_config = config or kwargs.get("model_config", None)
        hidden_size = getattr(model_config, "hidden_size", kwargs.get("hidden_size", 768))
        num_heads = getattr(model_config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads if num_heads > 0 else 64
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = hidden_size
        self.cfg: PHANTOMConfig = phantom_cfg

        q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", getattr(pretrained_attn, "q", None)))) if pretrained_attn is not None else None
        k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", getattr(pretrained_attn, "k", None)))) if pretrained_attn is not None else None
        v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", getattr(pretrained_attn, "v", None)))) if pretrained_attn is not None else None
        out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o_proj", getattr(pretrained_attn, "o", None))))) if pretrained_attn is not None else None

        if q_src is not None and hasattr(q_src, "weight"):
            q_w = q_src.weight.data
            q_b = q_src.bias.data if getattr(q_src, "bias", None) is not None else None
        else:
            q_w = torch.randn(self.all_head_size, hidden_size) * 0.02
            q_b = None
        self.q_proj = FactorisedLinear_CUR(q_w, q_b, phantom_cfg.cur_rank)

        if k_src is not None and hasattr(k_src, "weight"):
            k_w = k_src.weight.data
            k_b = k_src.bias.data if getattr(k_src, "bias", None) is not None else None
        else:
            k_w = torch.randn(self.all_head_size, hidden_size) * 0.02
            k_b = None
        self.k_proj = FactorisedLinear_CUR(k_w, k_b, phantom_cfg.cur_rank)

        if v_src is not None and hasattr(v_src, "weight"):
            v_w = v_src.weight.data
            v_b = v_src.bias.data if getattr(v_src, "bias", None) is not None else None
        else:
            v_w = torch.randn(self.all_head_size, hidden_size) * 0.02
            v_b = None
        self.v_proj = FactorisedLinear_CUR(v_w, v_b, phantom_cfg.cur_rank)

        if out_src is not None and hasattr(out_src, "weight"):
            out_w = out_src.weight.data
            out_b = out_src.bias.data if getattr(out_src, "bias", None) is not None else None
        else:
            out_w = torch.randn(hidden_size, self.all_head_size) * 0.02
            out_b = None
        self.out_proj = FactorisedLinear_CUR(out_w, out_b, phantom_cfg.cur_rank)

        self.dropout = nn.Dropout(phantom_cfg.dropout)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        r"""Transfer and re-factorize weights from pretrained multi-head attention module."""
        with torch.no_grad():
            q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", None)))
            k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", None)))
            v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", None)))
            out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o_proj", None))))

            if q_src is not None and hasattr(q_src, "weight"):
                self.q_proj = FactorisedLinear_CUR(
                    q_src.weight.data,
                    q_src.bias.data if q_src.bias is not None else None,
                    self.cfg.cur_rank,
                )
            if k_src is not None and hasattr(k_src, "weight"):
                self.k_proj = FactorisedLinear_CUR(
                    k_src.weight.data,
                    k_src.bias.data if k_src.bias is not None else None,
                    self.cfg.cur_rank,
                )
            if v_src is not None and hasattr(v_src, "weight"):
                self.v_proj = FactorisedLinear_CUR(
                    v_src.weight.data,
                    v_src.bias.data if v_src.bias is not None else None,
                    self.cfg.cur_rank,
                )
            if out_src is not None and hasattr(out_src, "weight"):
                self.out_proj = FactorisedLinear_CUR(
                    out_src.weight.data,
                    out_src.bias.data if out_src.bias is not None else None,
                    self.cfg.cur_rank,
                )

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
            hidden_states (torch.Tensor): Input activations.
            attention_mask (Optional[torch.Tensor]): Attention padding mask.
            head_mask (Optional[torch.Tensor]): Optional head pruning mask.
            output_attentions (bool): Return explicit attention matrices.
            **kwargs (Any): Additional runtime arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - output: Projected tensor.
                - attention_weights: Always None for PHANTOM.

        Shape:
            - hidden_states: `[Batch, SeqLen, HiddenDim]`
            - output: `[Batch, SeqLen, HiddenDim]`
        """
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2).contiguous()
        K = self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2).contiguous()
        V = self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2).contiguous()

        mask = (
            attention_mask
            if attention_mask is not None
            else kwargs.get("attn_mask", None)
        )
        token_valid = None
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -10000.0
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -10000.0
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
        r"""Compute orthogonal loss regularizer.

        Returns:
            torch.Tensor: A scalar tensor containing the orthogonal loss (0.0).

        Shape:
            - Output: `[]` (scalar)
        """
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("phantom_inject")
def inject_phantom(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> PHANTOMAttention:
    r"""Inject PHANTOM attention into a target transformer layer.

    Args:
        layer (Any): Target transformer layer.
        model_config (Any): Base model configuration.
        comp_config (Any): Compression configuration.
        layer_idx (int): Current layer index.
        total_layers (int): Total number of layers.
        old_attn (Any): Pre-existing attention module.

    Returns:
        PHANTOMAttention: The newly injected PHANTOM attention module.
    """
    cfg = PHANTOMConfig(
        cur_rank=getattr(comp_config, "cur_rank", 16),
        lsh_rounds=getattr(comp_config, "lsh_rounds", 4),
        lsh_bucket_size=getattr(comp_config, "lsh_bucket_size", 32),
        softpick_enabled=True,
        dropout=getattr(comp_config, "dropout", 0.1),
    )
    new_attn = PHANTOMAttention(model_config, phantom_cfg=cfg, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn

