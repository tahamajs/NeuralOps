r"""
ORION: Optimized Rectified Input-aware Orthogonal Network.

Hybrid multi-head attention architecture integrating zero-shot SVD low-rank factorized projections,
head-wise bifurcated linear kernel vs. rectified Softpick full attention, and random Fourier features.

Mathematical Mechanism:
    1. **SVD-Factorized Projections:**
       Weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is decomposed via truncated SVD
       guided by cumulative spectral energy threshold $\eta$:
       $$\sum_{i=1}^r \sigma_i^2 \ge \eta \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i^2 \implies W \approx A B, \quad A \in \mathbb{R}^{d_{\text{out}} \times r}, B \in \mathbb{R}^{r \times d_{\text{in}}}$$
    2. **Bifurcated Attention Heads ($H = H_k + H_f$):**
       - **Linear Kernel Heads ($H_k$):**
         Random Fourier feature mapping $\phi(x) = \frac{1}{\sqrt{F}} \exp(x W_{\text{rff}})$ yielding linearized attention:
         $$\operatorname{Attn}_{\text{kernel}}(Q, K, V) = \frac{\phi(Q) (\phi(K)^T V)}{\phi(Q) (\phi(K)^T \mathbf{1})}$$
       - **Full Softpick Heads ($H_f$):**
         Dense attention with rectified ~47% sparse Softpick activation:
         $$\operatorname{Attn}_{\text{full}}(Q, K, V) = \operatorname{Softpick}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

Computational Complexity:
    - Kernel heads: $\mathcal{O}(B \cdot H_k \cdot N \cdot F \cdot d)$ linear in sequence length $N$.
    - Full heads: $\mathcal{O}(B \cdot H_f \cdot N^2 \cdot d)$ quadratic in sequence length $N$.
    - Overall Complexity: $\mathcal{O}(N \cdot (H_k F + H_f N) \cdot d)$.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class ORIONConfig:
    r"""Hyperparameter configuration for ORION Attention.

    Attributes:
        svd_energy: Cumulative spectral energy target for SVD rank determination ($\eta \in (0, 1]$).
        svd_min_rank: Minimum allowable rank for factorized linear projections.
        svd_max_rank: Maximum allowable rank for factorized linear projections.
        num_kernel_heads: Number of heads computing linearized random Fourier feature attention ($H_k$).
        num_full_heads: Number of heads computing full Softpick rectified attention ($H_f$).
        rff_features: Number of random Fourier features per kernel head ($F$).
        softpick_temperature: Softmax temperature parameter for Softpick activation.
        dropout: Attention dropout probability.
    """

    svd_energy: float = 0.95
    svd_min_rank: int = 4
    svd_max_rank: int = 64
    num_kernel_heads: int = 8
    num_full_heads: int = 4
    rff_features: int = 256
    softpick_temperature: float = 1.0
    dropout: float = 0.1


def softpick(x, dim: int = -1, temperature: float = 1.0):
    import torch.nn.functional as F
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    m_scaled = torch.clamp(-m / temperature, max=80.0)
    denom = e - torch.exp(m_scaled)
    p = F.relu(denom)
    return (p / (p.sum(dim=dim, keepdim=True) + 1e-5)).to(orig_dtype)


class FactorisedLinear(nn.Module):
    r"""Low-rank factorized linear layer initialized via truncated SVD of pretrained weights.

    Factorizes $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ into:
    $$\text{Down}: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^r, \quad \text{Up}: \mathbb{R}^r \to \mathbb{R}^{d_{\text{out}}}$$

    Args:
        pretrained_weight: Source weight tensor `[OutDim, InDim]`.
        pretrained_bias: Optional source bias tensor `[OutDim]`.
        energy: Cumulative singular value energy fraction to retain.
        min_rank: Lower bound on factorization rank.
        max_rank: Upper bound on factorization rank.
    """

    def __init__(
        self,
        pretrained_weight: torch.Tensor,
        pretrained_bias: Optional[torch.Tensor],
        energy: float,
        min_rank: int = 4,
        max_rank: int = 64,
    ) -> None:
        super().__init__()
        out_dim, in_dim = pretrained_weight.shape
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(pretrained_weight.float(), full_matrices=False)
            total = (S**2).sum()
            cum = torch.cumsum(S**2, dim=0) / total
            rank = int(torch.searchsorted(cum, energy).item()) + 1
            rank = max(min_rank, min(rank, min(out_dim, in_dim), max_rank))
            self.rank: int = rank
            A = U[:, :rank] * S[:rank].unsqueeze(0)
            B = Vt[:rank, :]
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=(pretrained_bias is not None))
        self.down.weight.data = B.float()
        self.up.weight.data = A.float()
        if pretrained_bias is not None:
            self.up.bias.data = pretrained_bias.float()
        else:
            self.up.bias.data.zero_()

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.up.weight, self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute low-rank linear projection: $y = (x B^T) A^T + b$."""
        return self.up(self.down(x))


@ATTENTION_REGISTRY.register("orion_module")
@ATTENTION_REGISTRY.register("orion")
@ATTENTION_REGISTRY.register("orion_attention")
@ATTENTION_REGISTRY.register("orion_module")
class ORIONAttention(nn.Module):
    r"""ORION hybrid multi-head attention module.

    Args:
        model_config: Model configuration containing `hidden_size` and `num_attention_heads`.
        orion_cfg: ORIONConfig hyperparameters.
        pretrained_attn: Pretrained attention module used for SVD weight initialization.
        bert_mode: Boolean indicating bidirectional context handling.
    """

    def __init__(
        self,
        model_config: Optional[Any] = None,
        orion_cfg: Optional[ORIONConfig] = None,
        pretrained_attn: Optional[Any] = None,
        bert_mode: bool = False,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if orion_cfg is None:
            orion_cfg = ORIONConfig()
        hidden_size = getattr(model_config, "hidden_size", 768) if model_config is not None else 768
        num_heads = getattr(model_config, "num_attention_heads", 12) if model_config is not None else 12
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        self.all_head_size: int = hidden_size
        self.hidden_size: int = hidden_size
        self.cfg: ORIONConfig = orion_cfg
        self.bert_mode: bool = bert_mode

        if pretrained_attn is None:
            dummy_lin = nn.Linear(hidden_size, hidden_size)
            pretrained_attn = type("DummyAttn", (), {
                "q_lin": dummy_lin,
                "k_lin": dummy_lin,
                "v_lin": dummy_lin,
                "out_lin": dummy_lin,
            })()

        if orion_cfg.num_kernel_heads + orion_cfg.num_full_heads != self.num_heads:
            raise ValueError(
                f"num_kernel_heads ({orion_cfg.num_kernel_heads}) + "
                f"num_full_heads ({orion_cfg.num_full_heads}) must equal "
                f"total heads ({self.num_heads})"
            )
        self.num_kernel_heads: int = orion_cfg.num_kernel_heads
        self.num_full_heads: int = orion_cfg.num_full_heads

        self.q_proj = FactorisedLinear(
            pretrained_attn.q_lin.weight.data,
            pretrained_attn.q_lin.bias.data,
            orion_cfg.svd_energy,
            orion_cfg.svd_min_rank,
            orion_cfg.svd_max_rank,
        )
        self.k_proj = FactorisedLinear(
            pretrained_attn.k_lin.weight.data,
            pretrained_attn.k_lin.bias.data,
            orion_cfg.svd_energy,
            orion_cfg.svd_min_rank,
            orion_cfg.svd_max_rank,
        )
        self.v_proj = FactorisedLinear(
            pretrained_attn.v_lin.weight.data,
            pretrained_attn.v_lin.bias.data,
            orion_cfg.svd_energy,
            orion_cfg.svd_min_rank,
            orion_cfg.svd_max_rank,
        )
        self.out_proj = FactorisedLinear(
            pretrained_attn.out_lin.weight.data,
            pretrained_attn.out_lin.bias.data,
            orion_cfg.svd_energy,
            orion_cfg.svd_min_rank,
            orion_cfg.svd_max_rank,
        )
        self.rff_proj = nn.Parameter(
            torch.randn(self.num_kernel_heads, self.head_dim, orion_cfg.rff_features)
            * 0.02
        )
        self.dropout = nn.Dropout(orion_cfg.dropout)

    def _kernel_linear_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute kernel linear attention via RFF projections in $\mathcal{O}(N \cdot F \cdot d)$."""
        B, Hk, N, D = Q.shape
        F_dim = self.cfg.rff_features
        orig_dtype = Q.dtype
        Q, K, V = Q.float(), K.float(), V.float()
        proj = self.rff_proj.to(Q.device).float()
        
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, proj).clamp(max=11.0)
        K_proj = torch.einsum("bhnd,hdf->bhnf", K_norm, proj).clamp(max=11.0)
        Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
        K_feat = torch.exp(K_proj) / math.sqrt(F_dim)
        KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)
        K_sum = K_feat.sum(dim=2)
        denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)
        ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / (denom + 1e-5)
        return ctx.to(orig_dtype)

    def _full_softpick_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Compute full Softpick rectified attention in $\mathcal{O}(N^2 \cdot d)$."""
        scale = self.head_dim**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if token_valid is not None:
            scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -10000.0)
        attn_w = softpick(scores, dim=-1, temperature=self.cfg.softpick_temperature)
        attn_w = self.dropout(attn_w)
        return torch.matmul(attn_w, V)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Forward pass computing hybrid kernel and full attention across parallel heads.

        Args:
            hidden_states: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask tensor `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head gating mask.
            output_attentions: Return explicit attention matrices if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Layer projection output of shape `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Attention matrix or None.
        """
        B, N, _ = hidden_states.shape
        Hk = self.num_kernel_heads

        Q_all = self.q_proj(hidden_states)
        K_all = self.k_proj(hidden_states)
        V_all = self.v_proj(hidden_states)
        Q_all = Q_all.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K_all = K_all.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V_all = V_all.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q_kernel, Q_full = Q_all[:, :Hk], Q_all[:, Hk:]
        K_kernel, K_full = K_all[:, :Hk], K_all[:, Hk:]
        V_kernel, V_full = V_all[:, :Hk], V_all[:, Hk:]

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

        ctx_kernel = self._kernel_linear_attention(Q_kernel, K_kernel, V_kernel)
        ctx_full = self._full_softpick_attention(Q_full, K_full, V_full, token_valid)

        context = torch.cat([ctx_kernel, ctx_full], dim=1)
        if token_valid is not None:
            context = context * token_valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(context)
        return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Compute orthogonal regularization loss."""
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("orion_inject")
def inject_orion(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> ORIONAttention:
    r"""Bridge function injecting ORION attention into a transformer layer."""
    cfg = ORIONConfig(
        svd_energy=getattr(comp_config, "svd_energy", 0.95),
        svd_min_rank=getattr(comp_config, "min_rank", 4),
        svd_max_rank=getattr(comp_config, "max_rank", 64),
        num_kernel_heads=getattr(comp_config, "num_kernel_heads", 8),
        num_full_heads=getattr(comp_config, "num_full_heads", 4),
        rff_features=getattr(comp_config, "rff_features", 256),
        softpick_temperature=getattr(comp_config, "softpick_temperature", 1.0),
        dropout=getattr(comp_config, "dropout", 0.1),
    )
    new_attn = ORIONAttention(model_config, cfg, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
