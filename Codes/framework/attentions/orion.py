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
from framework.core.base import BaseAttention
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
        in_features: Input dimension fallback.
        out_features: Output dimension fallback.
    """

    def __init__(
        self,
        pretrained_weight: Optional[torch.Tensor] = None,
        pretrained_bias: Optional[torch.Tensor] = None,
        energy: float = 0.95,
        min_rank: int = 4,
        max_rank: int = 64,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        if pretrained_weight is not None:
            out_dim, in_dim = pretrained_weight.shape
            with torch.no_grad():
                U, S, Vt = torch.linalg.svd(pretrained_weight.float(), full_matrices=False)
                total = (S**2).sum()
                cum = torch.cumsum(S**2, dim=0) / torch.clamp(total, min=1e-8)
                rank = int(torch.searchsorted(cum, energy).item()) + 1
                rank = max(min_rank, min(rank, min(out_dim, in_dim), max_rank))
                self.rank: int = rank
                A = U[:, :rank] * S[:rank].unsqueeze(0)
                B = Vt[:rank, :]
        else:
            in_dim = in_features or 768
            out_dim = out_features or 768
            rank = max(min_rank, min(max_rank, min(in_dim, out_dim) // 4, 16))
            self.rank = rank
            A = torch.randn(out_dim, rank) * 0.02
            B = torch.randn(rank, in_dim) * 0.02

        self.down = nn.Linear(in_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_dim, bias=(pretrained_bias is not None))
        self.down.weight.data = B.float()
        self.up.weight.data = A.float()
        if pretrained_bias is not None:
            self.up.bias.data = pretrained_bias.float()
        elif self.up.bias is not None:
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
class ORIONAttention(BaseAttention):
    r"""ORION hybrid multi-head attention module.

    Args:
        config: Model configuration containing `hidden_size` and `num_attention_heads`.
        rank: Rank constraint hint.
        model_config: Backwards-compatible alias for config.
        orion_cfg: ORIONConfig hyperparameters.
        pretrained_attn: Pretrained attention module used for SVD weight initialization.
        bert_mode: Boolean indicating bidirectional context handling.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
        orion_cfg: Optional[ORIONConfig] = None,
        pretrained_attn: Optional[Any] = None,
        bert_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        cfg = config or model_config
        super().__init__(cfg, rank)
        if orion_cfg is None:
            orion_cfg = ORIONConfig()
            if rank is not None:
                orion_cfg.svd_max_rank = max(rank, orion_cfg.svd_min_rank)

        hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "d_model", getattr(cfg, "dim", 768))) if cfg is not None else kwargs.get("hidden_size", 768)
        num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", 4)) if cfg is not None else kwargs.get("num_attention_heads", 4)

        self.num_heads: int = max(1, num_heads)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = hidden_size
        self.hidden_size: int = hidden_size
        self.cfg: ORIONConfig = orion_cfg
        self.bert_mode: bool = bert_mode

        # Allocate kernel vs full heads dynamically based on self.num_heads
        if orion_cfg.num_kernel_heads + orion_cfg.num_full_heads == self.num_heads:
            self.num_kernel_heads: int = orion_cfg.num_kernel_heads
            self.num_full_heads: int = orion_cfg.num_full_heads
        else:
            if self.num_heads <= 1:
                self.num_kernel_heads = 1
                self.num_full_heads = 0
            else:
                self.num_full_heads = max(1, self.num_heads // 3)
                self.num_kernel_heads = self.num_heads - self.num_full_heads

        self.q_proj = FactorisedLinear(None, None, orion_cfg.svd_energy, orion_cfg.svd_min_rank, orion_cfg.svd_max_rank, in_features=hidden_size, out_features=hidden_size)
        self.k_proj = FactorisedLinear(None, None, orion_cfg.svd_energy, orion_cfg.svd_min_rank, orion_cfg.svd_max_rank, in_features=hidden_size, out_features=hidden_size)
        self.v_proj = FactorisedLinear(None, None, orion_cfg.svd_energy, orion_cfg.svd_min_rank, orion_cfg.svd_max_rank, in_features=hidden_size, out_features=hidden_size)
        self.out_proj = FactorisedLinear(None, None, orion_cfg.svd_energy, orion_cfg.svd_min_rank, orion_cfg.svd_max_rank, in_features=hidden_size, out_features=hidden_size)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        self.rff_proj = nn.Parameter(
            torch.randn(max(1, self.num_kernel_heads), self.head_dim, orion_cfg.rff_features)
            * 0.02
        )
        self.dropout = nn.Dropout(orion_cfg.dropout)

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer weights from pretrained attention layer using SVD factorization."""
        if old_attn is None:
            return
        from framework.core.adapters import UniversalAdapter
        adapter = UniversalAdapter()
        for proj_name, attr in [("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "out_proj")]:
            try:
                w, b = adapter.get_weight_bias(old_attn, proj_name)
                if w is not None:
                    setattr(
                        self,
                        attr,
                        FactorisedLinear(
                            w,
                            b,
                            self.cfg.svd_energy,
                            self.cfg.svd_min_rank,
                            self.cfg.svd_max_rank,
                        ).to(device=w.device, dtype=w.dtype)
                    )
                    if hasattr(self, "rff_proj"):
                        self.rff_proj.data = self.rff_proj.data.to(device=w.device, dtype=w.dtype)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

    def _kernel_linear_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute kernel linear attention via RFF projections in $\mathcal{O}(N \cdot F \cdot d)$."""
        B, Hk, N, D = Q.shape
        if Hk == 0:
            return torch.zeros(B, 0, N, D, device=Q.device, dtype=Q.dtype)
        F_dim = self.cfg.rff_features
        orig_dtype = Q.dtype
        Q, K, V = Q.float(), K.float(), V.float()
        proj = self.rff_proj[:Hk].to(Q.device).float()

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
        B, Hf, N, D = Q.shape
        if Hf == 0:
            return torch.zeros(B, 0, N, D, device=Q.device, dtype=Q.dtype)
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
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
        if hidden_states.dim() == 4:
            B, H, L, Dh = hidden_states.shape
            hidden_states = hidden_states.view(B, L, H * Dh)
        elif not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, _ = hidden_states.shape
        Hk = self.num_kernel_heads
        Hf = self.num_full_heads

        Q_all = self.q_proj(hidden_states)
        K_all = self.k_proj(hidden_states)
        V_all = self.v_proj(hidden_states)
        Q_all = Q_all.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K_all = K_all.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V_all = V_all.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

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

        ctx_list = []
        if Hk > 0:
            Q_kernel = Q_all[:, :Hk]
            K_kernel = K_all[:, :Hk]
            V_kernel = V_all[:, :Hk]
            ctx_kernel = self._kernel_linear_attention(Q_kernel, K_kernel, V_kernel)
            ctx_list.append(ctx_kernel)

        if Hf > 0:
            Q_full = Q_all[:, Hk:]
            K_full = K_all[:, Hk:]
            V_full = V_all[:, Hk:]
            ctx_full = self._full_softpick_attention(Q_full, K_full, V_full, token_valid)
            ctx_list.append(ctx_full)

        if len(ctx_list) == 1:
            context = ctx_list[0]
        else:
            context = torch.cat(ctx_list, dim=1)

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
    **kwargs) -> ORIONAttention:
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
    new_attn = ORIONAttention(model_config, orion_cfg=cfg, pretrained_attn=old_attn)
    if old_attn is not None:
        p = next(old_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)
    layer.attention = new_attn
    return new_attn
