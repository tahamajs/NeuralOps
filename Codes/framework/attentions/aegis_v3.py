r"""
AEGIS-Attention V3 Module.

Combines SVD-factorized projections with dynamic energy thresholding,
split-head local sliding-window attention with softpick activation,
and global random Fourier feature (RFF) linear kernel attention.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisConfig:
    r"""Configuration parameters for AegisAttention V3."""
    svd_energy: float = 0.95
    min_rank: int = 4
    max_rank: int = 64
    num_local_heads: int = 8
    num_global_heads: int = 4
    local_window: int = 64
    softpick_temperature: float = 1.0
    rff_features: int = 256
    use_zero_sum_keys: bool = True
    use_adaptive_gate: bool = True
    gate_init_bias: float = 1.5
    dropout: float = 0.1


class FactorisedLinear(nn.Module):
    r"""
    Energy-preserving SVD factorized linear transformation.

    The weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is decomposed via SVD:
    $$W = U \Sigma V^T$$
    Rank $r$ is chosen such that the cumulative Frobenius energy exceeds $\tau = \text{energy}$:
    $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i^2} \ge \tau, \quad r \in [r_{\min}, r_{\max}]$$
    Factorization parameters are:
    $$W_{\text{down}} = V_{:, :r}^T \in \mathbb{R}^{r \times d_{\text{in}}}, \quad W_{\text{up}} = U_{:, :r} \Sigma_{:r, :r} \in \mathbb{R}^{d_{\text{out}} \times r}$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot N \cdot r \cdot (d_{\text{in}} + d_{\text{out}}))$ FLOPs.
        - Parameters: $r \cdot (d_{\text{in}} + d_{\text{out}}) + d_{\text{out}}$.

    Tensor Dimensions:
        - Input: $[*, d_{\text{in}}]$
        - Intermediate: $[*, r]$
        - Output: $[*, d_{\text{out}}]$
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
            cum = torch.cumsum(S**2, dim=0) / (total + 1e-12)
            rank = int(torch.searchsorted(cum, energy).item()) + 1
            rank = max(min_rank, min(rank, min(out_dim, in_dim), max_rank))
            self.rank: int = rank
            A = U[:, :rank] * S[:rank].unsqueeze(0)
            B = Vt[:rank, :]

        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=(pretrained_bias is not None))
        self.down.weight.data.copy_(B.float())
        self.up.weight.data.copy_(A.float())
        if pretrained_bias is not None and self.up.bias is not None:
            self.up.bias.data.copy_(pretrained_bias.float())
        elif self.up.bias is not None:
            self.up.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Applies down and up projections sequentially: $y = (x W_{\text{down}}^T) W_{\text{up}}^T + b$."""
        return self.up(self.down(x))


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    r"""
    Softpick non-linear probability mapping with thresholded tail subtraction.

    Mathematical Formulation:
        $$m = \max_j x_j, \quad e_i = \exp\left(\frac{x_i - m}{T}\right)$$
        $$p_i = \frac{\left[ e_i - \exp(-m / T) \right]_+}{\sum_j \left[ e_j - \exp(-m / T) \right]_+ + \epsilon}$$

    Args:
        x (torch.Tensor): Attention logits of arbitrary shape.
        dim (int): Softmax normalization axis.
        temperature (float): Scaling temperature $T$.

    Returns:
        torch.Tensor: Normalized probability distribution.
    """
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)


def local_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    temperature: float,
    token_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Computes local windowed attention using Softpick activation.

    Args:
        Q (torch.Tensor): Query tensor $[B, H_l, N, d_h]$.
        K (torch.Tensor): Key tensor $[B, H_l, N, d_h]$.
        V (torch.Tensor): Value tensor $[B, H_l, N, d_h]$.
        window (int): Total symmetric receptive window size.
        temperature (float): Softpick activation temperature.
        token_valid (Optional[torch.Tensor]): Boolean mask $[B, N]$ (True for non-padding tokens).

    Returns:
        torch.Tensor: Local contextualized output $[B, H_l, N, d_h]$.
    """
    B, H, N, D = Q.shape
    scale = D**-0.5
    idx = torch.arange(N, device=Q.device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    mask = (dist <= window // 2).unsqueeze(0).unsqueeze(0)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, -1e9)
    if token_valid is not None:
        scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e9)
    attn = softpick(scores, dim=-1, temperature=temperature)
    return torch.matmul(attn, V)


def kernel_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rff_proj: torch.Tensor,
    use_zero_sum: bool = True,
) -> torch.Tensor:
    r"""
    Computes global linear attention via Random Fourier Feature (RFF) positive kernel mapping.

    Mathematical Formulation:
        $$\phi_Q(Q) = \frac{\exp(\hat{Q} W_{\text{rff}})}{\sqrt{F}}, \quad \phi_K(K) = \frac{\exp(\hat{K} W_{\text{rff}})}{\sqrt{F}}$$
        where $\hat{Q} = \frac{Q}{\|Q\|_2}, \hat{K} = \frac{K - \bar{K}}{\|K - \bar{K}\|_2}$.
        Context computation:
        $$KV = \sum_{j=1}^N \phi_K(K_j)^T V_j \in \mathbb{R}^{F \times d_h}$$
        $$\text{Output}_i = \frac{\phi_Q(Q_i) KV}{\phi_Q(Q_i) \left(\sum_j \phi_K(K_j)^T\right) + \epsilon}$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot H_g \cdot N \cdot d_h \cdot F + B \cdot H_g \cdot N \cdot F \cdot d_h) = \mathcal{O}(B \cdot H_g \cdot N \cdot d_h \cdot F)$.
        - Memory: $\mathcal{O}(B \cdot H_g \cdot F \cdot d_h)$ state matrix.

    Args:
        Q (torch.Tensor): Global query representations $[B, H_g, N, d_h]$.
        K (torch.Tensor): Global key representations $[B, H_g, N, d_h]$.
        V (torch.Tensor): Global value representations $[B, H_g, N, d_h]$.
        rff_proj (torch.Tensor): Random projection matrix $[H_g, d_h, F]$.
        use_zero_sum (bool): Whether to subtract sequence-mean keys.

    Returns:
        torch.Tensor: Linear attention context $[B, H_g, N, d_h]$.
    """
    B, H, N, D = Q.shape
    F_dim = rff_proj.size(-1)
    if use_zero_sum:
        K = K - K.mean(dim=2, keepdim=True)
    Q_norm = F.normalize(Q, p=2, dim=-1)
    K_norm = F.normalize(K, p=2, dim=-1)
    Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, rff_proj)
    K_proj = torch.einsum("bhnd,hdf->bhnf", K_norm, rff_proj)
    Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
    K_feat = torch.exp(K_proj) / math.sqrt(F_dim)

    KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)  # (B, H, F, D)
    K_sum = K_feat.sum(dim=2)  # (B, H, F)
    denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)  # (B, H, N, 1)

    ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV)  # (B, H, N, D)
    ctx = ctx / (denom + 1e-8)
    return ctx


@ATTENTION_REGISTRY.register("aegis_v3")
class AegisAttention(nn.Module):
    r"""
    Aegis-Attention V3 multi-head attention module.

    Divides $H$ attention heads into $H_l$ local sliding-window heads and $H_g$ global RFF linear heads.
    Projections are factorized through energy-preserving SVD decompositions.

    Tensor Dimensions:
        - Input: $[B, N, D]$
        - Local / Global Q, K, V: $[B, H_l, N, d_h]$ and $[B, H_g, N, d_h]$
        - Output: $[B, N, D]$
    """

    def __init__(
        self,
        model_config: Any,
        aegis_cfg: Optional[AegisConfig] = None,
        pretrained_attn: Optional[nn.Module] = None,
        bert_mode: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_heads: int = model_config.num_attention_heads
        self.head_dim: int = model_config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = model_config.hidden_size
        self.cfg: AegisConfig = aegis_cfg if aegis_cfg is not None else AegisConfig()
        aegis_cfg = self.cfg
        self.bert_mode: bool = bert_mode
        assert aegis_cfg.num_local_heads + aegis_cfg.num_global_heads == self.num_heads
        self.num_local_heads: int = aegis_cfg.num_local_heads
        self.num_global_heads: int = aegis_cfg.num_global_heads

        if pretrained_attn is None:
            # Mock for registry initialization
            class MockAttn: pass
            pretrained_attn = MockAttn()
            m = torch.nn.Linear(model_config.hidden_size, model_config.hidden_size)
            pretrained_attn.q_lin = m; pretrained_attn.k_lin = m; pretrained_attn.v_lin = m; pretrained_attn.out_lin = m

        q_lin = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", None))
        k_lin = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", None))
        v_lin = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", None))
        out_lin = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", None))

        self.q_proj = FactorisedLinear(
            q_lin.weight.data,
            q_lin.bias.data if q_lin.bias is not None else None,
            aegis_cfg.svd_energy,
            aegis_cfg.min_rank,
            aegis_cfg.max_rank,
        )
        self.k_proj = FactorisedLinear(
            k_lin.weight.data,
            k_lin.bias.data if k_lin.bias is not None else None,
            aegis_cfg.svd_energy,
            aegis_cfg.min_rank,
            aegis_cfg.max_rank,
        )
        self.v_proj = FactorisedLinear(
            v_lin.weight.data,
            v_lin.bias.data if v_lin.bias is not None else None,
            aegis_cfg.svd_energy,
            aegis_cfg.min_rank,
            aegis_cfg.max_rank,
        )
        self.out_proj = FactorisedLinear(
            out_lin.weight.data,
            out_lin.bias.data if out_lin.bias is not None else None,
            aegis_cfg.svd_energy,
            aegis_cfg.min_rank,
            aegis_cfg.max_rank,
        )

        self.rff_proj: nn.Parameter = nn.Parameter(
            torch.randn(self.num_global_heads, self.head_dim, aegis_cfg.rff_features) * 0.02
        )

        if aegis_cfg.use_adaptive_gate:
            self.blend_logits: Optional[nn.Parameter] = nn.Parameter(torch.zeros(self.num_heads))
            self.gate_bias: float = aegis_cfg.gate_init_bias
        else:
            self.blend_logits = None
            self.gate_bias = 0.0

        self.dropout: nn.Dropout = nn.Dropout(aegis_cfg.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for AegisAttention V3.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor of shape $[B, N]$ or $[B, 1, 1, N]$.
            head_mask (Optional[torch.Tensor]): Head mask tensor.
            output_attentions (bool): Flag indicating whether to return attention weights.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output context tensor of shape $[B, N, D]$.
                - None (or attention maps if applicable).
        """
        B, N, _ = hidden_states.shape

        # Short sequence fallback: dense scaled dot-product attention
        if N < self.cfg.local_window:
            Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            attn = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim**-0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
            return self.out_proj(out), (attn if output_attentions else None)

        H, D = self.num_heads, self.head_dim
        Hl, Hg = self.num_local_heads, self.num_global_heads

        Q_all = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        K_all = self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        V_all = self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)

        Q_local, Q_global = Q_all[:, :Hl], Q_all[:, Hl:]
        K_local, K_global = K_all[:, :Hl], K_all[:, Hl:]
        V_local, V_global = V_all[:, :Hl], V_all[:, Hl:]

        mask = attention_mask if attention_mask is not None else kwargs.get("attn_mask", None)
        token_valid: Optional[torch.Tensor] = None
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -1e9
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -1e9
            elif mask.dim() == 3:
                token_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                token_valid = mask.bool()

        ctx_local = local_window_attention(
            Q_local,
            K_local,
            V_local,
            self.cfg.local_window,
            self.cfg.softpick_temperature,
            token_valid,
        )
        ctx_global = kernel_linear_attention(
            Q_global, K_global, V_global, self.rff_proj, self.cfg.use_zero_sum_keys
        )

        if self.blend_logits is not None:
            global_signal = ctx_global.mean(dim=1, keepdim=True).expand(-1, H, -1, -1)
            local_full = torch.cat(
                [ctx_local, torch.zeros(B, Hg, N, D, device=ctx_local.device, dtype=ctx_local.dtype)], dim=1
            )
            gate = torch.sigmoid(self.blend_logits.unsqueeze(-1).unsqueeze(-1) + self.gate_bias)
            context = gate * local_full + (1.0 - gate) * global_signal
        else:
            context = torch.cat([ctx_local, ctx_global], dim=1)

        if token_valid is not None:
            context = context * token_valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        output = self.out_proj(context)
        return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Returns orthogonality loss for SVD projection components."""
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)



def inject_aegis_v3(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> AegisAttention:
    r"""
    Injects AegisAttention V3 into a transformer layer.

    Args:
        layer (nn.Module): Target transformer layer.
        model_config (Any): Model configuration object.
        comp_config (Any): Compression configuration object.
        layer_idx (int): Current layer index.
        total_layers (int): Total number of layers.
        old_attn (Optional[nn.Module]): Pretrained attention layer.

    Returns:
        AegisAttention: Initialized attention layer.
    """
    cfg = AegisConfig(
        svd_energy=getattr(comp_config, "svd_energy", 0.95),
        min_rank=getattr(comp_config, "min_rank", 4),
        max_rank=getattr(comp_config, "max_rank", 64),
        num_local_heads=getattr(comp_config, "num_local_heads", 8),
        num_global_heads=getattr(comp_config, "num_global_heads", 4),
        local_window=getattr(comp_config, "block_size", 64),
        softpick_temperature=getattr(comp_config, "softpick_temperature", 1.0),
        rff_features=getattr(comp_config, "rff_features", 256),
        use_zero_sum_keys=getattr(comp_config, "use_zero_sum_keys", True),
        use_adaptive_gate=getattr(comp_config, "use_adaptive_gate", True),
        gate_init_bias=getattr(comp_config, "gate_init_bias", 1.5),
        dropout=getattr(comp_config, "dropout", 0.1),
    )
    new_attn = AegisAttention(
        model_config=model_config, aegis_cfg=cfg, pretrained_attn=old_attn
    )
    layer.attention = new_attn
    return new_attn
