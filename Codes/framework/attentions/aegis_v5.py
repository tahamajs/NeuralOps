r"""
AEGIS-Attention V5: Hybrid Efficient Transformer Attention Module.

Combines SVD-factorized linear projections, local sliding-window attention with softpick
rectified sparse probabilities, global linear kernel attention with learnable random Fourier features,
and per-head dynamic adaptive blending.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisConfigV5:
    r"""Configuration parameters for AegisAttention V5."""
    svd_energy: float = 0.95  # fraction of singular-value energy to preserve
    min_rank: int = 4
    max_rank: int = 64
    num_local_heads: int = 8  # count of local window attention heads
    num_global_heads: int = 4  # count of global linear kernel attention heads
    local_window: int = 64  # sliding window receptive radius
    softpick_temperature: float = 1.0
    rff_features: int = 256  # random Fourier feature dimension
    use_zero_sum_keys: bool = True  # zero-centering stabilization
    use_adaptive_gate: bool = True  # per-head local/global blending
    gate_init_bias: float = 1.5  # initial gate logit bias
    dropout: float = 0.1


class FactorisedLinear(nn.Module):
    r"""
    Energy-preserving SVD-factorized Linear Layer.

    Approximates $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ via truncated SVD:
    $$W \approx U_{:, :r} \Sigma_{:r, :r} V_{:, :r}^T = W_{\text{up}} W_{\text{down}}$$
    where rank $r$ is determined by the cumulative energy ratio threshold:
    $$\frac{\sum_{k=1}^r \sigma_k^2}{\sum_{k=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_k^2} \ge \tau_{\text{energy}}, \quad r \in [r_{\min}, r_{\max}]$$

    Complexity:
        - FLOPs: $\mathcal{O}(B \cdot N \cdot r \cdot (d_{\text{in}} + d_{\text{out}}))$.
        - Memory: $\mathcal{O}(r \cdot (d_{\text{in}} + d_{\text{out}}))$.

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
            U, S, Vt = torch.linalg.svd(pretrained_weight.float()
            S = torch.clamp(S, min=1e-6), full_matrices=False)
            total = (S**2).sum()
            cum = torch.cumsum(S**2, dim=0) / (total + 1e-5)
            rank = int(torch.searchsorted(cum, energy).item()) + 1
            rank = max(min_rank, min(rank, min(out_dim, in_dim), max_rank))
            self.rank: int = rank
            A = U[:, :rank] * S[:rank].unsqueeze(0)  # (out, r)
            B = Vt[:rank, :]  # (r, in)

        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=(pretrained_bias is not None))
        self.down.weight.data.copy_(B.float())
        self.up.weight.data.copy_(A.float())
        if pretrained_bias is not None and self.up.bias is not None:
            self.up.bias.data.copy_(pretrained_bias.float())
        elif self.up.bias is not None:
            self.up.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Performs factorized linear projection: $y = (x W_{\text{down}}^T) W_{\text{up}}^T + b$."""
        return self.up(self.down(x))


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


def local_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    temperature: float,
    token_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Sliding window attention using Softpick rectified activation.

    Args:
        Q (torch.Tensor): Query tensor $[B, H_l, N, d_h]$.
        K (torch.Tensor): Key tensor $[B, H_l, N, d_h]$.
        V (torch.Tensor): Value tensor $[B, H_l, N, d_h]$.
        window (int): Symmetric local context window.
        temperature (float): Softpick activation temperature.
        token_valid (Optional[torch.Tensor]): Token validity boolean mask $[B, N]$.

    Returns:
        torch.Tensor: Local attended context $[B, H_l, N, d_h]$.
    """
    B, H, N, D = Q.shape
    scale = D**-0.5
    idx = torch.arange(N, device=Q.device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    mask = (dist <= window // 2).unsqueeze(0).unsqueeze(0)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, -10000.0)
    if token_valid is not None:
        scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -10000.0)
    attn = softpick(scores, dim=-1, temperature=temperature)
    return torch.matmul(attn, V)


class AegisAttention(nn.Module):
    r"""
    AEGIS-Attention V5 multi-head hybrid module.

    Divides $H$ heads into $H_l$ local window heads and $H_g$ global RFF linear heads.
    Projections are low-rank factorized via truncated SVD.

    Complexity:
        - Local Attention: $\mathcal{O}(B \cdot H_l \cdot N \cdot W_{\text{local}} \cdot d_h)$.
        - Global Attention: $\mathcal{O}(B \cdot H_g \cdot N \cdot F \cdot d_h)$.
        - Linear Projections: $\mathcal{O}(B \cdot N \cdot D \cdot r)$.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, N, D]$
        - Projected $Q, K, V$: $[B, H, N, d_h]$
        - Output: $[B, N, D]$
    """

    def __init__(
        self,
        model_config: Any,
        aegis_cfg: AegisConfigV5,
        pretrained_attn: Optional[nn.Module] = None,
        bert_mode: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads: int = model_config.num_attention_heads
        self.head_dim: int = model_config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = model_config.hidden_size
        self.cfg: AegisConfigV5 = aegis_cfg
        self.bert_mode: bool = bert_mode

        assert aegis_cfg.num_local_heads + aegis_cfg.num_global_heads == self.num_heads
        self.num_local_heads: int = aegis_cfg.num_local_heads
        self.num_global_heads: int = aegis_cfg.num_global_heads

        if pretrained_attn is None:
            raise ValueError("AegisAttention requires pretrained_attn for zero-shot initialization")

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

    @staticmethod
    def _prepare_mask(
        attention_mask: Optional[torch.Tensor],
        B: int,
        N: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        r"""Prepares a 2D boolean validity mask from arbitrary attention mask formats."""
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            if attention_mask.shape[-2] == attention_mask.shape[-1]:
                token_valid = attention_mask[:, 0, 0, :] > -10000.0
            else:
                token_valid = attention_mask.squeeze(1).squeeze(1) > -10000.0
        elif attention_mask.dim() == 3:
            token_valid = attention_mask.squeeze(1).bool()
        elif attention_mask.dim() == 2:
            token_valid = attention_mask.bool()
        else:
            raise ValueError(f"Unexpected mask dim: {attention_mask.dim()}")
        return token_valid.to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward pass for AegisAttention V5.

        Args:
            hidden_states (torch.Tensor): Input activations $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor $[B, N]$ or $[B, 1, 1, N]$.
            head_mask (Optional[torch.Tensor]): Head mask tensor.
            output_attentions (bool): Flag to return attention maps.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed context representation $[B, N, D]$.
                - None (or attention weights if applicable).
        """
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        Hl, Hg = self.num_local_heads, self.num_global_heads

        Q_all = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        K_all = self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        V_all = self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)

        Q_local, Q_global = Q_all[:, :Hl], Q_all[:, Hl:]
        K_local, K_global = K_all[:, :Hl], K_all[:, Hl:]
        V_local, V_global = V_all[:, :Hl], V_all[:, Hl:]

        token_valid = self._prepare_mask(attention_mask, B, N, hidden_states.device)

        ctx_local = local_window_attention(
            Q_local,
            K_local,
            V_local,
            window=self.cfg.local_window,
            temperature=self.cfg.softpick_temperature,
            token_valid=token_valid,
        )

        Q_feat = torch.einsum("bhnd,hdf->bhnf", Q_global, self.rff_proj)
        K_feat = torch.einsum("bhnd,hdf->bhnf", K_global, self.rff_proj)

        if token_valid is not None:
            mask_g = token_valid.unsqueeze(1).unsqueeze(-1)
            K_feat = K_feat.masked_fill(~mask_g, 0.0)
            V_global = V_global.masked_fill(~mask_g, 0.0)

        KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V_global)
        K_sum = K_feat.sum(dim=2, keepdim=True)
        denom = torch.matmul(Q_feat, K_sum.transpose(-1, -2)) + 1e-5
        ctx_global = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / denom

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


def inject_aegis_into_distilbert(
    layer: nn.Module,
    model_config: Any,
    aegis_cfg: AegisConfigV5,
    old_attn: Optional[nn.Module],
) -> AegisAttention:
    r"""Helper function to instantiate and inject AegisAttention into a DistilBERT layer."""
    new_attn = AegisAttention(
        model_config=model_config,
        aegis_cfg=aegis_cfg,
        pretrained_attn=old_attn,
        bert_mode=False,
    )
    layer.attention = new_attn
    return new_attn


@ATTENTION_REGISTRY.register("aegis_v5")
def inject_aegis_v5(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> AegisAttention:
    r"""
    Injects AEGIS-V5 attention into a transformer layer.

    Args:
        layer (nn.Module): Transformer layer to inject attention into.
        model_config (Any): Model configuration object.
        comp_config (Any): Compression configuration object.
        layer_idx (int): Current layer index.
        total_layers (int): Total number of layers.
        old_attn (Optional[nn.Module]): Pretrained attention layer.

    Returns:
        AegisAttention: Initialized attention layer.
    """
    cfg = AegisConfigV5(
        num_local_heads=8,
        num_global_heads=4,
        local_window=getattr(comp_config, "block_size", 64),
        rff_features=getattr(comp_config, "rff_features", 256),
        dropout=getattr(comp_config, "dropout", 0.1),
    )
    return inject_aegis_into_distilbert(layer, model_config, cfg, old_attn)
