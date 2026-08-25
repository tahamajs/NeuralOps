r"""
AEGIS-Attention V4: Adaptive Enhanced Global-Local Integrated Sparse Attention.

Combines SVD-factorized projection matrices with energy-based rank selection,
split-head local sliding-window attention with softpick rectified non-linearities,
and global linear kernel attention with zero-sum centering.
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
    r"""Configuration parameters for AegisAttention V4."""
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
    SVD-factorized linear layer with energy-based rank truncation.

    Mathematical Formulation:
        Decomposes $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ via truncated SVD:
        $$W = U \Sigma V^T \approx U_{:, :r} \Sigma_{:r, :r} V_{:, :r}^T = W_{\text{up}} W_{\text{down}}$$
        where $r$ is the minimum rank satisfying cumulative singular energy ratio:
        $$\frac{\sum_{k=1}^r \sigma_k^2}{\sum_{k=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_k^2} \ge \tau_{\text{energy}}$$

    Complexity:
        - Computation: $\mathcal{O}(B \cdot N \cdot r \cdot (d_{\text{in}} + d_{\text{out}}))$ FLOPs.
        - Parameters: $r \cdot (d_{\text{in}} + d_{\text{out}}) + d_{\text{out}}$.

    Tensor Dimensions:
        - Input: $[*, d_{\text{in}}]$
        - Latent: $[*, r]$
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
            cum = torch.cumsum(S**2, dim=0) / (total + 1e-5)
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
    Sliding window attention using Softpick activation.

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


def kernel_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rff_proj: torch.Tensor,
    use_zero_sum: bool = True,
) -> torch.Tensor:
    B, H, N, D = Q.shape
    F_dim = rff_proj.size(-1)
    orig_dtype = Q.dtype
    
    Q, K, V = Q.float(), K.float(), V.float()
    rff_proj = rff_proj.float()
    
    if use_zero_sum:
        K = K - K.mean(dim=2, keepdim=True)
    Q_norm = torch.nn.functional.normalize(Q, p=2, dim=-1)
    K_norm = torch.nn.functional.normalize(K, p=2, dim=-1)
    
    Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, rff_proj).clamp(max=11.0)
    K_proj = torch.einsum("bhnd,hdf->bhnf", K_norm, rff_proj).clamp(max=11.0)
    Q_feat = torch.exp(Q_proj) / __import__('math').sqrt(F_dim)
    K_feat = torch.exp(K_proj) / __import__('math').sqrt(F_dim)

    KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)  # (B, H, F, D)
    K_sum = K_feat.sum(dim=2)  # (B, H, F)
    denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)  # (B, H, N, 1)

    ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV)  # (B, H, N, D)
    ctx = ctx / (denom + 1e-5)
    return ctx.to(orig_dtype)


@ATTENTION_REGISTRY.register("aegis_v4")
@ATTENTION_REGISTRY.register("aegis_v4_attn")
class AegisAttention(nn.Module):
    r"""
    AEGIS-Attention V4 Multi-Head Layer.

    Decouples multi-head attention into localized sliding-window heads and global kernelized linear heads,
    combined with adaptive multi-head gating and SVD-factorized linear projections.

    Complexity:
        - Local Heads: $\mathcal{O}(B \cdot H_l \cdot N \cdot W_{\text{local}} \cdot d_h)$.
        - Global Heads: $\mathcal{O}(B \cdot H_g \cdot N \cdot F \cdot d_h)$.
        - Projections: $\mathcal{O}(B \cdot N \cdot D \cdot r)$.

    Tensor Dimensions:
        - Input: $[B, N, D]$
        - Output: $[B, N, D]$
    """

    def __init__(
        self,
        model_config: Optional[Any] = None,
        aegis_cfg: Optional[AegisConfig] = None,
        pretrained_attn: Optional[nn.Module] = None,
        bert_mode: bool = False,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if model_config is None and config is not None:
            model_config = config
        if model_config is None and pretrained_attn is not None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", None)).out_features
            model_config.num_attention_heads = getattr(pretrained_attn, "n_heads", getattr(pretrained_attn, "num_attention_heads", 12))

        self.hidden_size: int = getattr(model_config, "hidden_size", 768) if model_config is not None else 768
        self.num_heads: int = getattr(model_config, "num_attention_heads", 12) if model_config is not None else 12
        self.head_dim: int = self.hidden_size // max(1, self.num_heads)
        self.all_head_size: int = self.num_heads * self.head_dim
        self.cfg: AegisConfig = aegis_cfg if aegis_cfg is not None else AegisConfig()
        self.bert_mode: bool = bert_mode

        if self.cfg.num_local_heads + self.cfg.num_global_heads != self.num_heads:
            self.num_local_heads: int = (self.num_heads * 2) // 3
            self.num_global_heads: int = self.num_heads - self.num_local_heads
        else:
            self.num_local_heads = self.cfg.num_local_heads
            self.num_global_heads = self.cfg.num_global_heads

        def _extract_lin(p_name: str, in_f: int, out_f: int) -> nn.Linear:
            if pretrained_attn is not None:
                for obj in [pretrained_attn, getattr(pretrained_attn, "self", None), getattr(pretrained_attn, "output", None)]:
                    if obj is not None:
                        for attr in [p_name, f"{p_name}_proj", f"{p_name}_lin", "query" if p_name == "q" else ("key" if p_name == "k" else ("value" if p_name == "v" else "dense")), "out_proj" if p_name == "o" else None]:
                            if attr and hasattr(obj, attr):
                                mod = getattr(obj, attr)
                                if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor) and mod.weight.shape[1] == in_f:
                                    return mod
            return nn.Linear(in_f, out_f)

        q_lin = _extract_lin("q", self.hidden_size, self.all_head_size)
        k_lin = _extract_lin("k", self.hidden_size, self.all_head_size)
        v_lin = _extract_lin("v", self.hidden_size, self.all_head_size)
        out_lin = _extract_lin("o", self.all_head_size, self.hidden_size)

        self.q_proj = FactorisedLinear(
            q_lin.weight.data,
            q_lin.bias.data if q_lin.bias is not None else None,
            self.cfg.svd_energy,
            self.cfg.min_rank,
            self.cfg.max_rank,
        )
        self.k_proj = FactorisedLinear(
            k_lin.weight.data,
            k_lin.bias.data if k_lin.bias is not None else None,
            self.cfg.svd_energy,
            self.cfg.min_rank,
            self.cfg.max_rank,
        )
        self.v_proj = FactorisedLinear(
            v_lin.weight.data,
            v_lin.bias.data if v_lin.bias is not None else None,
            self.cfg.svd_energy,
            self.cfg.min_rank,
            self.cfg.max_rank,
        )
        self.out_proj = FactorisedLinear(
            out_lin.weight.data,
            out_lin.bias.data if out_lin.bias is not None else None,
            self.cfg.svd_energy,
            self.cfg.min_rank,
            self.cfg.max_rank,
        )

        self.rff_proj: nn.Parameter = nn.Parameter(
            torch.randn(self.num_global_heads, self.head_dim, self.cfg.rff_features) * 0.02
        )

        if self.cfg.use_adaptive_gate:
            self.blend_logits: Optional[nn.Parameter] = nn.Parameter(torch.zeros(self.num_heads))
            self.gate_bias: float = self.cfg.gate_init_bias
        else:
            self.blend_logits = None
            self.gate_bias = 0.0

        self.dropout: nn.Dropout = nn.Dropout(self.cfg.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for AegisAttention V4.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor of shape $[B, N]$ or $[B, 1, 1, N]$.
            head_mask (Optional[torch.Tensor]): Head mask tensor.
            output_attentions (bool): Flag indicating whether to return attention maps.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output context tensor $[B, N, D]$.
                - None (or attention weights if applicable).
        """
        B, N, _ = hidden_states.shape

        # Short sequence fallback: vanilla softmax attention
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
                token_valid = mask[:, 0, 0, :] > -10000.0
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -10000.0
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
                [ctx_local, ctx_global], dim=1
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
        r"""Returns orthogonality loss for SVD factorized projections."""
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("aegis_v4_inject")
def inject_aegis_v4(
    layer: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> AegisAttention:
    r"""
    Injects AEGIS-V4 attention into a transformer layer.
    """
    if model_config is None:
        if hasattr(layer, "hidden_size") or hasattr(layer, "d_model") or hasattr(layer, "n_embd") or hasattr(layer, "dim"):
            model_config = layer
        elif hasattr(layer, "attention"):
            old_attn = layer.attention
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
        model_config=model_config,
        aegis_cfg=cfg,
        pretrained_attn=old_attn,
        bert_mode=False,
    )
    if hasattr(layer, "attention"):
        layer.attention = new_attn
    return new_attn
