r"""
Aegis-Attention V2 Module.

Implements hybrid low-rank factorized attention with local sliding window attention,
global linear kernel attention with adaptive soft-pruning gating, and dynamic multi-head fusion.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisAttentionConfigV2:
    r"""Configuration dataclass for AegisAttentionV2."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    local_window: int = 64
    rank: int = 64
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_adaptive_pruning: bool = True
    gate_init_bias: float = 1.8


class AegisAttentionV2(nn.Module):
    r"""
    Aegis-Attention V2 Layer.

    Mathematical Formulation:
        1. Low-Rank Projection:
           $$Q = X W_{Q1} W_{Q2}, \quad K = X W_{K1} W_{K2}, \quad V = X W_{V1} W_{V2}$$
           with $W_1 \in \mathbb{R}^{D \times r}, W_2 \in \mathbb{R}^{r \times D}$.

        2. Local Band Attention:
           $$A^{\text{local}}_{ij} = \text{Softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_h}} + M^{\text{band}}_{ij} + M^{\text{pad}}_j\right)$$
           $$O^{\text{local}}_i = \sum_{j} A^{\text{local}}_{ij} V_j$$

        3. Global Kernel Linear Attention with Token Pruning:
           $$\phi(u) = \text{ELU}(u) + 1, \quad g = \sigma(\text{PruneGate}(X)) \in [0, 1]^{B \times N \times H}$$
           $$K_g = g \odot K, \quad V_g = g \odot V$$
           $$O^{\text{global}}_i = \frac{\phi(Q_i) \left(\sum_j \phi(K_{g, j})^T V_{g, j}\right)}{\phi(Q_i) \left(\sum_j \phi(K_{g, j})^T \mathbf{1}\right) + \epsilon}$$

        4. Gated Fusion:
           $$\alpha = \sigma(\text{FusionGate}(X) + b_{\text{gate}}) \in [0, 1]^{B \times N \times H \times 1}$$
           $$O_i = \alpha_i \odot O^{\text{local}}_i + (1 - \alpha_i) \odot O^{\text{global}}_i$$
           $$\text{Output} = O W_{O1} W_{O2}$$
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(config, AegisAttentionConfigV2):
            h_size = getattr(config, "hidden_size", 768) if config is not None else 768
            n_heads = getattr(config, "num_attention_heads", 12) if config is not None else 12
            config = AegisAttentionConfigV2(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                head_dim=h_size // max(1, n_heads),
                local_window=getattr(config, "block_size", getattr(config, "local_window", 64)) if config is not None else 64,
                rank=getattr(config, "lora_r", getattr(config, "rank", 64)) if config is not None else 64,
                kernel_feature_dim=getattr(config, "rff_features", getattr(config, "kernel_feature_dim", 128)) if config is not None else 128,
                dropout=getattr(config, "dropout", 0.1) if config is not None else 0.1,
                use_adaptive_pruning=getattr(config, "use_gate", True) if config is not None else True,
                gate_init_bias=getattr(config, "gate_init_bias", 1.8) if config is not None else 1.8,
            )

        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim
        self.local_window: int = config.local_window
        self.rank: int = config.rank
        self.kernel_dim: int = config.kernel_feature_dim
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

        self.q_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.q_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)
        self.k_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.k_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)
        self.v_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.v_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)

        self.out_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.out_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)

        if config.use_adaptive_pruning:
            self.prune_gate: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, self.num_heads),
            )
        else:
            self.prune_gate = None

        self.fusion_gate = nn.Linear(self.hidden_size, self.num_heads)
        self.fusion_gate.bias.data.fill_(config.gate_init_bias)

        if pretrained_attn is not None:
            self._copy_pretrained(pretrained_attn)

    def _copy_pretrained(self, old_attn: nn.Module) -> None:
        q_src = getattr(old_attn, "q_lin", getattr(old_attn, "query", None))
        k_src = getattr(old_attn, "k_lin", getattr(old_attn, "key", None))
        v_src = getattr(old_attn, "v_lin", getattr(old_attn, "value", None))
        out_src = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None))
        if q_src is not None:
            self._svd_init(q_src, self.q_proj1, self.q_proj2, self.rank)
        if k_src is not None:
            self._svd_init(k_src, self.k_proj1, self.k_proj2, self.rank)
        if v_src is not None:
            self._svd_init(v_src, self.v_proj1, self.v_proj2, self.rank)
        if out_src is not None:
            self._svd_init(out_src, self.out_proj1, self.out_proj2, self.rank)

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "AegisAttentionV2":
        r"""Instantiates AegisAttentionV2 with weights transferred from pretrained attention."""
        if model_config is None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None)).out_features
            model_config.num_attention_heads = getattr(old_attn, "n_heads", getattr(old_attn, "num_attention_heads", 12))
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    @staticmethod
    def _svd_init(
        original_linear: nn.Linear,
        lin1: nn.Linear,
        lin2: nn.Linear,
        rank: int,
    ) -> None:
        r"""Initializes factorized weights using truncated SVD of the source layer."""
        W = original_linear.weight.data.float()
        bias = original_linear.bias.data if original_linear.bias is not None else None
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = min(rank, U.size(1), Vh.size(0))
        lin1.weight.data.zero_()
        lin2.weight.data.zero_()
        lin1.weight.data[:r, :].copy_(Vh[:r, :].to(lin1.weight.dtype))
        lin2.weight.data[:, :r].copy_((U[:, :r] @ torch.diag(S[:r])).to(lin2.weight.dtype))
        if bias is not None and lin2.bias is not None:
            lin2.bias.data.copy_(bias.to(lin2.bias.dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, H_dim = hidden_states.shape

        token_valid: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                if attention_mask.shape[-2] == attention_mask.shape[-1]:
                    token_valid = attention_mask[:, 0, 0, :] > -10000.0
                else:
                    token_valid = attention_mask.squeeze(1).squeeze(1) > -10000.0
            elif attention_mask.dim() == 3:
                token_valid = attention_mask.squeeze(1).bool()
            elif attention_mask.dim() == 2:
                token_valid = attention_mask.bool()

        mask_2d = token_valid

        def lr(lin1: nn.Linear, lin2: nn.Linear, x: torch.Tensor) -> torch.Tensor:
            return lin2(lin1(x))

        Q = lr(self.q_proj1, self.q_proj2, hidden_states)
        K = lr(self.k_proj1, self.k_proj2, hidden_states)
        V = lr(self.v_proj1, self.v_proj2, hidden_states)

        Q = Q.view(B, N, self.num_heads, self.head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim)

        if mask_2d is not None:
            m4 = mask_2d.unsqueeze(-1).unsqueeze(-1)
            Q = Q * m4
            K = K * m4
            V = V * m4

        # ---- Local attention ----
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bnhd,bmhd->bhnm", Q, K) * scale
        idx = torch.arange(N, device=hidden_states.device)
        band = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.local_window
        band = band.unsqueeze(0).unsqueeze(0)
        mask_val = -10000.0 if scores.dtype in (torch.float16, torch.bfloat16) else -10000.0
        scores = scores.masked_fill(~band, mask_val)
        if mask_2d is not None:
            key_mask = mask_2d.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, mask_val)

        attn_local = F.softmax(scores, dim=-1)
        attn_local = torch.nan_to_num(attn_local, nan=0.0)
        attn_local = self.dropout(attn_local)
        local_out = torch.einsum("bhnm,bmhd->bnhd", attn_local, V)

        # ---- Global linear attention ----
        def phi(x: torch.Tensor) -> torch.Tensor:
            return F.elu(x) + 1.0

        if self.prune_gate is not None:
            keep = torch.sigmoid(self.prune_gate(hidden_states))
            Kg = K * keep.unsqueeze(-1)
            Vg = V * keep.unsqueeze(-1)
        else:
            Kg, Vg = K, V

        phi_K = phi(Kg)
        phi_Q = phi(Q)

        # Mask phi_K and Vg after applying ELU+1 non-linearity so padding tokens don't accumulate 1s in denominator
        if mask_2d is not None:
            m4 = mask_2d.unsqueeze(-1).unsqueeze(-1).to(phi_K.dtype)
            phi_K = phi_K * m4
            Vg = Vg * m4

        phi_K_t = phi_K.permute(0, 2, 3, 1)
        Vg_t = Vg.permute(0, 2, 1, 3)

        KV = torch.matmul(phi_K_t, Vg_t)
        ones = torch.ones(B, self.num_heads, N, 1, device=Kg.device, dtype=Kg.dtype)
        Z = torch.matmul(phi_K_t, ones)

        phi_Q_t = phi_Q.permute(0, 2, 1, 3).unsqueeze(-2)
        norm_eps = 1e-4 if Kg.dtype in (torch.float16, torch.bfloat16) else 1e-5
        global_out = torch.matmul(phi_Q_t, KV.unsqueeze(2)).squeeze(-2)
        norm = torch.matmul(phi_Q_t, Z.unsqueeze(2)).squeeze(-2)
        global_out = global_out / (norm.clamp(min=norm_eps))
        global_out = global_out.permute(0, 2, 1, 3)

        if mask_2d is not None:
            global_out = global_out * mask_2d.unsqueeze(-1).unsqueeze(-1)

        # ---- Adaptive fusion ----
        alpha = torch.sigmoid(self.fusion_gate(hidden_states)).unsqueeze(-1)
        fused = alpha * local_out + (1.0 - alpha) * global_out

        fused = fused.reshape(B, N, H_dim)
        out = self.out_proj2(self.out_proj1(fused))
        out = self.dropout(out)
        if mask_2d is not None:
            out = out * mask_2d.unsqueeze(-1)

        return out, (attn_local if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj2.weight.device)


@ATTENTION_REGISTRY.register("aegis_v2")
@ATTENTION_REGISTRY.register("aegis_v2_inject")
def inject_aegis_v2(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> AegisAttentionV2:
    r"""Injects AegisAttentionV2 into a transformer layer."""
    if model_config is None and hasattr(layer, "attention"):
        old_attn = layer.attention
    h_size = getattr(model_config, "hidden_size", 768) if model_config is not None else 768
    n_heads = getattr(model_config, "num_attention_heads", 12) if model_config is not None else 12

    cfg = AegisAttentionConfigV2(
        hidden_size=h_size,
        num_attention_heads=n_heads,
        head_dim=h_size // max(1, n_heads),
        local_window=getattr(comp_config, "block_size", 64),
        rank=getattr(comp_config, "lora_r", 64),
        kernel_feature_dim=getattr(comp_config, "rff_features", 128),
        dropout=getattr(comp_config, "dropout", 0.1),
        use_adaptive_pruning=getattr(comp_config, "use_gate", True),
        gate_init_bias=getattr(comp_config, "gate_init_bias", 1.8),
    )
    if model_config is None and old_attn is not None:
        new_attn = AegisAttentionV2.from_distilbert_pretrained(old_attn, config=cfg)
    else:
        new_attn = AegisAttentionV2(config=cfg, pretrained_attn=old_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_attn
    return new_attn
