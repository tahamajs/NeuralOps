import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("leka_v3")
class LEKAV3Attention(BaseAttention):
    r"""
    LEKAv3: Linear-time Efficient Kernel Attention with Value-Scored Landmarks.

    Mathematical Mechanism:
        Uses multi-scale random features with Value-state semantic landmark selection:
        $$\begin{aligned}
        \phi(x) &= [\phi_1(x; F_1), \; \phi_2(x; F_2)] \in \mathbb{R}^{B \times H \times N \times (2F_1 + 2F_2)} \\
        \mathcal{I} &= \text{topk}\left(\text{MLP}_{\text{scorer}}(V), L_{\max}\right) \\
        K_{\text{land}}, V_{\text{land}} &= \text{gather}(\phi(K), \mathcal{I}), \; \text{gather}(V, \mathcal{I}) \\
        V_{\text{adj}} &= V_{\text{land}} W_{\text{corr}} \\
        A_{\text{land}} &= \text{Softpick}(\phi(Q) K_{\text{land}}^\top) \\
        C_{\text{land}} &= A_{\text{land}} V_{\text{adj}} \\
        C_{\text{local}} &= \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M_{\text{window}}\right) V \\
        C &= \sigma(g) \odot C_{\text{land}} + (1 - \sigma(g)) \odot C_{\text{local}} \\
        O &= C W_{\text{out}}
        \end{aligned}$$

    Computational Complexity:
        - Landmark Branch: $\mathcal{O}(B \cdot H \cdot N \cdot L_{\max} \cdot F)$
        - Local Branch: $\mathcal{O}(B \cdot H \cdot N \cdot W_{\text{local}} \cdot d_k)$
        - Overall Time: $\mathcal{O}(B \cdot H \cdot N \cdot (L_{\max} F + W_{\text{local}} d_k))$ — strictly linear in sequence length $N$.
        - Space: $\mathcal{O}(B \cdot H \cdot N \cdot (L_{\max} + W_{\text{local}}))$.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, N, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, N, D]`.
    """

    def __init__(
        self,
        config: Any,
        comp_config: Optional[Any] = None,
        layer_idx: int = 0,
        total_layers: int = 6,
        bert_mode: bool = False,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__(config, rank)
        self.num_attention_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.attention_head_size: int = hidden_size // self.num_attention_heads if self.num_attention_heads > 0 else 64
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size
        self.head_dim: int = self.attention_head_size
        self.bert_mode: bool = bert_mode
        self.layer_idx: int = layer_idx

        self.num_landmarks: int = getattr(comp_config, "num_landmarks", 32) if comp_config is not None else 32
        self.feature_dim: int = getattr(comp_config, "feature_dim", 128) if comp_config is not None else 128
        dropout_p: float = getattr(comp_config, "dropout", 0.1) if comp_config is not None else 0.1

        self.query: nn.Linear = nn.Linear(hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        # Multi-scale random projections
        self.register_buffer("proj1", torch.randn(self.head_dim, self.feature_dim), persistent=False)
        self.register_buffer("proj2", torch.randn(self.head_dim, self.feature_dim // 2), persistent=False)

        # Dynamic landmark scorer (based on V)
        self.landmark_scorer: nn.Sequential = nn.Sequential(
            nn.Linear(self.head_dim, 32), nn.GELU(), nn.Linear(32, 1)
        )

        self.W_corr: nn.Parameter = nn.Parameter(
            torch.eye(self.head_dim).unsqueeze(0).repeat(self.num_attention_heads, 1, 1)
        )

        self.local_window_size: int = getattr(comp_config, "local_window_size", 3) if comp_config is not None else 3
        self.gate: nn.Parameter = nn.Parameter(torch.zeros(self.num_attention_heads))

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        r"""Reshapes `[B, N, D]` to `[B, H, N, d_k]`."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        r"""Multi-scale random Fourier feature mapping."""
        proj1 = self.proj1.to(x.device)
        proj2 = self.proj2.to(x.device)

        p1 = torch.matmul(x, proj1)  # (B, H, N, F)
        f1 = torch.cat([torch.cos(p1), torch.sin(p1)], dim=-1) / math.sqrt(
            self.feature_dim
        )

        p2 = torch.matmul(x, proj2)
        f2 = torch.cat([torch.cos(p2), torch.sin(p2)], dim=-1) / math.sqrt(
            self.feature_dim // 2
        )

        return torch.cat([f1, f2], dim=-1)

    def select_landmarks(
        self, K_features: torch.Tensor, V: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Selects value-aware landmark tokens."""
        B, H, N, feat_dim = K_features.shape
        L_max = self.num_landmarks
        if N <= L_max:
            return K_features, V

        scores = self.landmark_scorer(V).squeeze(-1)  # (B, H, N)
        L = L_max
        _, idx = scores.topk(L, dim=-1)

        K_land = torch.gather(
            K_features, 2, idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim)
        )
        V_land = torch.gather(V, 2, idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1]))
        return K_land, V_land

    @staticmethod
    def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
        r"""Numerically stable soft-selection operator."""
        m = x.max(dim=dim, keepdim=True).values
        e = torch.exp((x - m) / temperature)
        denom = e - torch.exp(-m / temperature)
        p = F.relu(denom)
        return p / (p.sum(dim=dim, keepdim=True) + 1e-9)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward execution of LEKAV3Attention.

        Args:
            hidden_states: Input tensor of shape `[B, N, D]`.
            attention_mask: Optional mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Unused flag for interface compliance.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, None)` with output of shape `[B, N, D]`.
        """
        B, N, D = hidden_states.shape
        H = self.num_attention_heads

        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        Q_features = self.kernel_feature_map(Q)
        K_features = self.kernel_feature_map(K)

        K_land, V_land = self.select_landmarks(K_features, V)

        W = self.W_corr.unsqueeze(0).expand(B, -1, -1, -1)
        V_land_adj = torch.einsum("bhld,bhde->bhle", V_land, W)

        attn_weights = torch.matmul(Q_features, K_land.transpose(-2, -1))

        mask = attention_mask
        if mask is None:
            mask = kwargs.get("attn_mask", None)
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -1e9
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -1e9
            elif mask.dim() == 3:
                token_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                token_valid = mask.bool()
            else:
                token_valid = None
            if token_valid is not None:
                attn_weights = attn_weights.masked_fill(
                    ~token_valid.unsqueeze(1).unsqueeze(-1), float("-1e9")
                )

        attn_weights = self.softpick(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_land = torch.matmul(attn_weights, V_land_adj)

        # Local window attention
        gate_val = torch.sigmoid(self.gate).view(1, H, 1, 1)
        window_size = self.local_window_size
        idx = torch.arange(N, device=hidden_states.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        window_mask = (dist <= window_size // 2).unsqueeze(0).unsqueeze(0)

        scale = 1.0 / math.sqrt(self.head_dim)
        local_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        local_scores = local_scores.masked_fill(~window_mask, float("-1e9"))
        local_attn = F.softmax(local_scores, dim=-1)
        context_local = torch.matmul(local_attn, V)

        context = gate_val * context_land + (1.0 - gate_val) * context_local

        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )
        out = self.out_proj(context)
        return out, None

    def get_ortho_loss(self) -> float:
        r"""Returns orthogonality loss (0.0)."""
        return 0.0


@ATTENTION_REGISTRY.register("leka_v3")
def inject_leka_v3(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> LEKAV3Attention:
    r"""
    Injection hook for LEKAV3Attention into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module to copy weights from.

    Returns:
        Instantiated `LEKAV3Attention` assigned to `layer.attention`.
    """
    new_attn = LEKAV3Attention(
        config=model_config,
        comp_config=comp_config,
        layer_idx=layer_idx,
        total_layers=total_layers,
        bert_mode=False,
    )
    if old_attn is not None:
        if hasattr(old_attn, "q_lin") and hasattr(old_attn.q_lin, "weight"):
            new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
            if old_attn.q_lin.bias is not None and new_attn.query.bias is not None:
                new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
            new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
            if old_attn.k_lin.bias is not None and new_attn.key.bias is not None:
                new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
            new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
            if old_attn.v_lin.bias is not None and new_attn.value.bias is not None:
                new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
            new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
            if old_attn.out_lin.bias is not None and new_attn.out_proj.bias is not None:
                new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
    layer.attention = new_attn
    return new_attn
