import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("mixattention_moska")
class MixAttentionMoSKA(BaseAttention):
    r"""
    MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention (FASA).

    Mathematical Mechanism:
        Computes token variance across channel dimensions as an information-theoretic importance metric,
        pruning the key-value sequence down to a fraction $\rho = \text{fasa\_keep\_ratio}$:
        $$\begin{aligned}
        I_t &= \frac{1}{H} \sum_{h=1}^{H} \text{Var}_{d}\left(K_{h, t, :}\right) \in \mathbb{R}^{B \times L} \\
        \mathcal{I}_{\text{keep}} &= \text{topk}\left(I, \max(1, \lfloor \rho L \rfloor), \text{dim}=-1\right) \\
        K_{\text{pruned}} &= \text{gather}\left(K, \mathcal{I}_{\text{keep}}\right), \quad V_{\text{pruned}} = \text{gather}\left(V, \mathcal{I}_{\text{keep}}\right) \\
        S &= \frac{Q K_{\text{pruned}}^\top}{\sqrt{d_k}} + M_{\text{mask}} \\
        P &= \text{softmax}(S, \text{dim}=-1) \\
        O &= \left(P V_{\text{pruned}}\right) W_O
        \end{aligned}$$

    Computational Complexity:
        - Variance Importance: $\mathcal{O}(B \cdot H \cdot L \cdot d_k)$
        - Pruned Attention Map: $\mathcal{O}(B \cdot H \cdot L \cdot (\rho L) \cdot d_k)$
        - Memory Footprint: $\mathcal{O}(B \cdot H \cdot L \cdot (\rho L))$ vs $\mathcal{O}(B \cdot H \cdot L^2)$ for full attention.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, attn_weights)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 64,
        fasa_keep_ratio: float = 0.189,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = getattr(config, "hidden_size", 768)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.fasa_keep_ratio: float = fasa_keep_ratio

        self.W_q: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj: nn.Linear = nn.Linear(self.dim, self.dim)

    def _fasa_prune(
        self, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Prunes key and value tensors according to token variance.

        Args:
            k: Key tensor `[B, H, L, d_k]`.
            v: Value tensor `[B, H, L, d_k]`.
            mask: Optional mask tensor.

        Returns:
            Tuple `(k_pruned, v_pruned, pruned_mask)`.
        """
        B, H, L, D_h = k.shape
        importance = k.float().var(dim=-1).mean(dim=1)  # (B, L)
        keep = max(1, int(L * self.fasa_keep_ratio))
        _, idx = torch.topk(importance, keep, dim=-1)
        idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D_h)
        k_pruned = k.gather(2, idx_exp)
        v_pruned = v.gather(2, idx_exp)

        pruned_mask = None
        if mask is not None:
            if mask.dim() == 2:
                pruned_mask = mask.gather(1, idx)
            elif mask.dim() == 4:
                mask_2d = mask[:, 0, 0, :]
                pruned_mask = mask_2d.gather(1, idx)

        return k_pruned, v_pruned, pruned_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward pass for MixAttentionMoSKA.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Boolean flag to return attention matrix.
            mask: Optional alias for attention_mask.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, attn_weights)` where `output` is `[B, L, D]`.
        """
        if attention_mask is None and mask is not None:
            attention_mask = mask

        x = hidden_states
        B, L, D = x.shape
        H = self.num_heads
        q = self.W_q(x).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(x).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(x).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)

        k_p, v_p, p_mask = self._fasa_prune(k, v, attention_mask)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k_p.transpose(-2, -1)) * scale

        if p_mask is not None:
            attn_scores = attn_scores + (1.0 - p_mask[:, None, None, :].to(attn_scores.dtype)) * -10000.0

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v_p).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        output = self.out_proj(out)
        return output, (attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("mixattention_moska")
def inject_mixattention_moska(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> MixAttentionMoSKA:
    r"""
    Injection hook for MixAttentionMoSKA into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module.

    Returns:
        Instantiated `MixAttentionMoSKA` assigned to `layer.attention`.
    """
    fasa_keep_ratio: float = getattr(comp_config, "fasa_keep_ratio", 0.189)
    rank: int = getattr(comp_config, "rank", 64)
    new_attn = MixAttentionMoSKA(
        config=model_config,
        rank=rank,
        fasa_keep_ratio=fasa_keep_ratio,
    )
    layer.attention = new_attn
    return new_attn
