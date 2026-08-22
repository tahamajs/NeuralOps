r"""
Progressive Token Pruning (PTP).

Dynamic token selection module evaluating learned per-token saliency scores to progressively
prune redundant sequence representations while maintaining relative token ordering.

Mathematical Mechanism:
    For input representations $X \in \mathbb{R}^{B \times L \times D}$:
    1. **Saliency Scoring:**
       $$s_i = \sigma(X_i W_s + b_s) \in (0, 1), \quad \forall i \in \{1, \dots, L\}$$
    2. **Top-$k$ Token Selection:**
       With retention ratio $\rho \in (0, 1]$, determine retention count $k = \max(1, \lfloor \rho \cdot L \rfloor)$.
       $$\mathcal{I} = \operatorname{sort}\left(\operatorname{arg\,top-}k(\{s_i\}_{i=1}^L)\right) \subset \{1, \dots, L\}$$
    3. **Gather Operation:**
       $$X_{\text{pruned}} = \operatorname{gather}(X, \text{dim}=1, \text{index}=\mathcal{I}) \in \mathbb{R}^{B \times k \times D}$$

Computational Complexity:
    - Scoring projection: $\mathcal{O}(B \cdot L \cdot D)$.
    - Top-$k$ selection & sorting: $\mathcal{O}(B \cdot L + B \cdot k \log k)$.
    - Gather: $\mathcal{O}(B \cdot k \cdot D)$.
    - Total module complexity: $\mathcal{O}(B \cdot L \cdot D)$.
"""

from typing import Tuple, Any

import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


class ProgressiveTokenPruning(nn.Module):
    r"""Learnable token pruning module with lightweight scoring projection."""

    def __init__(self, d_model: int, keep_ratio: float = 0.8, old_attn: Any = None) -> None:
        super().__init__()
        self.score_network = nn.Linear(d_model, 1)
        self.keep_ratio: float = keep_ratio
        self.old_attn = old_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Any = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Any]:
        x = hidden_states
        B, L, D = x.shape
        scores = torch.sigmoid(self.score_network(x)).squeeze(-1)  # (B, L)
        k = max(1, min(L, int(L * self.keep_ratio)))
        _, top_indices = torch.topk(scores, k, dim=-1)  # (B, k)
        top_indices, _ = torch.sort(top_indices, dim=-1)
        
        pruned_x = torch.gather(x, 1, top_indices.unsqueeze(-1).expand(-1, -1, D))
        gathered_scores = torch.gather(scores, 1, top_indices).unsqueeze(-1)
        pruned_x = pruned_x * gathered_scores

        # To prevent crashing the residual connection (which expects L), we must project back to L
        # by scattering the pruned tokens back into a zero tensor of shape [B, L, D]
        out = torch.zeros_like(x)
        out.scatter_(1, top_indices.unsqueeze(-1).expand(-1, -1, D), pruned_x)
        
        # We also want to pass this through standard attention so it actually acts as an attention layer
        if self.old_attn is not None:
            out, attn = self.old_attn(out, attention_mask=attention_mask, output_attentions=output_attentions, **kwargs)
            return out, attn
        
        return out, None

@ATTENTION_REGISTRY.register("progressive_token_pruning")
def inject_progressive_token_pruning(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any = None,
) -> ProgressiveTokenPruning:
    keep_ratio = getattr(comp_config, "keep_ratio", 0.5)
    return ProgressiveTokenPruning(
        d_model=model_config.hidden_size,
        keep_ratio=keep_ratio,
        old_attn=old_attn
    )
