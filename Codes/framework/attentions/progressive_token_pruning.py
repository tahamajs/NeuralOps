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


@ATTENTION_REGISTRY.register("progressive_token_pruning")
class ProgressiveTokenPruning(nn.Module):
    r"""Learnable token pruning module with lightweight scoring projection.

    Dynamically selects and preserves the most informative token representations along the sequence length.

    Args:
        d_model: Hidden feature dimension $D$.
        keep_ratio: Fraction of tokens to retain ($\rho \in (0.0, 1.0]$).
    """

    def __init__(self, d_model: Any, keep_ratio: float = 0.8, **kwargs: Any) -> None:
        super().__init__()
        if not isinstance(d_model, int):
            d_model = getattr(d_model, "hidden_size", getattr(d_model, "dim", 768))
        self.score_network = nn.Linear(d_model, 1)
        self.keep_ratio: float = keep_ratio

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluate saliency scores and gather top-$k$ tokens.

        Args:
            x: Input sequence tensor of shape `[Batch, SeqLen, HiddenDim]`.

        Returns:
            Tuple of:
                - pruned_x: Pruned sequence tensor `[Batch, NewSeqLen, HiddenDim]`.
                - kept_indices: Monotonically sorted token indices `[Batch, NewSeqLen]`.
        """
        B, L, D = x.shape
        scores = torch.sigmoid(self.score_network(x)).squeeze(-1)  # (B, L)
        k = max(1, min(L, int(L * self.keep_ratio)))
        _, top_indices = torch.topk(scores, k, dim=-1)  # (B, k)
        top_indices, _ = torch.sort(top_indices, dim=-1)
        pruned_x = torch.gather(x, 1, top_indices.unsqueeze(-1).expand(-1, -1, D))
        # Multiply by gathered scores to enable gradient flow to score_network
        gathered_scores = torch.gather(scores, 1, top_indices).unsqueeze(-1)
        pruned_x = pruned_x * gathered_scores
        return pruned_x, top_indices
