import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("progressive_token_pruning")
class ProgressiveTokenPruning(nn.Module):
    """
    Learnable token pruning with a lightweight score network.
    After training, it dynamically removes low‑importance tokens.
    """
    def __init__(self, d_model: int, keep_ratio: float = 0.8):
        super().__init__()
        self.score_network = nn.Linear(d_model, 1)
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, hidden)
        Returns:
            pruned_x: (batch, new_len, hidden)
            kept_indices: (batch, new_len) – sorted indices of kept tokens
        """
        B, L, D = x.shape
        scores = torch.sigmoid(self.score_network(x)).squeeze(-1)   # (B, L)
        k = max(1, int(L * self.keep_ratio))
        _, top_indices = torch.topk(scores, k, dim=-1)              # (B, k)
        # Sort to preserve original order (important for positional information)
        top_indices, _ = torch.sort(top_indices, dim=-1)
        pruned_x = torch.gather(x, 1, top_indices.unsqueeze(-1).expand(-1, -1, D))
        return pruned_x, top_indices