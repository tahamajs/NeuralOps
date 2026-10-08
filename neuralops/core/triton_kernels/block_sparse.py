"""Block-sparse attention with butterfly patterns."""
from __future__ import annotations
import torch

__all__ = ["butterfly_mask", "block_sparse_attention"]


def butterfly_mask(N: int, block_size: int = 64, device=None) -> torch.Tensor:
    """Generate butterfly block-sparse mask."""
    num_blocks = (N + block_size - 1) // block_size
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    
    for i in range(num_blocks):
        # Diagonal block
        start = i * block_size
        end = min(start + block_size, N)
        mask[start:end, start:end] = True
        # Off-diagonal for butterfly
        for offset in [1, 2, 4, 8]:
            j = i ^ offset
            if j < num_blocks:
                js, je = j * block_size, min((j + 1) * block_size, N)
                mask[start:end, js:je] = True
    
    return mask


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    block_size: int = 64,
) -> torch.Tensor:
    """Block-sparse attention using mask."""
    N = q.shape[-2]
    if mask is None:
        mask = butterfly_mask(N, block_size, q.device)
    
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
