import torch
import torch.nn.functional as F

def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """
    Rectified softmax — ~47% sparse, no attention sink.
    Reference: Zuhri et al., arXiv 2504.20966 (Apr 2025).
    """
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)

def select_rank_svd(W: torch.Tensor, energy: float, min_r: int = 4) -> int:
    """
    Energy‑thresholded rank selection from singular values.
    
    Args:
        W: Weight matrix or tensor to analyze.
        energy: Cumulative energy threshold (0.0 to 1.0).
        min_r: Minimum allowable rank.
        
    Returns:
        int: Selected rank.
    """
    with torch.no_grad():
        # Ensure we are on CPU/float for SVD if needed, though torch.linalg.svd is usually fine
        _, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
        total = (S ** 2).sum()
        cum = torch.cumsum(S ** 2, dim=0) / total
        r = int(torch.searchsorted(cum, energy).item()) + 1
        return max(min_r, min(r, min(W.shape[0], W.shape[1])))

def transpose_for_scores(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """
    (B, N, all_head_size) → (B, num_heads, N, head_dim)
    """
    new_shape = x.size()[:-1] + (num_heads, head_dim)
    x = x.view(*new_shape)
    return x.permute(0, 2, 1, 3)
