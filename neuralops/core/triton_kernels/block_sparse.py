"""Block-sparse attention with butterfly patterns and autograd backward support."""
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


def _block_sparse_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


class _BlockSparseAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        ctx.save_for_backward(q, k, v, mask)
        return _block_sparse_attention_fwd(q, k, v, mask)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, mask = ctx.saved_tensors
        q_r = q.detach().requires_grad_(True)
        k_r = k.detach().requires_grad_(True)
        v_r = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _block_sparse_attention_fwd(q_r, k_r, v_r, mask)
            g_q, g_k, g_v = torch.autograd.grad(out, [q_r, k_r, v_r], grad_output, retain_graph=False)
        return g_q, g_k, g_v, None


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    block_size: int = 64,
) -> torch.Tensor:
    """Block-sparse attention with full autograd backward support."""
    N = q.shape[-2]
    if mask is None:
        mask = butterfly_mask(N, block_size, q.device)
    
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad or v.requires_grad:
        return _BlockSparseAutograd.apply(q, k, v, mask)
    return _block_sparse_attention_fwd(q, k, v, mask)
