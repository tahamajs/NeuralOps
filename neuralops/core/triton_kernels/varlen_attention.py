"""Variable-length attention for batched sequences."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def varlen_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
) -> torch.Tensor:
    """PyTorch fallback for varlen attention.
    
    Args:
        q: [total_q, H, D] — flattened across batch
        k: [total_k, H, D]
        v: [total_k, H, D]
        cu_seqlens_q: [B+1] cumulative lengths
        cu_seqlens_k: [B+1]
        max_seqlen_q: max sequence length in q
        max_seqlen_k: max sequence length in k
        causal: whether to apply causal mask
    
    Returns:
        [total_q, H, D]
    """
    B = cu_seqlens_q.shape[0] - 1
    outputs = []
    
    for b in range(B):
        qs, qe = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        ks, ke = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        
        q_b = q[qs:qe].transpose(0, 1).unsqueeze(0)  # [1, H, Nq, D]
        k_b = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_b = v[ks:ke].transpose(0, 1).unsqueeze(0)
        
        out_b = F.scaled_dot_product_attention(
            q_b, k_b, v_b, is_causal=causal,
        )
        outputs.append(out_b.squeeze(0).transpose(0, 1))
    
    return torch.cat(outputs, dim=0)


__all__ = ["varlen_attention_pytorch"]
