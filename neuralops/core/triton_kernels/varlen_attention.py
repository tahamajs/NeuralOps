"""Variable-length attention for batched sequences with autograd backward support."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _varlen_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
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


class _VarlenAttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        causal: bool,
    ):
        ctx.save_for_backward(q, k, v, cu_seqlens_q, cu_seqlens_k)
        ctx.causal = causal
        return _varlen_attention_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        q_r = q.detach().requires_grad_(True)
        k_r = k.detach().requires_grad_(True)
        v_r = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _varlen_attention_fwd(q_r, k_r, v_r, cu_seqlens_q, cu_seqlens_k, causal=ctx.causal)
            g_q, g_k, g_v = torch.autograd.grad(out, [q_r, k_r, v_r], grad_output, retain_graph=False)
        return g_q, g_k, g_v, None, None, None


def varlen_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """PyTorch implementation for varlen attention with full autograd backward support."""
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad or v.requires_grad:
        return _VarlenAttentionAutograd.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
    return _varlen_attention_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)


__all__ = ["varlen_attention_pytorch"]
