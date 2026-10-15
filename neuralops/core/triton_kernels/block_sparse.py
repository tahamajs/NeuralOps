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


# ---------------------------------------------------------------------------
# Fused MoBA Block Router Combine (MoBAAttention)
# ---------------------------------------------------------------------------
def fused_moba_block_router_combine_pytorch(
    Qb: torch.Tensor,
    Kb: torch.Tensor,
    Vb: torch.Tensor,
    scale: float,
    top_k: int,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact PyTorch reference for MoBA block score routing."""
    # Qb, Kb, Vb: (B, H, nb, bs, d)
    Q_mean = Qb.mean(dim=3)
    K_mean = Kb.mean(dim=3)
    block_scores = torch.matmul(Q_mean.float(), K_mean.float().transpose(-2, -1).contiguous()) * scale
    nb = Qb.shape[2]
    if not is_causal:
        block_scores[:, :, :, 0] = block_scores[:, :, :, 0] + 1e4
    else:
        causal_block = torch.triu(
            torch.full((nb, nb), -1e4, device=Qb.device, dtype=torch.float32),
            diagonal=1,
        )
        block_scores = block_scores + causal_block.unsqueeze(0).unsqueeze(0)
    act_top_k = min(top_k, nb)
    _, top_idx = block_scores.topk(act_top_k, dim=-1)
    return block_scores, top_idx


from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl


if HAS_TRITON and tl is not None:
    @triton.jit
    def _fused_moba_block_router_kernel(
        Q_mean_ptr, K_mean_ptr, Scores_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_sb, stride_sh, stride_sn, stride_sk,
        H, nb, D, scale,
        IS_CAUSAL: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """In-SRAM Fused MoBA Block Score Router Kernel."""
        pid_m = tl.program_id(0)  # block row in nb
        pid_bh = tl.program_id(1)
        i_b = pid_bh // H
        i_h = pid_bh % H

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        q_ptr = Q_mean_ptr + i_b * stride_qb + i_h * stride_qh + pid_m * stride_qn + offs_d * stride_qd
        q_vec = tl.load(q_ptr, mask=mask_d, other=0.0).to(tl.float32)

        for col in range(nb):
            k_ptr = K_mean_ptr + i_b * stride_kb + i_h * stride_kh + col * stride_kn + offs_d * stride_kd
            k_vec = tl.load(k_ptr, mask=mask_d, other=0.0).to(tl.float32)

            score = tl.sum(q_vec * k_vec) * scale
            if IS_CAUSAL and col > pid_m:
                score = -1e4
            elif not IS_CAUSAL and col == 0:
                score += 1e4

            s_ptr = Scores_ptr + i_b * stride_sb + i_h * stride_sh + pid_m * stride_sn + col * stride_sk
            tl.store(s_ptr, score)


def fused_moba_block_router_combine_triton(
    Qb: torch.Tensor,
    Kb: torch.Tensor,
    Vb: torch.Tensor,
    scale: float,
    top_k: int,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MoBA block router calculation in GPU SRAM."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and Qb.is_cuda:
        try:
            B, H, nb, bs, D = Qb.shape
            Q_mean = Qb.mean(dim=3).contiguous()
            K_mean = Kb.mean(dim=3).contiguous()

            scores = torch.empty((B, H, nb, nb), dtype=torch.float32, device=Qb.device)
            BLOCK_D = triton.next_power_of_2(D)
            grid = (nb, B * H)
            _fused_moba_block_router_kernel[grid](
                Q_mean, K_mean, scores,
                Q_mean.stride(0), Q_mean.stride(1), Q_mean.stride(2), Q_mean.stride(3),
                K_mean.stride(0), K_mean.stride(1), K_mean.stride(2), K_mean.stride(3),
                scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
                H, nb, D, scale,
                IS_CAUSAL=is_causal,
                BLOCK_D=BLOCK_D,
            )
            act_top_k = min(top_k, nb)
            _, top_idx = scores.topk(act_top_k, dim=-1)
            return scores, top_idx
        except Exception:
            pass
    return fused_moba_block_router_combine_pytorch(Qb, Kb, Vb, scale, top_k, is_causal)


__all__ = [
    "butterfly_mask",
    "block_sparse_attention",
    "fused_moba_block_router_combine_triton",
    "fused_moba_block_router_combine_pytorch",
]
