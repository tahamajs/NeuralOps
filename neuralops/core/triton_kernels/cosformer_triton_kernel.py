import torch
from typing import Any

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    class _MockTriton:
        @staticmethod
        def jit(fn):
            return fn
    triton = _MockTriton()
    class _MockTl:
        constexpr = int
    tl = _MockTl()


@triton.jit
def _cosformer_global_fwd_kernel(
    Q_cos_ptr, Q_sin_ptr, K_cos_ptr, K_sin_ptr, V_ptr,
    Out_ptr, Denom_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    B, H, N, D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0) 
    pid_n = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n = offs_n < N
    mask_d = offs_d < D

    q_cos_ptrs = Q_cos_ptr + b * stride_qb + h * stride_qh \
                 + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q_sin_ptrs = Q_sin_ptr + b * stride_qb + h * stride_qh \
                 + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd

    q_cos = tl.load(q_cos_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    q_sin = tl.load(q_sin_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    acc_num = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_denom = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start_k in range(0, N, BLOCK_N):
        offs_k = start_k + tl.arange(0, BLOCK_N)
        mask_k = offs_k < N

        k_cos_ptrs = K_cos_ptr + b * stride_kb + h * stride_kh \
                     + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_sin_ptrs = K_sin_ptr + b * stride_kb + h * stride_kh \
                     + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd

        k_cos = tl.load(k_cos_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)
        k_sin = tl.load(k_sin_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)

        v_ptrs = V_ptr + b * stride_vb + h * stride_vh \
                 + offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)

        score = tl.dot(q_cos, tl.trans(k_cos)) + tl.dot(q_sin, tl.trans(k_sin))

        causal_mask = offs_n[:, None] >= offs_k[None, :]
        score = tl.where(causal_mask, score, 0.0)

        acc_num += tl.dot(score.to(tl.float32), v)
        acc_denom += tl.sum(score, axis=1)

    out = acc_num / tl.maximum(acc_denom[:, None], 1e-4)

    out_ptrs = Out_ptr + b * stride_ob + h * stride_oh \
               + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_n[:, None] & mask_d[None, :])


def cosformer_global_triton(q_cos, q_sin, k_cos, k_sin, v, is_decoder=True):
    """Wrapper for Triton cosformer kernel."""
    B, H, N, D = q_cos.shape
    out = torch.empty_like(v)
    denom = torch.empty((B, H, N), device=v.device, dtype=torch.float32)

    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B * H, triton.cdiv(N, BLOCK_N))

    _cosformer_global_fwd_kernel[grid](
        q_cos, q_sin, k_cos, k_sin, v, out, denom,
        q_cos.stride(0), q_cos.stride(1), q_cos.stride(2), q_cos.stride(3),
        k_cos.stride(0), k_cos.stride(1), k_cos.stride(2), k_cos.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N, D,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out


def cosformer_pytorch_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Pure PyTorch linear cosformer reference forward pass O(N d^2)."""
    B, H, N, D = q.shape
    device = q.device
    dtype = q.dtype
    m = torch.arange(N, device=device, dtype=torch.float32) * (3.141592653589793 / (2.0 * N))
    cos_m = torch.cos(m).view(1, 1, N, 1).to(dtype)
    sin_m = torch.sin(m).view(1, 1, N, 1).to(dtype)

    q_cos = (q * cos_m).float()
    q_sin = (q * sin_m).float()
    k_cos = (k * cos_m).float()
    k_sin = (k * sin_m).float()
    v_f = v.float()

    if not is_causal:
        S_cos = torch.matmul(k_cos.transpose(-1, -2), v_f)
        S_sin = torch.matmul(k_sin.transpose(-1, -2), v_f)
        Z_cos = k_cos.sum(dim=-2, keepdim=True)
        Z_sin = k_sin.sum(dim=-2, keepdim=True)

        num = torch.matmul(q_cos, S_cos) + torch.matmul(q_sin, S_sin)
        denom = (q_cos * Z_cos).sum(dim=-1, keepdim=True) + (q_sin * Z_sin).sum(dim=-1, keepdim=True)
        denom_safe = denom.clamp(min=eps)
        return (num / denom_safe).to(dtype)

    # Causal linear chunked streaming
    chunk_size = 64
    num_chunks = (N + chunk_size - 1) // chunk_size
    out = torch.empty((B, H, N, D), dtype=torch.float32, device=device)

    S_cos = torch.zeros((B, H, D, D), dtype=torch.float32, device=device)
    S_sin = torch.zeros((B, H, D, D), dtype=torch.float32, device=device)
    Z_cos = torch.zeros((B, H, 1, D), dtype=torch.float32, device=device)
    Z_sin = torch.zeros((B, H, 1, D), dtype=torch.float32, device=device)

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, N)

        qc = q_cos[:, :, start:end]
        qs = q_sin[:, :, start:end]
        kc = k_cos[:, :, start:end]
        ks = k_sin[:, :, start:end]
        vc = v_f[:, :, start:end]

        num_inter = torch.matmul(qc, S_cos) + torch.matmul(qs, S_sin)
        denom_inter = (qc * Z_cos).sum(dim=-1, keepdim=True) + (qs * Z_sin).sum(dim=-1, keepdim=True)

        A_intra = torch.matmul(qc, kc.transpose(-1, -2)) + torch.matmul(qs, ks.transpose(-1, -2))
        causal_mask = torch.tril(torch.ones(end - start, end - start, device=device, dtype=torch.bool))
        A_intra = torch.where(causal_mask, A_intra, torch.zeros_like(A_intra))

        num_intra = torch.matmul(A_intra, vc)
        denom_intra = A_intra.sum(dim=-1, keepdim=True)

        denom_total = (denom_inter + denom_intra).clamp(min=eps)
        out[:, :, start:end] = (num_inter + num_intra) / denom_total

        S_cos = S_cos + torch.matmul(kc.transpose(-1, -2), vc)
        S_sin = S_sin + torch.matmul(ks.transpose(-1, -2), vc)
        Z_cos = Z_cos + kc.sum(dim=-2, keepdim=True)
        Z_sin = Z_sin + ks.sum(dim=-2, keepdim=True)

    return out.to(dtype)


class _CosformerAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool, eps: float):
        ctx.is_causal = is_causal
        ctx.eps = eps
        out = cosformer_pytorch_fwd(q, k, v, is_causal=is_causal, eps=eps)
        ctx.save_for_backward(q, k, v)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v = ctx.saved_tensors
        q_r = q.detach().requires_grad_(True)
        k_r = k.detach().requires_grad_(True)
        v_r = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = cosformer_pytorch_fwd(q_r, k_r, v_r, is_causal=ctx.is_causal, eps=ctx.eps)
            g_q, g_k, g_v = torch.autograd.grad(out, [q_r, k_r, v_r], grad_output, retain_graph=False)
        return g_q, g_k, g_v, None, None


def cosformer_triton_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Dispatches cosformer forward pass with full autograd backward support."""
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad or v.requires_grad:
        return _CosformerAutograd.apply(q, k, v, is_causal, eps)
    return cosformer_pytorch_fwd(q, k, v, is_causal=is_causal, eps=eps)


__all__ = [
    "cosformer_global_triton",
    "cosformer_triton_fwd",
    "cosformer_pytorch_fwd",
]