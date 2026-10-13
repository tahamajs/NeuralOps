import torch
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

        # بارگذاری V
        v_ptrs = V_ptr + b * stride_vb + h * stride_vh \
                 + offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)

        # محاسبهٔ score = Q_cos @ K_cos^T + Q_sin @ K_sin^T  →  (BLOCK_N, BLOCK_N)
        score = tl.dot(q_cos, tl.trans(k_cos)) + tl.dot(q_sin, tl.trans(k_sin))

        # اگر decoder هستیم، ماسک causal اعمال شود
        causal_mask = offs_n[:, None] >= offs_k[None, :]
        score = tl.where(causal_mask, score, 0.0)

        acc_num += tl.dot(score.to(tl.float32), v)
        acc_denom += tl.sum(score, axis=1)

    out = acc_num / tl.maximum(acc_denom[:, None], 1e-4)

    out_ptrs = Out_ptr + b * stride_ob + h * stride_oh \
               + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_n[:, None] & mask_d[None, :])


def cosformer_global_triton(q_cos, q_sin, k_cos, k_sin, v, is_decoder=True):
    """Wrapper for kernel up.

    Args:
        q_cos, q_sin: (B, H, N, D)
        k_cos, k_sin: (B, H, N, D)
        v: (B, H, N, D)
    Returns:
        out: (B, H, N, D)
    """
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
    """Pure PyTorch cosformer reference forward pass."""
    B, H, N, D = q.shape
    device = q.device
    dtype = q.dtype
    m = torch.arange(N, device=device, dtype=torch.float32) * (3.141592653589793 / (2.0 * N))
    cos_m = torch.cos(m).view(1, 1, N, 1).to(dtype)
    sin_m = torch.sin(m).view(1, 1, N, 1).to(dtype)

    q_cos = q * cos_m
    q_sin = q * sin_m
    k_cos = k * cos_m
    k_sin = k * sin_m

    scores = torch.matmul(q_cos, k_cos.transpose(-2, -1)) + torch.matmul(q_sin, k_sin.transpose(-2, -1))
    if is_causal:
        mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.view(1, 1, N, N), 0.0)

    denom = scores.sum(dim=-1, keepdim=True).clamp(min=eps)
    weights = scores / denom
    return torch.matmul(weights, v)


def cosformer_triton_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Dispatches cosformer forward pass."""
    return cosformer_pytorch_fwd(q, k, v, is_causal=is_causal, eps=eps)


__all__ = [
    "cosformer_global_triton",
    "cosformer_triton_fwd",
    "cosformer_pytorch_fwd",
]