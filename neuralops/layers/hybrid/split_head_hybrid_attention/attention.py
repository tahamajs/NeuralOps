import math
import torch
import torch.nn.functional as F
from neuralops.core.common import temperature_softmax

def local_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    softpick_temperature: float = 1.0,
    token_valid: torch.Tensor | None = None,
    is_decoder: bool = False,
    past_len: int = 0,
) -> torch.Tensor:
    """
    Sliding-window attention.

    V5-FIX-02: `past_len` shifts the query positions so that the window is
    measured in absolute token positions. In prefill, past_len=0. In decode
    with a KV-cache of length L and N new tokens, past_len=L.

    V5-FIX-09: `window` is the full width of the sliding window. The effective
    radius used for masking is `window // 2`, so a query at position q attends
    to keys in [q - window//2, q + window//2].
    """
    B, H, N, D = Q.shape
    N_k = K.shape[2]
    radius = window // 2
    if (
        Q.is_cuda
        and past_len == 0
        and N == N_k
        and token_valid is None
        and abs(softpick_temperature - 1.0) < 1e-3
        and D <= 256
    ):
        try:
            from neuralops.ops.cuda.hybrid_local_kernel import hybrid_local_attn

            return hybrid_local_attn(Q, K, V, radius=radius, is_causal=is_decoder)
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

    scale = (1.0 / math.sqrt(max(int(D), 1))) / max(1e-4, float(softpick_temperature))
    chunk_size = 64
    mask_val = -10000.0
    outputs = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        q_abs_start = past_len + start
        q_abs_end = past_len + end
        k_start = max(0, q_abs_start - radius)
        k_end = min(N_k, q_abs_end if is_decoder else q_abs_end + radius)
        if k_end <= k_start:
            outputs.append(Q.new_zeros(B, H, end - start, D))
            continue

        Qc = Q[:, :, start:end]
        Kc = K[:, :, k_start:k_end]
        Vc = V[:, :, k_start:k_end]

        abs_q = torch.arange(q_abs_start, q_abs_end, device=Q.device).unsqueeze(1)
        abs_k = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
        mask = ((abs_q - abs_k).abs() <= radius).unsqueeze(0).unsqueeze(0)
        if is_decoder:
            mask = mask & (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(Qc.float(), Kc.float().transpose(-2, -1)) * scale
        scores = scores.masked_fill(~mask, mask_val)
        if token_valid is not None:
            kv_valid = token_valid[:, None, None, k_start:k_end].to(torch.bool)
            scores = scores.masked_fill(~kv_valid, mask_val)
            combined = mask & kv_valid
        else:
            combined = mask
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

        any_valid = combined.any(dim=-1, keepdim=True)
        attn = temperature_softmax(scores, dim=-1, temperature=softpick_temperature)
        attn = attn * any_valid.to(attn.dtype)
        outputs.append(torch.matmul(attn, Vc.to(attn.dtype)))

    return torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]


def kernel_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rff_proj: torch.Tensor,
    use_zero_sum: bool = True,
    max_clamp: float = 10.0,
) -> torch.Tensor:
    """
    Non-causal global kernel linear attention (FAVOR+ style).

    rff_proj: (H_g, D, F) random Fourier features for global heads.
    Returns context of shape (B, H_g, N, D).
    """
    B, H, N, D = Q.shape
    F_dim = rff_proj.size(-1)
    orig_dtype = Q.dtype

    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()
    rff_proj_f = rff_proj.float()

    if use_zero_sum:
        K_f = K_f - K_f.mean(dim=2, keepdim=True)

    Q_norm = F.normalize(Q_f, p=2, dim=-1)
    K_norm = F.normalize(K_f, p=2, dim=-1)

    Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, rff_proj_f).clamp(max=max_clamp)
    K_proj = torch.einsum("bhnd,hdf->bhnf", K_norm, rff_proj_f).clamp(max=max_clamp)
    Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
    K_feat = torch.exp(K_proj) / math.sqrt(F_dim)

    KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V_f)
    K_sum = K_feat.sum(dim=2)
    denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)
    ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / (denom + 1e-5)

    return torch.nan_to_num(ctx, nan=0.0).to(orig_dtype)


def causal_kernel_linear_attention(
    Q_feat: torch.Tensor,
    K_feat: torch.Tensor,
    V: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    V5-FIX-03: causal kernel linear attention using chunked cumsum.

    Inputs:
        Q_feat: (B, H, N, F) — query features (after exp)
        K_feat: (B, H, N, F) — key features (after exp)
        V:      (B, H, N, D) — values
    Output:
        context: (B, H, N, D)

    Memory: O(B*H*chunk_size*F*D) at peak, vs O(B*H*N*F*D) for the naive version.
    Time:   O(N * F * D) forward.
    """
    B, H, N, F_dim = Q_feat.shape
    N_kv = K_feat.shape[2]
    past_len = max(0, N_kv - N)
    D = V.shape[-1]
    device = Q_feat.device
    dtype = Q_feat.dtype

    state_kv = torch.zeros(B, H, F_dim, D, device=device, dtype=dtype)
    state_k = torch.zeros(B, H, F_dim, device=device, dtype=dtype)

    if past_len > 0:
        for p_start in range(0, past_len, chunk_size):
            p_end = min(p_start + chunk_size, past_len)
            Kp = K_feat[:, :, p_start:p_end]
            Vp = V[:, :, p_start:p_end]
            state_kv = state_kv + torch.matmul(Kp.transpose(-2, -1), Vp)
            state_k = state_k + Kp.sum(dim=2)

    outputs = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Qc = Q_feat[:, :, start:end]
        Kc = K_feat[:, :, past_len + start : past_len + end]
        Vc = V[:, :, past_len + start : past_len + end]

        intra_scores = torch.matmul(Qc, Kc.transpose(-2, -1)).tril()
        if start == 0 and past_len == 0:
            num = torch.matmul(intra_scores, Vc)
            den = intra_scores.sum(dim=-1).clamp(min=1e-6)
        else:
            num = torch.matmul(Qc, state_kv) + torch.matmul(intra_scores, Vc)
            den = (torch.matmul(Qc, state_k.unsqueeze(-1)).squeeze(-1) + intra_scores.sum(dim=-1)).clamp(min=1e-6)
        outputs.append(num / den.unsqueeze(-1))

        if end < N:
            state_kv = state_kv + torch.matmul(Kc.transpose(-2, -1), Vc)
            state_k = state_k + Kc.sum(dim=2)

    out = torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
    return torch.nan_to_num(out, nan=0.0)
