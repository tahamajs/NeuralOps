import math
import torch
import torch.nn.functional as F

def _local_window_attention_v2(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    token_valid: torch.Tensor | None = None,
    is_decoder: bool = False,
    past_len: int = 0,
) -> torch.Tensor:
    """
    V2 local attention primitive.

    V2-FIX-03: past_len shifts query positions to absolute indices.
    window here is the FULL window width (not radius); radius = window // 2.

    Args:
        Q: (B, H, N, D)
        K: (B, H, N_kv, D)
        V: (B, H, N_kv, D)
        token_valid: (B, N_kv) boolean padding mask
        past_len: cached tokens (for KV-cache decode)
    """
    B, H, N, D = Q.shape
    N_k = K.shape[2]
    radius = window // 2
    scale = 1.0 / math.sqrt(max(int(D), 1))

    if Q.is_cuda and token_valid is None and past_len == 0 and N == N_k and D <= 256:
        try:
            from neuralops.ops.cuda.hybrid_local_kernel import hybrid_local_attn

            return hybrid_local_attn(Q, K, V, radius=radius, is_causal=is_decoder)
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

    chunk_size = 64
    mask_val = -1e4
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
        band = ((abs_q - abs_k).abs() <= radius).unsqueeze(0).unsqueeze(0)
        if is_decoder:
            band = band & (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(Qc.float(), Kc.float().transpose(-2, -1)) * scale
        scores = scores.masked_fill(~band, mask_val)
        if token_valid is not None:
            kv_valid = token_valid[:, None, None, k_start:k_end].to(torch.bool)
            scores = scores.masked_fill(~kv_valid, mask_val)
            combined_valid = band & kv_valid
        else:
            combined_valid = band
        scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e4, neginf=mask_val)
        any_valid = combined_valid.any(dim=-1, keepdim=True).to(scores.dtype)
        attn = F.softmax(scores, dim=-1).to(Q.dtype) * any_valid
        outputs.append(torch.matmul(attn, Vc.to(attn.dtype)))

    return torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]


def _phi_elu(x: torch.Tensor) -> torch.Tensor:
    """V2 kernel feature map: ELU + 1 (Katharopoulos et al.)."""
    return F.elu(x) + 1.0


def _global_linear_attention_v2(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    token_valid: torch.Tensor | None,
    is_decoder: bool = False,
) -> torch.Tensor:
    """
    Non-causal or causal ELU+1 kernel linear attention.

    V2-FIX-08: adds a causal path for decoders so the global branch is
    meaningful in GPT-2 / Qwen and does not leak future tokens.
    """
    B, H, N, D = Q.shape
    N_k = K.shape[2]

    phi_Q = _phi_elu(Q.float())
    phi_K = _phi_elu(K.float())
    V_f = V.float()

    if token_valid is not None:
        mask_kv = token_valid[:, None, :, None].to(phi_K.dtype)
        phi_K = phi_K * mask_kv
        V_f = V_f * mask_kv

    if is_decoder:
        if N_k != N:
            KV = torch.matmul(phi_K.transpose(-2, -1).contiguous(), V_f)
            Z = phi_K.sum(dim=-2, keepdim=True)
            num = torch.matmul(phi_Q, KV)
            den = torch.matmul(phi_Q, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6)
            return torch.nan_to_num(num / den, nan=0.0).to(Q.dtype)

        from neuralops.ops.triton._types import FORCE_PYTORCH_FALLBACK, HAS_TRITON

        if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and Q.is_cuda and token_valid is None and D <= 128:
            try:
                from neuralops.ops.triton.linear_attn import fused_svd_linear_chunk_scan_triton

                return fused_svd_linear_chunk_scan_triton(
                    phi_Q.to(Q.dtype), phi_K.to(Q.dtype), V, chunk_size=64, eps=1e-6
                )
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass

        chunk = 64
        state_kv = torch.zeros(B, H, phi_Q.shape[-1], D, device=Q.device, dtype=phi_Q.dtype)
        state_k = torch.zeros(B, H, phi_Q.shape[-1], device=Q.device, dtype=phi_Q.dtype)
        outs = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            Qc = phi_Q[:, :, start:end]
            Kc = phi_K[:, :, start:end]
            Vc = V_f[:, :, start:end]

            intra_scores = torch.matmul(Qc, Kc.transpose(-2, -1)).tril()
            if start == 0:
                num = torch.matmul(intra_scores, Vc)
                den = intra_scores.sum(dim=-1).clamp(min=1e-6)
            else:
                num = torch.matmul(Qc, state_kv) + torch.matmul(intra_scores, Vc)
                den = (torch.matmul(Qc, state_k.unsqueeze(-1)).squeeze(-1) + intra_scores.sum(dim=-1)).clamp(min=1e-6)
            outs.append(num / den.unsqueeze(-1))

            if end < N:
                state_kv = state_kv + torch.matmul(Kc.transpose(-2, -1), Vc)
                state_k = state_k + Kc.sum(dim=2)

        out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        return torch.nan_to_num(out, nan=0.0).to(Q.dtype)

    KV = torch.matmul(phi_K.transpose(-2, -1).contiguous(), V_f)
    Z = phi_K.sum(dim=-2, keepdim=True)
    num = torch.matmul(phi_Q, KV)
    den = torch.matmul(phi_Q, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6)
    return torch.nan_to_num(num / den, nan=0.0).to(Q.dtype)
