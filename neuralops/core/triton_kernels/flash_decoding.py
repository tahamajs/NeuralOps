"""Flash-Decoding attention implementation with chunked KV split and logsumexp reduction."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON

__all__ = ["flash_decoding_attention", "flash_decoding_pytorch"]


def flash_decoding_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Split-KV Flash-Decoding reference implementation.

    Splits the long KV sequence along sequence length into chunks, computes
    per-chunk local softmax and logsumexp, and merges across chunks using
    stable logsumexp normalization.

    Args:
        q: Query tensor of shape (B, H, 1, D) (decoding token) or (B, H, N_q, D).
        k: Key tensor of shape (B, H, N_kv, D).
        v: Value tensor of shape (B, H, N_kv, D).
        chunk_size: KV chunk size along sequence dimension.

    Returns:
        Output tensor of shape (B, H, N_q, D).
    """
    B, H, N_q, D = q.shape
    _, _, N_kv, _ = k.shape
    scale = 1.0 / math.sqrt(D)

    if N_kv <= chunk_size:
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    num_chunks = (N_kv + chunk_size - 1) // chunk_size
    chunk_outputs = []
    chunk_logsumexps = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, N_kv)
        k_chunk = k[:, :, start:end, :]
        v_chunk = v[:, :, start:end, :]

        scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale  # (B, H, N_q, chunk_len)
        m = scores.max(dim=-1, keepdim=True).values                  # (B, H, N_q, 1)
        p = torch.exp(scores - m)                                    # (B, H, N_q, chunk_len)
        l = p.sum(dim=-1, keepdim=True)                              # (B, H, N_q, 1)
        o = torch.matmul(p, v_chunk)                                 # (B, H, N_q, D)

        logsumexp = m + torch.log(l.clamp(min=1e-20))                # (B, H, N_q, 1)
        chunk_outputs.append(o)
        chunk_logsumexps.append(logsumexp)

    # Global logsumexp reduction across chunks
    all_lse = torch.cat(chunk_logsumexps, dim=-1)                    # (B, H, N_q, num_chunks)
    global_lse = torch.logsumexp(all_lse, dim=-1, keepdim=True)      # (B, H, N_q, 1)

    total_out = torch.zeros_like(q)
    for o, lse in zip(chunk_outputs, chunk_logsumexps):
        alpha = torch.exp(lse - global_lse)                          # (B, H, N_q, 1)
        total_out = total_out + alpha * (o / torch.exp(lse - lse.detach()).clamp(min=1e-20))

    # Standard weighted combination
    total_out = torch.zeros_like(q)
    for o, lse in zip(chunk_outputs, chunk_logsumexps):
        # o was sum(exp(scores - m) * v)
        # we need sum(exp(scores - global_lse) * v) = o * exp(m - global_lse)
        m_chunk = lse - torch.log(torch.exp(lse - lse).clamp(min=1.0))
        # More directly: exp(scores - global_lse) = exp(scores - m) * exp(m - global_lse)
        # logsumexp = m + log(l) => m = lse - log(l)
        # exp(scores - global_lse) = (p / l) * exp(lse - global_lse)
        # Thus total_chunk_contrib = (o / l) * exp(lse - global_lse)
        pass

    # Clean exact formula:
    res = torch.zeros_like(q)
    for i in range(num_chunks):
        o_i = chunk_outputs[i]
        lse_i = chunk_logsumexps[i]
        # lse_i = m_i + log(l_i) => exp(m_i - global_lse) * o_i
        # Since o_i = sum(exp(s - m_i) * v), exp(m_i - global_lse) * o_i = sum(exp(s - global_lse) * v)
        # exp(m_i - global_lse) = exp(lse_i - global_lse) / l_i
        weight = torch.exp(lse_i - global_lse)
        # wait, o_i / l_i is the local attention output! Let's verify:
        # local_out = o_i / l_i
        # global_out = sum_i( weight_i * local_out_i ) where weight_i = exp(lse_i) / sum(exp(lse_j)) = exp(lse_i - global_lse)
        # Indeed!
        # Let's compute local_out:
        p_sum = torch.exp(lse_i - lse_i.detach()) # placeholder, let's just do direct:
        pass

    # Exact, elegant recomputation:
    out = torch.zeros_like(q)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, N_kv)
        k_chunk = k[:, :, start:end, :]
        v_chunk = v[:, :, start:end, :]
        scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
        weights = torch.exp(scores - global_lse)
        out = out + torch.matmul(weights, v_chunk)

    return out


def flash_decoding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Dispatches flash decoding attention across chunks."""
    return flash_decoding_pytorch(q, k, v, chunk_size=chunk_size)
