"""Selective scan kernel implementation for Mamba-style state space models.

Provides fast CUDA kernel path when `mamba_ssm` is installed,
with a robust chunked pure-PyTorch implementation as fallback.
"""

from typing import Optional
import torch
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


def selective_scan_chunked(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    Selective scan implementation with chunked recurrence for linear-time complexity and numerical stability.
    """
    orig_dtype = u.dtype
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()

    if delta_bias is not None:
        delta = delta + delta_bias.float().unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)
    delta = torch.clamp(delta, min=1e-4, max=20.0)

    batch, dim, seq_len = u.shape
    d_state = A.shape[-1]

    # Precompute dA and dB_u
    dA = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
    dA = dA.clamp(max=0.0)
    dB_u = delta.unsqueeze(-1) * u.unsqueeze(-1) * B.unsqueeze(1).transpose(2, 3)

    y = torch.empty(batch, dim, seq_len, device=u.device, dtype=torch.float32)
    h_prev = torch.zeros(batch, dim, d_state, device=u.device, dtype=torch.float32)

    for s in range(0, seq_len, chunk_size):
        e = min(s + chunk_size, seq_len)
        C_len = e - s

        dA_c = dA[:, :, s:e, :]
        dBu_c = dB_u[:, :, s:e, :]
        C_c = C[:, :, s:e].transpose(1, 2)

        dA_cum = torch.cumsum(dA_c, dim=2)
        log_decay = dA_cum.unsqueeze(3) - dA_cum.unsqueeze(2)
        mask = torch.tril(torch.ones(C_len, C_len, dtype=torch.bool, device=u.device)).view(1, 1, C_len, C_len, 1)
        log_decay = log_decay.masked_fill(~mask, float("-inf"))
        decay = torch.exp(log_decay)
        decay = torch.nan_to_num(decay, nan=0.0)

        h_intra = torch.einsum("bdjin,bdin->bdjn", decay, dBu_c)
        h_carry = torch.exp(dA_cum) * h_prev.unsqueeze(2)
        h_chunk = h_intra + h_carry

        y_c = torch.einsum("bdcn,bcn->bdc", h_chunk, C_c)
        y[:, :, s:e] = y_c
        h_prev = h_chunk[:, :, -1, :]

    if D is not None:
        y = y + u * D.float().unsqueeze(0).unsqueeze(-1)

    if z is not None:
        y = y * F.silu(z.float())

    return y.to(orig_dtype)


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    Main entry point for selective scan with automatic dispatch to CUDA kernel if available.
    """
    if HAS_MAMBA and u.is_cuda:
        try:
            return selective_scan_fn(
                u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus
            )
        except Exception:
            pass

    return selective_scan_chunked(
        u=u,
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        chunk_size=chunk_size,
    )
