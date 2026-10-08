"""Quantized attention routines: FP8 and INT8 attention with scaling factors."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

__all__ = [
    "fp8_scaled_attention",
    "int8_scaled_attention",
    "quantize_to_int8",
    "dequantize_from_int8",
]


def quantize_to_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor INT8 quantization: returns (q_x, scale)."""
    max_val = torch.max(torch.abs(x))
    scale = (max_val / 127.0).clamp(min=1e-8)
    q_x = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return q_x, scale


def dequantize_from_int8(q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantizes INT8 tensor back to float."""
    return q_x.to(scale.dtype) * scale


def int8_scaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_factor: float | None = None,
) -> torch.Tensor:
    """Attention execution simulating INT8 tensor core quantization for Q, K, V."""
    if scale_factor is None:
        scale_factor = 1.0 / math.sqrt(q.shape[-1])

    q_int, s_q = quantize_to_int8(q)
    k_int, s_k = quantize_to_int8(k)
    v_int, s_v = quantize_to_int8(v)

    q_deq = dequantize_from_int8(q_int, s_q)
    k_deq = dequantize_from_int8(k_int, s_k)
    v_deq = dequantize_from_int8(v_int, s_v)

    scores = torch.matmul(q_deq, k_deq.transpose(-2, -1)) * scale_factor
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v_deq)


def fp8_scaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_factor: float | None = None,
) -> torch.Tensor:
    """Simulated FP8 (E4M3 / E5M2) scaled attention with hardware-friendly range scaling."""
    if scale_factor is None:
        scale_factor = 1.0 / math.sqrt(q.shape[-1])

    # If torch supports float8 types on device, cast; otherwise clamp to simulated E4M3 range (~448.0)
    has_fp8 = hasattr(torch, "float8_e4m3fn")
    if has_fp8:
        try:
            q_fp8 = q.to(torch.float8_e4m3fn).to(q.dtype)
            k_fp8 = k.to(torch.float8_e4m3fn).to(k.dtype)
            v_fp8 = v.to(torch.float8_e4m3fn).to(v.dtype)
        except Exception:
            q_fp8 = q.clamp(-448.0, 448.0)
            k_fp8 = k.clamp(-448.0, 448.0)
            v_fp8 = v.clamp(-448.0, 448.0)
    else:
        q_fp8 = q.clamp(-448.0, 448.0)
        k_fp8 = k.clamp(-448.0, 448.0)
        v_fp8 = v.clamp(-448.0, 448.0)

    scores = torch.matmul(q_fp8, k_fp8.transpose(-2, -1)) * scale_factor
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v_fp8)
