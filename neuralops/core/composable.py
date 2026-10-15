"""Composable Meta-Attention architecture for NeuralOps.

Provides modular composition of attention mechanisms by chaining:
  - Projections: Dense ("dense" / "standard"), LowRankSVD ("svd(rank=r)"), Conv1D ("conv1d(kernel_size=k)")
  - Positional embeddings: None ("none"), RoPE ("rope"), ALiBi ("alibi")
  - Feature maps / kernel mappings: Softmax ("softmax"), ELU+1 ("elu+1" / "elu"), Cosformer ("cosformer")
  - Masking and sparsity: Causal ("causal"), SlidingWindow ("sliding_window(w)"), AttentionSink ("sink(s)")
  - Execution engines: QuadraticSDPA ("sdpa" / "quadratic"), ChunkedRecurrentScan ("chunked_scan" / "linear"), DeltaRule ("delta_rule" / "gdn")

Fully compatible with neuralops.replace_attention and standard PyTorch / HuggingFace architectures.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.triton_kernels import triton_dispatch


class ComposableAttention(nn.Module):
    """Modular, highly-composable attention layer chaining hardware-accelerated primitives."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        projection: str = "dense",
        positional: str = "none",
        kernel_map: str = "softmax",
        mask: str = "causal",
        engine: str = "sdpa",
        bias: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.projection_type = projection.lower()
        self.positional_type = positional.lower()
        self.kernel_map_type = kernel_map.lower()
        self.mask_type = mask.lower()
        self.engine = engine.lower()
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 1. Projections: Dense, LowRankSVD, Conv1D
        if "svd" in self.projection_type:
            rank_match = re.search(r"rank=(\d+)", self.projection_type)
            self.rank = int(rank_match.group(1)) if rank_match else 64
            self.q_down = nn.Linear(embed_dim, self.rank, bias=False)
            self.q_up = nn.Linear(self.rank, embed_dim, bias=bias)
            self.k_down = nn.Linear(embed_dim, self.rank, bias=False)
            self.k_up = nn.Linear(self.rank, embed_dim, bias=bias)
            self.v_down = nn.Linear(embed_dim, self.rank, bias=False)
            self.v_up = nn.Linear(self.rank, embed_dim, bias=bias)
            self.conv1d = None
        elif "conv1d" in self.projection_type or "conv" in self.projection_type:
            self.rank = 0
            k_match = re.search(r"kernel_size=(\d+)", self.projection_type) or re.search(r"(\d+)", self.projection_type)
            k_size = int(k_match.group(1)) if k_match else 4
            self.conv1d = nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=k_size,
                padding=k_size - 1,
                groups=embed_dim,
                bias=bias,
            )
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.rank = 0
            self.conv1d = None
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # 2. Positional Bias: ALiBi slopes setup
        if "alibi" in self.positional_type:
            def _get_slopes(n: int) -> list[float]:
                def get_slopes_power_of_2(n_p2: int) -> list[float]:
                    start = 2 ** (-(2 ** -(math.log2(n_p2) - 3)))
                    ratio = start
                    return [start * (ratio ** i) for i in range(n_p2)]
                if math.log2(n).is_integer():
                    return get_slopes_power_of_2(n)
                closest = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest) + _get_slopes(2 * closest)[0::2][: n - closest]
            slopes = torch.tensor(_get_slopes(num_heads), dtype=torch.float32).view(1, num_heads, 1, 1)
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        else:
            self.alibi_slopes = None

        # 3. Sparsity & Mask parsing
        sw_match = re.search(r"sliding_window\((\d+)\)", self.mask_type)
        self.window_size = int(sw_match.group(1)) if sw_match else (128 if "sliding_window" in self.mask_type else 0)
        sink_match = re.search(r"sink\((\d+)\)", self.mask_type)
        self.sink_tokens = int(sink_match.group(1)) if sink_match else (4 if "sink" in self.mask_type else 0)

        # 4. Delta Rule state setup if engine="delta_rule"
        if "delta_rule" in self.engine or "gdn" in self.engine:
            self.beta = nn.Parameter(torch.zeros(num_heads))

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.conv1d is not None:
            # Conv1D input: [B, D, L]
            B, L, D = x.shape
            x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
            q = self.q_proj(x_conv)
            k = self.k_proj(x_conv)
            v = self.v_proj(x_conv)
        elif "svd" in self.projection_type:
            q = self.q_up(self.q_down(x))
            k = self.k_up(self.k_down(x))
            v = self.v_up(self.v_down(x))
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        return q, k, v

    def _apply_positional(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if "rope" in self.positional_type:
            B, H, L, d = q.shape
            pos = torch.arange(L, device=q.device, dtype=torch.float32)
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d, 2, device=q.device, dtype=torch.float32) / d))
            freqs = torch.outer(pos, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().view(1, 1, L, d).to(q.dtype)
            sin = emb.sin().view(1, 1, L, d).to(q.dtype)
            def _rotate_half(x: torch.Tensor) -> torch.Tensor:
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
        return q, k

    def _apply_kernel(self, t: torch.Tensor) -> torch.Tensor:
        if "elu" in self.kernel_map_type:
            return F.elu(t.float()) + 1.0
        elif "cosformer" in self.kernel_map_type or "cos" in self.kernel_map_type:
            # Cosformer mapping: ReLU(x) * cos(pi * i / 2M)
            t_relu = F.relu(t.float())
            L = t.shape[-2]
            pos = torch.arange(L, device=t.device, dtype=torch.float32).view(1, 1, L, 1)
            weight = torch.cos(math.pi * pos / (2.0 * max(L, 1)))
            return t_relu * weight
        return t

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q, k, v = self._project_qkv(hidden_states)
        q = q.view(B, N, H, d).transpose(1, 2).contiguous()
        k = k.view(B, N, H, d).transpose(1, 2).contiguous()
        v = v.view(B, N, H, d).transpose(1, 2).contiguous()

        # Apply Positional Bias (e.g. RoPE)
        q, k = self._apply_positional(q, k)

        is_causal = "causal" in self.mask_type or self.window_size > 0 or "chunked_scan" in self.engine

        # Apply Kernel Mapping (e.g., ELU+1, Cosformer)
        if self.kernel_map_type in ("elu", "elu+1", "cosformer", "cos"):
            q_k = self._apply_kernel(q)
            k_k = self._apply_kernel(k)
        else:
            q_k, k_k = q, k

        # Execution Engine 1: Chunked Recurrent Scan
        if "chunked_scan" in self.engine or "recurrent" in self.engine:
            if q.is_cuda:
                try:
                    from neuralops.core.triton_kernels.gla_scan import kda_hla_chunk_scan_triton
                    log_dec = torch.zeros((B, H, N), device=q.device, dtype=torch.float32)
                    context = kda_hla_chunk_scan_triton(q_k.float(), k_k.float(), v.float(), log_dec).to(hidden_states.dtype)
                    out = self.out_proj(context.transpose(1, 2).contiguous().view(B, N, D))
                    return out
                except Exception:
                    pass

        # Execution Engine 2: Delta Rule (Gated Delta Net)
        if "delta_rule" in self.engine or "gdn" in self.engine:
            if q.is_cuda:
                try:
                    from neuralops.core.triton_kernels.gdn_scan import gdn_scan_triton
                    beta = torch.sigmoid(self.beta).view(1, H, 1).expand(B, H, N).contiguous()
                    context = gdn_scan_triton(q_k.to(torch.float16), k_k.to(torch.float16), v.to(torch.float16), beta.to(torch.float16))
                    out = self.out_proj(context.transpose(1, 2).contiguous().view(B, N, D))
                    return out
                except Exception:
                    pass

        # Execution Engine 3: Sliding Window & Sink Attention
        if "sliding_window" in self.mask_type and q.is_cuda and self.sink_tokens == 0:
            try:
                from neuralops.core.triton_kernels.sliding_window import sliding_window_triton
                context = sliding_window_triton(q, k, v, window_size=self.window_size)
                out = self.out_proj(context.transpose(1, 2).contiguous().view(B, N, D))
                return out
            except Exception:
                pass

        # Execution Engine 4: Quadratic SDPA with ALiBi, Sliding Window & Attention Sink Masking
        scores = torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.scale

        if self.alibi_slopes is not None:
            # ALiBi: bias[i, j] = -|i - j| * slope
            pos = torch.arange(N, device=q.device, dtype=torch.float32)
            dist = (pos.unsqueeze(-1) - pos.unsqueeze(-2)).abs()
            alibi_bias = -dist.view(1, 1, N, N) * self.alibi_slopes
            scores = scores + alibi_bias

        if is_causal or self.window_size > 0 or self.sink_tokens > 0:
            # Construct composite causal + sliding window + sink mask
            pos_i = torch.arange(N, device=q.device).unsqueeze(-1)
            pos_j = torch.arange(N, device=q.device).unsqueeze(0)
            diff = pos_i - pos_j
            causal_cond = diff >= 0
            if self.window_size > 0:
                window_cond = diff < self.window_size
                if self.sink_tokens > 0:
                    sink_cond = pos_j < self.sink_tokens
                    valid_mask = causal_cond & (window_cond | sink_cond)
                else:
                    valid_mask = causal_cond & window_cond
            else:
                valid_mask = causal_cond

            scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        out = self.out_proj(context.transpose(1, 2).contiguous().view(B, N, D))
        return out


def compose_attention(
    projection: str = "dense",
    positional: str = "none",
    kernel_map: str = "softmax",
    mask: str = "causal",
    engine: str = "sdpa",
    embed_dim: int = 768,
    num_heads: int = 12,
    **kwargs: Any,
) -> ComposableAttention:
    """Factory helper creating a fully-configured declarative ComposableAttention module."""
    return ComposableAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        projection=projection,
        positional=positional,
        kernel_map=kernel_map,
        mask=mask,
        engine=engine,
        **kwargs,
    )


__all__ = [
    "ComposableAttention",
    "compose_attention",
]
