r"""
FlashAttention-3: Fast and Memory-Efficient Exact Attention with Asynchronous Hardware Overlap.

Integrates FlashAttention-3 kernels (with Hopper TMA asynchronous memory copy and FP8 precision support)
alongside graceful fallback to PyTorch's hardware-accelerated Scaled Dot-Product Attention (SDPA).
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


@ATTENTION_REGISTRY.register("flash3")
class FlashAttention3(BaseAttention):
    r"""
    FlashAttention-3 Multi-Head Exact Attention.

    Mathematical Formulation:
        Computes standard scaled dot-product attention without materializing intermediate attention matrices in High Bandwidth Memory (HBM):
        $$S = \frac{Q K^T}{\sqrt{d_h}} + M$$
        $$P = \text{Softmax}(S) = \frac{\exp(S - m)}{\sum \exp(S - m)}$$
        $$O = P V$$
        FlashAttention-3 tiles $Q, K, V$ into SRAM blocks and utilizes asynchronous hardware copy (Warp Specialization & TMA)
        to overlap GEMM operations with softmax reduction.

    Complexity:
        - FLOPs: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$.
        - HBM Memory Accesses: $\mathcal{O}\left(B \cdot H \cdot L \cdot d_h + \frac{L^2 d_h^2}{M_{\text{SRAM}}}\right)$ (IO-optimal).
        - Memory Footprint: $\mathcal{O}(B \cdot L \cdot D)$ activations in global memory.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, L, D]$
        - FlashAttention Layout: $[B, L, H, d_h]$
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        use_fp8: bool = False,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.use_fp8: bool = use_fp8

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for FlashAttention3.

        Args:
            hidden_states (torch.Tensor): Input activations $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor $[B, L]$ or $[B, 1, 1, L]$.
            output_attentions (bool): Whether to return attention probabilities (unsupported in FlashAttention kernels).
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed context representation $[B, L, D]$.
                - None.
        """
        B, L, D = hidden_states.shape
        H, D_head = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, D_head)  # (B, L, H, D_head)
        k = self.k_proj(hidden_states).view(B, L, H, D_head)
        v = self.v_proj(hidden_states).view(B, L, H, D_head)

        if FLASH_ATTN_AVAILABLE and hidden_states.is_cuda:
            q_fa = q.contiguous()
            k_fa = k.contiguous()
            v_fa = v.contiguous()

            if self.use_fp8 and hasattr(torch, "float8_e4m3fn"):
                q_fa = q_fa.to(torch.float8_e4m3fn)
                k_fa = k_fa.to(torch.float8_e4m3fn)
                v_fa = v_fa.to(torch.float8_e4m3fn)

            out = flash_attn_func(q_fa, k_fa, v_fa, causal=False)
            out = out.contiguous().view(B, L, D)
        else:
            q_sdpa = q.transpose(1, 2)  # (B, H, L, D_head)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)

            attn_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attn_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(q_sdpa.dtype)) * -1e9
                elif attention_mask.dim() == 4:
                    attn_mask = attention_mask.to(q_sdpa.dtype)

            out_sdpa = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                is_causal=False,
            )
            out = out_sdpa.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out), None
