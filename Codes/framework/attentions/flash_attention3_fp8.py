r"""
FlashAttention-3 FP8: Low-Precision 8-bit Float Transformer Attention.

Implements FlashAttention-3 utilizing FP8 (e4m3fn) precision format for Hopper Tensor Cores,
reducing KV memory bandwidth and compute latency while maintaining accuracy.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY

try:
    from flash_attn_interface import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    try:
        from flash_attn import flash_attn_func

        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_AVAILABLE = False


class FlashAttention3FP8(nn.Module):
    r"""
    FlashAttention-3 FP8 Hardware-Accelerated Multi-Head Attention.

    Mathematical Formulation:
        Given linear projections $Q, K, V \in \mathbb{R}^{B \times L \times H \times d_h}$:
        1. FP8 Quantization:
           Casts activations to 8-bit floating point format $E4M3$:
           $$Q_{\text{fp8}} = \text{cast}_{\text{fp8}}(Q), \quad K_{\text{fp8}} = \text{cast}_{\text{fp8}}(K), \quad V_{\text{fp8}} = \text{cast}_{\text{fp8}}(V)$$
        2. Exact Tiled Kernel Reduction:
           $$O = \text{FlashAttn3}(Q_{\text{fp8}}, K_{\text{fp8}}, V_{\text{fp8}})$$
        3. Output Projection:
           $$\text{Output} = \text{cast}_{\text{dtype}}(O) W_{\text{out}}^T + b_{\text{out}}$$

    Complexity:
        - Compute: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ with $2\times$ FP8 Tensor Core throughput.
        - Memory Bandwidth: $50\%$ reduction in SRAM-HBM traffic compared to FP16/BF16.

    Tensor Dimensions:
        - Input: $[B, L, D]$
        - Quantized Matrices: $[B, L, H, d_h]$ in FP8
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        d_model: Union[int, Any] = 768,
        n_heads: int = 12,
        use_fp8: bool = True,
    ) -> None:
        super().__init__()
        if hasattr(d_model, "hidden_size"):
            config = d_model
            d_model = config.hidden_size
            n_heads = config.num_attention_heads

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.head_dim: int = d_model // n_heads
        self.use_fp8: bool = use_fp8

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for FlashAttention3FP8.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor of shape $[B, L]$ or $[B, 1, 1, L]$.
            output_attentions (bool): Flag indicating whether to return attention matrices.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output context representation $[B, L, D]$.
                - None.
        """
        x = hidden_states
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)

        if FLASH_ATTN_AVAILABLE and self.use_fp8 and q.is_cuda and hasattr(torch, "float8_e4m3fn"):
            q_fp8 = q.to(torch.float8_e4m3fn)
            k_fp8 = k.to(torch.float8_e4m3fn)
            v_fp8 = v.to(torch.float8_e4m3fn)
            out = flash_attn_func(q_fp8, k_fp8, v_fp8, causal=False).to(x.dtype)
            out = out.contiguous().view(B, L, D)
        else:
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            attn_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attn_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(q.dtype)) * -10000.0
                elif attention_mask.dim() == 4:
                    attn_mask = attention_mask.to(q.dtype)

            out_sdpa = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, is_causal=False
            )
            out = out_sdpa.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out), None


@ATTENTION_REGISTRY.register("flash3_fp8")
def inject_flash3_fp8(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> FlashAttention3FP8:
    r"""
    Injects FlashAttention-3 FP8 attention module into a transformer layer.

    Args:
        layer (nn.Module): Host layer module.
        model_config (Any): Model configuration parameters.
        comp_config (Any): Compression hyperparameter configuration.
        layer_idx (int): Current layer index.
        total_layers (int): Total transformer layers.
        old_attn (Optional[nn.Module]): Pretrained attention layer.

    Returns:
        FlashAttention3FP8: Initialized FP8 attention layer.
    """
    use_fp8 = getattr(comp_config, "use_fp8", True)
    new_attn = FlashAttention3FP8(
        d_model=model_config.hidden_size,
        n_heads=model_config.num_attention_heads,
        use_fp8=use_fp8,
    )
    layer.attention = new_attn
    return new_attn
