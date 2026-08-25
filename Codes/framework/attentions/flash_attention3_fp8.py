r"""
FlashAttention-3 FP8: Low-Precision 8-bit Float Transformer Attention.

Implements FlashAttention-3 utilizing FP8 (e4m3fn) precision format for Hopper Tensor Cores,
reducing KV memory bandwidth and compute latency while maintaining accuracy.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
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


@ATTENTION_REGISTRY.register("flash3_fp8")
@ATTENTION_REGISTRY.register("flash_attention3_fp8")
class FlashAttention3FP8(BaseAttention):
    r"""
    FlashAttention-3 FP8 Hardware-Accelerated Multi-Head Attention.

    Mathematical Formulation:
        1. FP8 Quantization:
           $$Q_{\text{fp8}} = \operatorname{cast}_{\text{fp8}}(Q), \quad K_{\text{fp8}} = \operatorname{cast}_{\text{fp8}}(K), \quad V_{\text{fp8}} = \operatorname{cast}_{\text{fp8}}(V)$$
        2. Exact Tiled Kernel Reduction:
           $$O = \text{FlashAttn3}(Q_{\text{fp8}}, K_{\text{fp8}}, V_{\text{fp8}})$$
        3. Output Projection:
           $$\text{Output} = \operatorname{cast}_{\text{dtype}}(O) W_{\text{out}}^\top + b_{\text{out}}$$

    Complexity:
        - Compute: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ with $2\times$ FP8 Tensor Core throughput.
        - Memory Bandwidth: $50\%$ reduction in SRAM-HBM traffic compared to FP16/BF16.
    """

    def __init__(
        self,
        config: Any = None,
        rank: Optional[int] = None,
        d_model: Union[int, Any] = 768,
        n_heads: int = 12,
        use_fp8: bool = True,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if config is not None and hasattr(config, "hidden_size"):
            d_model = config.hidden_size
            n_heads = config.num_attention_heads

        self.num_heads: int = n_heads
        self.head_dim: int = d_model // n_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.use_fp8: bool = use_fp8

        self.q_proj = nn.Linear(d_model, self.all_head_size)
        self.k_proj = nn.Linear(d_model, self.all_head_size)
        self.v_proj = nn.Linear(d_model, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, d_model)

        dropout_p = (
            getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                ),
            )
            if config is not None
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        r"""Universal adapter weight copying."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.q_proj(hidden_states).view(B, L, H, Dh)
        k = self.k_proj(hidden_states).view(B, L, H, Dh)
        v = self.v_proj(hidden_states).view(B, L, H, Dh)

        if FLASH_ATTN_AVAILABLE and self.use_fp8 and q.is_cuda and hasattr(torch, "float8_e4m3fn"):
            q_fp8 = q.to(torch.float8_e4m3fn)
            k_fp8 = k.to(torch.float8_e4m3fn)
            v_fp8 = v.to(torch.float8_e4m3fn)
            out = flash_attn_func(q_fp8, k_fp8, v_fp8, causal=False).to(dtype)
            out = out.contiguous().view(B, L, D)
        else:
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)

            valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)
            attn_mask = (1.0 - valid_mask_2d.unsqueeze(1).unsqueeze(2).to(q_sdpa.dtype)) * -10000.0

            out_sdpa = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, is_causal=False
            )
            out = out_sdpa.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out), None


@ATTENTION_REGISTRY.register("flash3_fp8_inject")
def inject_flash3_fp8(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> FlashAttention3FP8:
    use_fp8 = getattr(comp_config, "use_fp8", True)
    new_attn = FlashAttention3FP8(
        config=model_config,
        use_fp8=use_fp8,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
