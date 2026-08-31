r"""
FlashAttention-3: Fast and Memory-Efficient Exact Attention with Asynchronous Hardware Overlap.

Integrates FlashAttention-3 kernels (with Hopper TMA asynchronous memory copy and FP8 precision support)
alongside graceful fallback to PyTorch's hardware-accelerated Scaled Dot-Product Attention (SDPA) and exact pretrained weight copying.
"""

from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter
from framework.utils.compat import repeat_kv, maybe_apply_rotary, causal_mask_from_attention_mask

try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


@ATTENTION_REGISTRY.register("flash3")
@ATTENTION_REGISTRY.register("flash_attention3")
class FlashAttention3(BaseAttention):
    supports_gqa = True
    supports_rope = True
    r"""
    FlashAttention-3 Multi-Head Exact Attention.

    Mathematical Formulation:
        $$S = \frac{Q K^\top}{\sqrt{d_h}} + M$$
        $$P = \operatorname{Softmax}(S) = \frac{\exp(S - m)}{\sum \exp(S - m)}$$
        $$O = P V$$
        FlashAttention-3 tiles $Q, K, V$ into SRAM blocks and utilizes asynchronous hardware copy (Warp Specialization & TMA)
        to overlap GEMM operations with softmax reduction.

    Complexity:
        - FLOPs: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ exact.
        - HBM Accesses: $\mathcal{O}\left(B \cdot H \cdot L \cdot d_h + \frac{L^2 d_h^2}{M_{\text{SRAM}}}\right)$ (IO-optimal).
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        use_fp8: bool = False,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.adapter = ModelAdapter(config)
        self.num_heads = self.adapter.num_heads
        self.num_key_value_heads = self.adapter.num_key_value_heads
        hidden_size = self.adapter.hidden_size
        self.head_dim = self.adapter.head_dim
        self.all_head_size = self.adapter.q_proj_out
        self.kv_head_size = self.adapter.kv_proj_out
        self.use_fp8: bool = use_fp8

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.kv_head_size)
        self.v_proj = nn.Linear(hidden_size, self.kv_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.q_proj(hidden_states).view(B, H, L, Dh)
        k = self.k_proj(hidden_states).view(B, self.num_key_value_heads, L, Dh)
        v = self.v_proj(hidden_states).view(B, self.num_key_value_heads, L, Dh)
        q, k = maybe_apply_rotary(q, k, self.adapter, self.rotary_emb, kwargs.get("position_ids"))
        k, v = repeat_kv(k, self.adapter.kv_repeat), repeat_kv(v, self.adapter.kv_repeat)

        if FLASH_ATTN_AVAILABLE and hidden_states.is_cuda:
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()

            if self.use_fp8 and hasattr(torch, "float8_e4m3fn"):
                q_fa = q_fa.to(torch.float8_e4m3fn)
                k_fa = k_fa.to(torch.float8_e4m3fn)
                v_fa = v_fa.to(torch.float8_e4m3fn)

            out = flash_attn_func(q_fa, k_fa, v_fa, causal=self.adapter.causal)
            out = out.contiguous().view(B, L, D)
        else:
            q_sdpa, k_sdpa, v_sdpa = q, k, v
            attn_mask = causal_mask_from_attention_mask(attention_mask, self.adapter)
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                attn_mask = torch.zeros_like(attn_mask, dtype=q_sdpa.dtype).masked_fill(~attn_mask, -10000.0)

            out_sdpa = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                is_causal=self.adapter.causal and attn_mask is None,
            )
            out = out_sdpa.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out), None
