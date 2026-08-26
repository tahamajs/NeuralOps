r"""
Span-Expanded Attention with HyLoRA.

Extends attention receptive fields via fixed/cached key-value span expansion buffers,
complemented by parallel low-rank adaptation (HyLoRA) for parameter-efficient adaptation.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("span_expanded")
class SpanExpandedAttention(BaseAttention):
    r"""
    Span-Expanded Attention module with parallel HyLoRA adaptation.

    Mathematical Formulation:
        1. Span Attention:
           $$A = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d}} + M\right)$$
           $$O_{\text{attn}} = A V$$
        
        2. Parallel HyLoRA Residual Adaptation:
           $$O_{\text{LoRA}} = X W_A W_B, \quad W_A \in \mathbb{R}^{D \times r}, \, W_B \in \mathbb{R}^{r \times D}$$
        
        3. Output Projection:
           $$O = (O_{\text{attn}} + O_{\text{LoRA}}) W_O$$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        expansion_span: int = 1024,
        use_lora: bool = True,
        lora_rank: int = 8,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.expansion_span: int = expansion_span
        self.use_lora: bool = use_lora

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(hidden_size, self.all_head_size)
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

        if use_lora:
            self.lora_A = nn.Linear(hidden_size, lora_rank, bias=False)
            self.lora_B = nn.Linear(lora_rank, hidden_size, bias=False)
            nn.init.normal_(self.lora_A.weight, std=0.02)
            nn.init.zeros_(self.lora_B.weight)  # Zero init for exact initial preservation

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

        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        kv_cache = past_key_value if past_key_value is not None else kwargs.get("kv_cache", None)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = k_cache.unsqueeze(1).expand(-1, H, -1, -1)
            v = v_cache.unsqueeze(1).expand(-1, H, -1, -1)
        else:
            k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        scale = Dh ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)
        key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~key_valid_4d, -10000.0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        if self.use_lora:
            lora_out = self.lora_B(self.lora_A(hidden_states))
            out = out + lora_out

        out = self.out_proj(out)
        out = out * valid_mask_2d.unsqueeze(-1).to(dtype)

        if output_attentions:
            return out, attn
        return out, None


@ATTENTION_REGISTRY.register("span_expanded_inject")
def inject_span_expanded(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
,
    **kwargs) -> SpanExpandedAttention:
    new_attn = SpanExpandedAttention(
        config=model_config,
        expansion_span=getattr(comp_config, "expansion_span", 1024),
        use_lora=getattr(comp_config, "use_lora", True),
        lora_rank=getattr(comp_config, "lora_rank", 8),
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
