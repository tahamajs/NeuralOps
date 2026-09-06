r"""
EAT: Efficient Attention Transformer with Progressive Token Pruning.

Implements attention-level efficient multi-head self-attention with optional
progressive token pruning and zero-shot weight inheritance from pretrained models.
"""

from typing import Any, Optional, Tuple, Union

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY
from .progressive_token_pruning import ProgressiveTokenPruning


@ATTENTION_REGISTRY.register("eat")
class EATBlock(nn.Module):
    r"""
    Efficient Attention Transformer (EAT) Attention Module with Progressive Token Pruning.
    """

    def __init__(
        self,
        d_model: Union[int, Any] = 768,
        n_heads: int = 12,
        prune_layer: Optional[ProgressiveTokenPruning] = None,
        config: Optional[Any] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", 768))
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
        elif hasattr(d_model, "hidden_size"):
            config = d_model
            d_model = config.hidden_size
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
        elif pretrained_attn is not None:
            d_model = getattr(pretrained_attn, "hidden_size", getattr(pretrained_attn, "embed_dim", getattr(pretrained_attn, "all_head_size", 768)))
            if d_model == 768 and hasattr(pretrained_attn, "q_proj") and hasattr(pretrained_attn.q_proj, "in_features"):
                d_model = pretrained_attn.q_proj.in_features
            n_heads = getattr(pretrained_attn, "num_heads", getattr(pretrained_attn, "num_attention_heads", 12))

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        if self.d_model % self.n_heads != 0:
            divisors = [h for h in range(1, self.d_model + 1) if self.d_model % h == 0]
            valid = [h for h in divisors if h <= self.n_heads]
            self.n_heads = valid[-1] if valid else min(divisors, key=lambda h: abs(h - self.n_heads))
        self.head_dim: int = self.d_model // max(1, self.n_heads)

        self.q_lin = nn.Linear(self.d_model, self.d_model)
        self.k_lin = nn.Linear(self.d_model, self.d_model)
        self.v_lin = nn.Linear(self.d_model, self.d_model)
        self.out_lin = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1) if config is not None else 0.1)
        self.prune_layer: Optional[ProgressiveTokenPruning] = prune_layer

        if pretrained_attn is not None:
            self._copy_pretrained(pretrained_attn)

    def _copy_pretrained(self, old_attn: nn.Module) -> None:
        def _get_layer(src, names):
            for name in names:
                if hasattr(src, name):
                    mod = getattr(src, name)
                    if hasattr(mod, "dense"):
                        mod = getattr(mod, "dense")
                    if hasattr(mod, "weight"):
                        return mod
            return None

        q_src = _get_layer(old_attn, ("q_lin", "query", "q_proj", "q"))
        k_src = _get_layer(old_attn, ("k_lin", "key", "k_proj", "k"))
        v_src = _get_layer(old_attn, ("v_lin", "value", "v_proj", "v"))
        out_src = _get_layer(old_attn, ("out_lin", "dense", "out_proj", "o_proj", "output", "proj", "o"))

        for src_mod, target_mod in [(q_src, self.q_lin), (k_src, self.k_lin), (v_src, self.v_lin), (out_src, self.out_lin)]:
            if src_mod is not None:
                w = src_mod.weight.data
                if w.shape == target_mod.weight.shape:
                    target_mod.weight.data.copy_(w)
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    target_mod.weight.data[:min_0, :min_1].copy_(w[:min_0, :min_1])
                if src_mod.bias is not None and target_mod.bias is not None:
                    b = src_mod.bias.data
                    if b.shape == target_mod.bias.shape:
                        target_mod.bias.data.copy_(b)
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.data[:min_b].copy_(b[:min_b])

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "EATBlock":
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

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
        B, N, D = hidden_states.shape

        n_heads = self.n_heads
        head_dim = self.head_dim
        if (n_heads * head_dim) != self.d_model or D != self.d_model:
            dim_to_split = self.d_model if (n_heads * head_dim == self.d_model) else D
            divisors = [h for h in range(1, dim_to_split + 1) if dim_to_split % h == 0]
            valid = [h for h in divisors if h <= n_heads]
            n_heads = valid[-1] if valid else min(divisors, key=lambda h: abs(h - n_heads))
            head_dim = dim_to_split // n_heads

        q = self.q_lin(hidden_states).view(B, N, n_heads, head_dim).transpose(1, 2)
        k = self.k_lin(hidden_states).view(B, N, n_heads, head_dim).transpose(1, 2)
        v = self.v_lin(hidden_states).view(B, N, n_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask) * -10000.0
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(1)
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, N, n_heads * head_dim)
        out = self.out_lin(context)
        if out.shape[-1] != D:
            out = F.pad(out, (0, D - out.shape[-1])) if out.shape[-1] < D else out[..., :D]

        if self.prune_layer is not None and not self.training:
            out, _ = self.prune_layer(out)

        if output_attentions:
            return out, attn_weights
        return out, None


@ATTENTION_REGISTRY.register("eat_inject")
def inject_eat(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> EATBlock:
    r"""Injects EAT attention into a transformer layer."""
    src = old_attn
    if src is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                src = getattr(layer, attr)
                break
    if model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    new_attn = EATBlock(config=model_config, pretrained_attn=src, **kwargs)
    for attr in ("attention", "self_attn", "attn", "self"):
        if hasattr(layer, attr):
            setattr(layer, attr, new_attn)
            return new_attn
    return new_attn
