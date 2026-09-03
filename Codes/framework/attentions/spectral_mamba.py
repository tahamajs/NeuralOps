r"""
Spectral-Mamba Attention Module: Hybrid Attention-State Space Module with Spectral Channel Modulation.

Combines pretrained multi-head attention with a Spectral Mamba SSM branch via learned adaptive gating.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

try:
    from framework.attentions.Mamba import Mamba
except Exception:
    class Mamba(nn.Module):
        def __init__(self, d_model: int = 768, **kwargs: Any) -> None:
            super().__init__()
            self.in_proj = nn.Linear(d_model, 2 * d_model)
            self.out_proj = nn.Linear(d_model, d_model)
        def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            x1, x2 = self.in_proj(x).chunk(2, dim=-1)
            y = F.silu(x1) * x2
            return self.out_proj(y), getattr(self, '_attn_weights', None)


@dataclass
class SpectralMambaConfig:
    r"""Configuration parameters for SpectralMamba."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    gate_init_bias: float = 1.8


class SpectralMamba(nn.Module):
    r"""Spectral Mamba layer with learnable channel-wise gating."""

    def __init__(self, d_model: int = 768, d_state: int = 16, **kwargs: Any) -> None:
        super().__init__()
        self.d_model = d_model
        self.mamba = Mamba(d_model)
        self.dt_scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mamba(x)
        if isinstance(out, tuple):
            out = out[0]
        gate = torch.sigmoid(self.dt_scale).unsqueeze(0).unsqueeze(0)
        return out * gate, getattr(self, '_attn_weights', None)


@ATTENTION_REGISTRY.register("spectral_mamba")
@ATTENTION_REGISTRY.register("spectral_mamba_inject")
class SpectralMambaAttention(BaseAttention):
    r"""
    Hybrid Spectral Mamba Attention Layer.
    Combines pretrained Multihead Attention with Spectral Mamba State Space Model.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(config, SpectralMambaConfig):
            h_size = getattr(config, "hidden_size", 768) if config is not None else 768
            n_heads = getattr(config, "num_attention_heads", 4) if config is not None else 12
            config = SpectralMambaConfig(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                head_dim=h_size // max(1, n_heads),
                dropout=getattr(config, "dropout", 0.1) if config is not None else 0.1,
                gate_init_bias=getattr(config, "gate_init_bias", 1.8) if config is not None else 1.8,
            )
        super().__init__(config)
        self.cfg: SpectralMambaConfig = config
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim

        self.q_lin = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_lin = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_lin = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_lin = nn.Linear(self.hidden_size, self.hidden_size)

        self.spectral_mamba = SpectralMamba(self.hidden_size, d_state=config.d_state)
        self.gate = nn.Linear(self.hidden_size, 1)
        self.gate.bias.data.fill_(config.gate_init_bias)
        self.dropout = nn.Dropout(config.dropout)

        if pretrained_attn is not None:
            self._copy_pretrained(pretrained_attn)

    def _copy_pretrained(self, old_attn: nn.Module) -> None:
        q_src = getattr(old_attn, "q_lin", getattr(old_attn, "query", None))
        k_src = getattr(old_attn, "k_lin", getattr(old_attn, "key", None))
        v_src = getattr(old_attn, "v_lin", getattr(old_attn, "value", None))
        out_src = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None))
        if q_src is not None:
            self.q_lin.weight.data.copy_(q_src.weight.data)
            if q_src.bias is not None:
                self.q_lin.bias.data.copy_(q_src.bias.data)
        if k_src is not None:
            self.k_lin.weight.data.copy_(k_src.weight.data)
            if k_src.bias is not None:
                self.k_lin.bias.data.copy_(k_src.bias.data)
        if v_src is not None:
            self.v_lin.weight.data.copy_(v_src.weight.data)
            if v_src.bias is not None:
                self.v_lin.bias.data.copy_(v_src.bias.data)
        if out_src is not None:
            self.out_lin.weight.data.copy_(out_src.weight.data)
            if out_src.bias is not None:
                self.out_lin.bias.data.copy_(out_src.bias.data)

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "SpectralMambaAttention":
        r"""Instantiates SpectralMambaAttention with weights transferred from pretrained attention."""
        if model_config is None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None)).out_features
            model_config.num_attention_heads = getattr(old_attn, "n_heads", getattr(old_attn, "num_attention_heads", 4))
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

        q = self.q_lin(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k   = self.k_lin(hidden_states).view(B, N, getattr(self, "num_kv_heads", getattr(self, "num_kv_heads", self.num_heads)), self.head_dim).transpose(1, 2)
        k = repeat_kv(k, getattr(self, 'kv_repeat', 1))
        k = repeat_kv(k, getattr(self, 'kv_repeat', 1))
        v   = self.v_lin(hidden_states).view(B, N, getattr(self, "num_kv_heads", getattr(self, "num_kv_heads", self.num_heads)), self.head_dim).transpose(1, 2)
        v = repeat_kv(v, getattr(self, 'kv_repeat', 1))
        v = repeat_kv(v, getattr(self, 'kv_repeat', 1))

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)

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
            attn_weights = attn_weights * head_mask

        attn_ctx = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, N, D)
        attn_out = self.out_lin(attn_ctx)

        # Spectral Mamba stream
        mamba_out = self.spectral_mamba(hidden_states)

        # Adaptive gating
        g = torch.sigmoid(self.gate(hidden_states))  # (B, N, 1)
        out = g * attn_out + (1.0 - g) * mamba_out

        if output_attentions:
            return out, attn_weights
        return out, None


def inject_spectral_mamba(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> SpectralMambaAttention:
    r"""Injects SpectralMambaAttention into a transformer layer."""
    if model_config is None and hasattr(layer, "attention"):
        old_attn = layer.attention
    h_size = getattr(model_config, "hidden_size", 768) if model_config is not None else 768
    n_heads = getattr(model_config, "num_attention_heads", 4) if model_config is not None else 12

    cfg = SpectralMambaConfig(
        hidden_size=h_size,
        num_attention_heads=n_heads,
        head_dim=h_size // max(1, n_heads),
        dropout=getattr(comp_config, "dropout", 0.1),
        gate_init_bias=getattr(comp_config, "gate_init_bias", 1.8),
    )
    if model_config is None and old_attn is not None:
        new_attn = SpectralMambaAttention.from_distilbert_pretrained(old_attn, config=cfg)
    else:
        new_attn = SpectralMambaAttention(config=cfg, pretrained_attn=old_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_attn
    return new_attn
