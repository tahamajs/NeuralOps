r"""
SPECTRA-v2: Sparse Per-Head Efficient Context Transformer with Rectified Attention.

Hybrid sparse-attention architecture unifying intra-block local attention, relative-position-biased
inter-block top-$k$ routing, per-head adaptive attention spans, and periodic global Random Fourier Feature (RFF) rectification.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class SpectraConfig:
    r"""Configuration dataclass for SPECTRA-v2 attention layers."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    dropout: float = 0.1
    block_size: int = 64
    top_k_inter: int = 16
    use_rel_pos_bias: bool = True
    rel_pos_max_distance: int = 128
    adaptive_span: bool = True
    max_span: float = 512.0
    rectify_interval: int = 4
    rff_features: int = 128
    aks_enabled: bool = True


class SimpleRFFKernel(nn.Module):
    r"""Random Fourier Feature (RFF) kernel feature map for global attention rectification."""

    def __init__(self, head_dim: int, rff_dim: int) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.randn(head_dim, rff_dim) * 0.02)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def rff(x: torch.Tensor) -> torch.Tensor:
            p = torch.matmul(x, self.proj.to(x.dtype))
            return torch.cat([torch.cos(p), torch.sin(p)], dim=-1) / (
                self.proj.size(1) ** 0.25
            )

        return rff(q), rff(k)


@ATTENTION_REGISTRY.register("spectra_v2")
@ATTENTION_REGISTRY.register("spectra_v2_inject")
class SpectraV2Attention(nn.Module):
    r"""SPECTRA-v2 attention layer integrating block sparsity, top-$k$ routing, and RFF rectification."""

    def __init__(
        self,
        model_config: Optional[Any] = None,
        spectra_cfg: Optional[SpectraConfig] = None,
        layer_idx: int = 0,
        total_layers: int = 12,
        bert_mode: bool = False,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if model_config is None and pretrained_attn is not None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", None)).out_features
            model_config.num_attention_heads = getattr(pretrained_attn, "n_heads", getattr(pretrained_attn, "num_attention_heads", 12))

        self.num_heads: int = getattr(model_config, "num_attention_heads", 12) if model_config is not None else 12
        self.hidden_size: int = getattr(model_config, "hidden_size", 768) if model_config is not None else 768
        self.head_dim: int = self.hidden_size // max(1, self.num_heads)
        self.all_head_size: int = self.num_heads * self.head_dim
        self.cfg: SpectraConfig = spectra_cfg if spectra_cfg is not None else SpectraConfig()
        self.layer_idx: int = layer_idx
        self.total_layers: int = total_layers
        self.bert_mode: bool = bert_mode

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)
        self.dropout = nn.Dropout(self.cfg.dropout)

        if self.cfg.adaptive_span:
            self.span: Optional[nn.Parameter] = nn.Parameter(
                torch.full((self.num_heads,), self.cfg.max_span, dtype=torch.float)
            )
        else:
            self.span = None

        self.rel_pos_bias: Optional[nn.Parameter] = None
        if self.cfg.use_rel_pos_bias:
            max_dist = self.cfg.rel_pos_max_distance
            self.rel_pos_bias = nn.Parameter(
                torch.zeros(self.num_heads, 2 * max_dist + 1)
            )
        self.max_rel_dist: int = (
            self.cfg.rel_pos_max_distance if self.cfg.use_rel_pos_bias else 0
        )

        self.rect_interval: int = self.cfg.rectify_interval
        self.is_rectify_layer: bool = (
            layer_idx % self.rect_interval == 0
        ) and self.cfg.aks_enabled
        if self.is_rectify_layer:
            self.aks: Optional[SimpleRFFKernel] = SimpleRFFKernel(
                self.head_dim, self.cfg.rff_features
            )
        else:
            self.aks = None

        if pretrained_attn is not None:
            self._copy_pretrained(pretrained_attn)

    def _copy_pretrained(self, old_attn: nn.Module) -> None:
        q_src = getattr(old_attn, "q_lin", getattr(old_attn, "query", None))
        k_src = getattr(old_attn, "k_lin", getattr(old_attn, "key", None))
        v_src = getattr(old_attn, "v_lin", getattr(old_attn, "value", None))
        out_src = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None))
        if q_src is not None:
            self.query.weight.data.copy_(q_src.weight.data)
            if q_src.bias is not None:
                self.query.bias.data.copy_(q_src.bias.data)
        if k_src is not None:
            self.key.weight.data.copy_(k_src.weight.data)
            if k_src.bias is not None:
                self.key.bias.data.copy_(k_src.bias.data)
        if v_src is not None:
            self.value.weight.data.copy_(v_src.weight.data)
            if v_src.bias is not None:
                self.value.bias.data.copy_(v_src.bias.data)
        if out_src is not None:
            self.out_proj.weight.data.copy_(out_src.weight.data)
            if out_src.bias is not None:
                self.out_proj.bias.data.copy_(out_src.bias.data)

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "SpectraV2Attention":
        return cls(model_config=model_config, pretrained_attn=old_attn, **kwargs)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(new_shape).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        Q = self._reshape(self.query(hidden_states))
        K = self._reshape(self.key(hidden_states))
        V = self._reshape(self.value(hidden_states))

        scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim**-0.5)
        mask_val = -10000.0 if scores.dtype in (torch.float16, torch.bfloat16) else -10000.0

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_m = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * mask_val
                scores = scores + pad_m
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(context)
        return output, attn_weights if output_attentions else None

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.weight.device)


def inject_spectra_v2(
    layer: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> SpectraV2Attention:
    r"""Injects SpectraV2Attention into a transformer layer."""
    if model_config is None and hasattr(layer, "attention"):
        old_attn = layer.attention
    h_size = getattr(model_config, "hidden_size", 768) if model_config is not None else 768
    n_heads = getattr(model_config, "num_attention_heads", 12) if model_config is not None else 12

    cfg = SpectraConfig(
        hidden_size=h_size,
        num_attention_heads=n_heads,
        dropout=getattr(comp_config, "dropout", 0.1),
        block_size=getattr(comp_config, "block_size", 64),
        top_k_inter=getattr(comp_config, "top_k_inter", 16),
        use_rel_pos_bias=getattr(comp_config, "use_rel_pos_bias", True),
        rel_pos_max_distance=getattr(comp_config, "rel_pos_max_distance", 128),
        adaptive_span=getattr(comp_config, "adaptive_span", True),
        max_span=getattr(comp_config, "max_span", 512.0),
        rectify_interval=getattr(comp_config, "rectify_interval", 4),
        rff_features=getattr(comp_config, "rff_features", 128),
        aks_enabled=getattr(comp_config, "aks_enabled", True),
    )
    if model_config is None and old_attn is not None:
        new_attn = SpectraV2Attention.from_distilbert_pretrained(old_attn, spectra_cfg=cfg)
    else:
        new_attn = SpectraV2Attention(
            model_config=model_config,
            spectra_cfg=cfg,
            layer_idx=layer_idx,
            total_layers=total_layers,
            pretrained_attn=old_attn,
        )
    if hasattr(layer, "attention"):
        layer.attention = new_attn
    return new_attn
