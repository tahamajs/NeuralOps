r"""
SOA Fused Attention: Low-Rank Latent Space Multi-Head Attention with Zero-Shot SVD Initialization.

Factorizes dense Q, K, and V linear projection operators into separate low-rank bottleneck pairs
initialized from truncated Singular Value Decomposition (SVD) of source attention weights.
"""

from typing import Any, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


class FactorizedLinear(nn.Module):
    r"""Factorized linear projection with reconstructed weight property for compatibility."""

    def __init__(self, down: nn.Linear, up: nn.Linear) -> None:
        super().__init__()
        self.down = down
        self.up = up

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.up.weight, self.down.weight)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.up, "bias", None)

    @property
    def in_features(self) -> int:
        return self.down.in_features

    @property
    def out_features(self) -> int:
        return self.up.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


@ATTENTION_REGISTRY.register("soa_fused_module")
@ATTENTION_REGISTRY.register("soa_fused_attn")
@ATTENTION_REGISTRY.register("SOAFusedAttention")
@ATTENTION_REGISTRY.register("soa_fused_attention")
class SOAFusedAttention(nn.Module):
    r"""Fused SOA Attention module factorizing Q/K/V projections with low-rank bottlenecks."""

    def __init__(
        self,
        model_config: Optional[Any] = None,
        k_rank: int = 64,
        dropout: float = 0.1,
        pretrained_attn: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if model_config is None and pretrained_attn is not None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", None)).out_features
            model_config.num_attention_heads = getattr(pretrained_attn, "n_heads", getattr(pretrained_attn, "num_attention_heads", 4))

        self.hidden_size: int = getattr(model_config, "hidden_size", 768) if model_config is not None else 768
        self.num_heads: int = getattr(model_config, "num_attention_heads", 4) if model_config is not None else 12
        self.head_dim: int = self.hidden_size // max(1, self.num_heads)
        self.k_rank: int = k_rank

        self.q_down = nn.Linear(self.hidden_size, k_rank, bias=False)
        self.k_down = nn.Linear(self.hidden_size, k_rank, bias=False)
        self.v_down = nn.Linear(self.hidden_size, k_rank, bias=False)

        self.q_up = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.k_up = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.v_up = nn.Linear(k_rank, self.hidden_size, bias=False)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

        if pretrained_attn is not None:
            self._init_from_pretrained(pretrained_attn)

    @property
    def q_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.q_down, self.q_up)

    @property
    def k_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.k_down, self.k_up)

    @property
    def v_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.v_down, self.v_up)

    # T5, BERT, DistilBERT, RoBERTa compatibility aliases
    @property
    def q(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def k(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def v(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def o(self) -> nn.Linear:
        return self.out_proj

    @property
    def query(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def dense(self) -> nn.Linear:
        return self.out_proj

    @property
    def q_lin(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def k_lin(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def v_lin(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def out_lin(self) -> nn.Linear:
        return self.out_proj

    def _init_from_pretrained(self, pretrained_attn: Any) -> None:
        r"""Initialize down- and up-projections via truncated SVD decomposition."""
        q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", getattr(pretrained_attn, "q", None))))
        k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", getattr(pretrained_attn, "k", None))))
        v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", getattr(pretrained_attn, "v", None))))
        out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o", None))))

        with torch.no_grad():
            for orig, down, up in [
                (q_src, self.q_down, self.q_up),
                (k_src, self.k_down, self.k_up),
                (v_src, self.v_down, self.v_up),
            ]:
                if orig is None or not hasattr(orig, "weight"):
                    continue
                W = orig.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                S = S.clamp(min=1e-6)
                r = min(self.k_rank, U.size(1), Vt.size(0))
                down.weight.data.zero_()
                up.weight.data.zero_()
                down.weight.data[:r, :] = Vt[:r, :].to(down.weight.dtype)
                up.weight.data[:, :r] = (U[:, :r] * S[:r]).to(up.weight.dtype)

            if out_src is not None and hasattr(out_src, "weight"):
                self.out_proj.weight.data.copy_(out_src.weight.data)
                if getattr(out_src, "bias", None) is not None:
                    self.out_proj.bias.data.copy_(out_src.bias.data)

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "SOAFusedAttention":
        return cls(model_config=model_config, pretrained_attn=old_attn, **kwargs)

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
        B, N, _ = hidden_states.shape

        z_q = self.q_down(hidden_states)
        z_k = self.k_down(hidden_states)
        z_v = self.v_down(hidden_states)

        q = self.q_up(z_q)
        k = self.k_up(z_k)
        v = self.v_up(z_v)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k  = k.view(B, N, getattr(self, "num_kv_heads", self.num_heads), self.head_dim).transpose(1, 2)
        k = repeat_kv(k, getattr(self, 'kv_repeat', 1))
        v  = v.view(B, N, getattr(self, "num_kv_heads", self.num_heads), self.head_dim).transpose(1, 2)
        v = repeat_kv(v, getattr(self, 'kv_repeat', 1))

        scale = self.head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask = attention_mask if attention_mask is not None else kwargs.get("attn_mask", None)
        if mask is not None:
            if mask.dim() == 2:
                mask_t = mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask_t) * -10000.0
            elif mask.dim() == 3:
                scores = scores + mask.unsqueeze(1)
            elif mask.dim() == 4:
                scores = scores + mask.to(scores.dtype)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)

        if output_attentions:
            return output, attn_weights
        return output, None


def inject_soa_fused_attn(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    **kwargs: Any,
) -> SOAFusedAttention:
    r"""Bridge helper injecting SOAFusedAttention into a target transformer layer."""
    if model_config is None and hasattr(layer, "attention"):
        old_attn = layer.attention
    k_rank = getattr(comp_config, "soa_rank", 64)
    dropout = getattr(comp_config, "dropout", 0.1)
    if model_config is None and old_attn is not None:
        new_attn = SOAFusedAttention.from_distilbert_pretrained(old_attn, k_rank=k_rank, dropout=dropout)
    else:
        new_attn = SOAFusedAttention(model_config=model_config, k_rank=k_rank, dropout=dropout, pretrained_attn=old_attn)
    if hasattr(layer, "attention"):
        layer.attention = new_attn
    return new_attn
