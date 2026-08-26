"""
SAMBA-M3: Hybrid Sliding-Window Attention and Spectral Mamba Block.
"""

from typing import Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.simple_ssm import SimpleSSM



# Internal spectral mamba block for SAMBA-M3
class SpectralMamba(nn.Module):
    r"""
    Mamba with learnable spectral gating.
    """

    def __init__(self, d_model: int = 768, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
        elif hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.mamba = SimpleSSM(d_model=d_model)
        self.dt_scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.mamba(x)
        if isinstance(out, tuple):
            out = out[0]
        gate = torch.sigmoid(self.dt_scale).unsqueeze(0).unsqueeze(0)
        out = out * gate
        return out, None


@ATTENTION_REGISTRY.register("samba_m3")
class SAMBAM3Block(nn.Module):
    r"""
    SAMBA-M3: Combination of Local Attention and Spectral Mamba.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        window: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if isinstance(config, int):
            d_model = config
            config = None
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "d_model", 768))
            else:
                d_model = kwargs.get("hidden_size", 768)
        if n_heads is None:
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", kwargs.get("num_attention_heads", 12)))
        window = getattr(config, "window_size", window)

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window

        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)

        self.mamba = SpectralMamba(d_model)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(d_model)
        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        with torch.no_grad():
            q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", None)))
            k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", None)))
            v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", None)))
            out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o_proj", None))))

            if q_src is not None and hasattr(q_src, "weight"):
                self.q_lin.weight.copy_(q_src.weight)
                if q_src.bias is not None:
                    self.q_lin.bias.copy_(q_src.bias)
            if k_src is not None and hasattr(k_src, "weight"):
                self.k_lin.weight.copy_(k_src.weight)
                if k_src.bias is not None:
                    self.k_lin.bias.copy_(k_src.bias)
            if v_src is not None and hasattr(v_src, "weight"):
                self.v_lin.weight.copy_(v_src.weight)
                if v_src.bias is not None:
                    self.v_lin.bias.copy_(v_src.bias)
            if out_src is not None and hasattr(out_src, "weight"):
                self.out_lin.weight.copy_(out_src.weight)
                if out_src.bias is not None:
                    self.out_lin.bias.copy_(out_src.bias)

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

        q = self.q_lin(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Bidirectional window mask
        idx = torch.arange(L, device=q.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        window_mask = dist > self.window
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), -10000.0)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, -10000.0)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask

        attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, L, D)
        attn_proj = self.out_lin(attn_out)

        mamba_out, _ = self.mamba(hidden_states)
        alpha = torch.sigmoid(self.alpha)
        out = self.norm(hidden_states + alpha * attn_proj + (1.0 - alpha) * mamba_out)

        return out, (probs if output_attentions else None)
