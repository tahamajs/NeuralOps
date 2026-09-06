r"""
UPI-Stuffed Mamba: Uncertainty-Penalized Interpolated Mamba SSM Block.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and trainable LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out


@ATTENTION_REGISTRY.register("upi_stuffed_mamba", supports_causal=True, supports_bidirectional=True)
class UPIMambaBlock(BaseAttention):
    r"""Bidirectional / Causal Uncertainty-Penalized Interpolated Mamba Block."""

    supports_causal: bool = True
    supports_bidirectional: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        dim: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn", None)

        if dim is None and d_model is None:
            if config is not None and not isinstance(config, int):
                dim = _cfg_get(config, "hidden_size", _cfg_get(config, "dim", kwargs.get("hidden_size", 768)))
            else:
                dim = kwargs.get("hidden_size", 768)
        else:
            dim = dim or d_model

        if hasattr(dim, "hidden_size"):
            dim = dim.hidden_size

        self.dim: int = int(dim)
        self.d_state: int = d_state
        self.d_conv: int = d_conv
        self.expand: int = expand
        self.inner_dim: int = self.dim * expand
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None and not isinstance(config, int):
            self.is_decoder = bool(getattr(config, "is_decoder", False))
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False

        self.in_proj = LoRALinear(
            self.dim, self.inner_dim * 2, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1,
        )
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.dim, 1))
        )
        self.D = nn.Parameter(torch.ones(self.dim))
        self.x_proj = nn.Linear(self.inner_dim, self.inner_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.out_proj = LoRALinear(
            self.inner_dim, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )

        if src_attn is not None:
            self.copy_weights(src_attn)

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # UniversalAdapter extraction
            for proj in ["q", "k", "v", "o"]:
                try:
                    w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                    if w is not None:
                        if proj == "q": w_q, b_q = w, b
                        elif proj == "k": w_k, b_k = w, b
                        elif proj == "v": w_v, b_v = w, b
                        elif proj == "o": w_o, b_o = w, b
                except Exception:
                    pass

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight.detach()
                    b_o = old_attn.output.dense.bias.detach() if old_attn.output.dense.bias is not None else None
            else:
                src = old_attn

            # GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if c_attn_w.shape[0] == self.dim and c_attn_w.shape[1] != self.dim:
                    c_attn_w = c_attn_w.t()

                w_q = c_attn_w[: self.dim, :]
                w_k = c_attn_w[self.dim : 2 * self.dim, :]
                w_v = c_attn_w[2 * self.dim : 3 * self.dim, :]

                out_w = src.c_proj.weight.detach()
                if out_w.shape[0] == self.dim and out_w.shape[1] != self.dim:
                    out_w = out_w.t()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # RoBERTa / BERT separate query, key, value
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            w_o = proj.weight.detach()
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            # Qwen / LLaMA
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None

                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None

            # DistilBERT
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None

                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None

            missing = []
            if w_q is None: missing.append("Q projection")
            if w_v is None: missing.append("V projection")
            if w_o is None and not self.is_nested_encoder: missing.append("Out projection")
            if missing:
                raise RuntimeError(f"UPIMambaBlock: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}")

            def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
                if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                    return w.t()
                return w

            w_v_or = _orient(w_v, self.dim)
            w_q_or = _orient(w_q, self.dim)

            # Copy input projection: [V; Q]
            v_chunk = w_v_or[: self.inner_dim, : self.dim]
            q_chunk = w_q_or[: self.inner_dim, : self.dim]

            min_in = min(self.inner_dim, v_chunk.shape[0])
            self.in_proj.weight.data[:min_in, : self.dim].copy_(v_chunk[:min_in, : self.dim])
            min_q = min(self.inner_dim, q_chunk.shape[0])
            self.in_proj.weight.data[self.inner_dim : self.inner_dim + min_q, : self.dim].copy_(q_chunk[:min_q, : self.dim])

            if w_o is not None:
                w_o_or = _orient(w_o, self.inner_dim)
                min_o_out = min(self.dim, w_o_or.shape[0])
                min_o_in = min(self.dim, w_o_or.shape[1])
                self.out_proj.weight.data[:min_o_out, :min_o_in].copy_(w_o_or[:min_o_out, :min_o_in])
                if b_o is not None and self.out_proj.bias is not None:
                    b_val = b_o.data if isinstance(b_o, nn.Parameter) else b_o
                    self.out_proj.bias.data[:min(len(self.out_proj.bias), len(b_val))].copy_(b_val[:min(len(self.out_proj.bias), len(b_val))])
            elif self.is_nested_encoder:
                self.out_proj.weight.data.zero_()
                min_d = min(self.out_proj.weight.shape[0], self.out_proj.weight.shape[1])
                self.out_proj.weight.data[:min_d, :min_d].fill_diagonal_(1.0)

            self.in_proj.freeze_base_weights()
            self.out_proj.freeze_base_weights()

            # Preserve source LoRA adapters
            source_submodules = {
                "in": ["q_proj", "q_lin", "query", "W_q"],
                "out": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
            }
            for name in source_submodules["in"]:
                if hasattr(old_attn, name):
                    sub = getattr(old_attn, name)
                    if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
                        src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
                        src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
                        if hasattr(src_lA, "data") and self.in_proj.lora_A is not None:
                            r_min = min(self.in_proj.lora_A.shape[0], src_lA.shape[0])
                            in_min = min(self.in_proj.lora_A.shape[1], src_lA.shape[1])
                            self.in_proj.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                            self.in_proj.lora_A.requires_grad = True
                        if hasattr(src_lB, "data") and self.in_proj.lora_B is not None:
                            out_min = min(self.in_proj.lora_B.shape[0], src_lB.shape[0])
                            r_min = min(self.in_proj.lora_B.shape[1], src_lB.shape[1])
                            self.in_proj.lora_B.data[:out_min, :r_min].copy_(src_lB.data[:out_min, :r_min])
                            self.in_proj.lora_B.requires_grad = True
                        if hasattr(sub, "lora_scaling"):
                            self.in_proj.scaling = sub.lora_scaling
                        break

            for name in source_submodules["out"]:
                if hasattr(old_attn, name):
                    sub = getattr(old_attn, name)
                    if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
                        src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
                        src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
                        if hasattr(src_lA, "data") and self.out_proj.lora_A is not None:
                            r_min = min(self.out_proj.lora_A.shape[0], src_lA.shape[0])
                            in_min = min(self.out_proj.lora_A.shape[1], src_lA.shape[1])
                            self.out_proj.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                            self.out_proj.lora_A.requires_grad = True
                        if hasattr(src_lB, "data") and self.out_proj.lora_B is not None:
                            out_min = min(self.out_proj.lora_B.shape[0], src_lB.shape[0])
                            r_min = min(self.out_proj.lora_B.shape[1], src_lB.shape[1])
                            self.out_proj.lora_B.data[:out_min, :r_min].copy_(src_lB.data[:out_min, :r_min])
                            self.out_proj.lora_B.requires_grad = True
                        if hasattr(sub, "lora_scaling"):
                            self.out_proj.scaling = sub.lora_scaling
                        break

    def _upi_scan(
        self,
        x_ssm: torch.Tensor,
        dt: torch.Tensor,
        gamma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Sequential UPI SSM scan with FP32 precision."""
        B, L, D = x_ssm.shape
        d_state = self.d_state
        A = torch.exp(torch.clamp(self.A_log.float(), min=-20.0, max=10.0))

        h = torch.zeros(B, D, d_state, device=x_ssm.device, dtype=torch.float32)
        outputs = []

        x_f = x_ssm.float()
        dt_f = dt.float()
        gamma_f = gamma.float()

        step_range = range(L - 1, -1, -1) if reverse else range(L)
        for t in step_range:
            xt = x_f[:, t, :]
            dt_t = dt_f[:, t, :].unsqueeze(-1)
            gamma_t = gamma_f[:, t, :].unsqueeze(-1)

            dA = torch.exp(torch.clamp(-dt_t * A.unsqueeze(0), min=-20.0, max=0.0)) * (1.0 - 0.5 * gamma_t)
            dB_x = (dt_t * xt.unsqueeze(-1)).repeat(1, 1, d_state)

            h_next = dA * h + dB_x

            if mask is not None:
                m_t = mask[:, t].view(B, 1, 1).float()
                h = m_t * h_next + (1.0 - m_t) * h
            else:
                h = h_next

            out_t = (h * self.D.unsqueeze(-1)).sum(dim=-1)
            outputs.append(out_t.to(x_ssm.dtype))

        if reverse:
            outputs.reverse()

        return torch.stack(outputs, dim=1)

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

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        mask_bool = None
        if mask is not None:
            if mask.dim() == 2:
                mask_bool = mask.bool().to(hidden_states.device)
            elif mask.dim() == 4:
                mask_bool = (mask[:, 0, 0, :] > -10000.0).to(hidden_states.device)
            else:
                mask_bool = get_token_valid_mask(mask, B, L, hidden_states.device)

        xz = self.in_proj(hidden_states)
        x_in, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_ssm = F.silu(x_conv)

        x_dbl = self.x_proj(x_ssm)
        dt_raw, gamma_raw = x_dbl.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        gamma = torch.sigmoid(gamma_raw)

        y_fwd = self._upi_scan(x_ssm, dt, gamma, mask=mask_bool, reverse=False)

        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )

        if is_decoder:
            y = y_fwd
        else:
            y_bwd = self._upi_scan(x_ssm, dt, gamma, mask=mask_bool, reverse=True)
            y = 0.5 * (y_fwd + y_bwd)

        gated = y * F.silu(z)

        if self.is_nested_encoder:
            out = gated
        else:
            out = self.out_proj(gated)

        if mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return (out, None)


@ATTENTION_REGISTRY.register("upi_stuffed_mamba_inject")
@ATTENTION_REGISTRY.register("inject_upi_mamba")
@ATTENTION_REGISTRY.register("upi_mamba_inject")
def inject_upi_mamba(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[UPIMambaBlock]:
    """Safe injection hook for UPIMambaBlock."""
    if old_attn is None:
        return None

    if model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    if model_config is None and hasattr(old_attn, "config"):
        model_config = old_attn.config

    dim = (
        _cfg_get(model_config, "hidden_size")
        or _cfg_get(model_config, "d_model")
        or _cfg_get(model_config, "dim")
        or getattr(old_attn, "hidden_size", None)
        or getattr(old_attn, "embed_dim", None)
        or (old_attn.q_proj.in_features if hasattr(old_attn, "q_proj") and hasattr(old_attn.q_proj, "in_features") else None)
        or (old_attn.q_lin.in_features if hasattr(old_attn, "q_lin") and hasattr(old_attn.q_lin, "in_features") else None)
        or (old_attn.query.in_features if hasattr(old_attn, "query") and hasattr(old_attn.query, "in_features") else None)
        or 768
    )

    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

    try:
        new_block = UPIMambaBlock(
            config=model_config,
            dim=dim,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=old_attn,
            **kwargs,
        )

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_block)
                return new_block

        layer.attention = new_block
        return new_block
    except Exception as e:
        logger.error("inject_upi_mamba failed at layer %d: %s", layer_idx, e)
        return None
