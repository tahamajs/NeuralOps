r"""
ViG Attention: Vision GNN-inspired 2D Grid Linear Attention with Gated Fusion.
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
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions: [-x2, x1].
    Robust to odd head_dim by rotating only the even slice 2 * (d // 2).
    """
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    if rotary_dim == 0:
        return x
    x_rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings safely handling even or odd dimensions."""
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., : q.shape[-2], :rotary_dim].to(q.dtype)
    sin_q = sin[..., : q.shape[-2], :rotary_dim].to(q.dtype)

    cos_k = cos[..., : k.shape[-2], :rotary_dim].to(k.dtype)
    sin_k = sin[..., : k.shape[-2], :rotary_dim].to(k.dtype)

    q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed

    return q, k


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
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
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
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register("vig", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("vig_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class ViGAttention(BaseAttention):
    r"""Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion."""

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn", None)

        if config is not None and not isinstance(config, int):
            self.adapter = ModelAdapter(config)
            self.num_heads: int = self.adapter.num_heads
            self.num_kv_heads: int = self.adapter.num_key_value_heads
            self.dim: int = self.adapter.hidden_size
            self.head_dim: int = self.adapter.head_dim
            self.all_head_size: int = self.num_heads * self.head_dim
            self.kv_head_size: int = self.num_kv_heads * self.head_dim
            self.kv_repeat: int = self.adapter.kv_repeat
            self.is_decoder: bool = getattr(config, "is_decoder", False)
        else:
            self.adapter = None
            self.dim = _cfg_get(config, "hidden_size", kwargs.get("hidden_size", kwargs.get("d_model", 768)))
            self.num_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_heads", 12))
            self.num_kv_heads = _cfg_get(config, "num_key_value_heads", self.num_heads)
            self.head_dim = self.dim // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
            self.is_decoder = bool(_cfg_get(config, "is_decoder", kwargs.get("is_decoder", False)))

        # Explicit overrides
        if "hidden_size" in kwargs:
            self.dim = kwargs["hidden_size"]
            self.head_dim = self.dim // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
        if "num_heads" in kwargs:
            self.num_heads = kwargs["num_heads"]
            self.head_dim = self.dim // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)

        self.is_nested_encoder: bool = False
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        self.q_proj = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.k_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.v_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )

        # Standard aliases
        self.W_q = self.q_proj
        self.W_k = self.k_proj
        self.W_v = self.v_proj
        self.out = self.out_proj
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj

        self.gate_net = nn.Linear(self.dim, 2 * self.num_heads)
        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(src_attn, "rotary_emb", None)

        if src_attn is not None:
            self.copy_weights(src_attn)

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(old_attn, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            weight = old_attn.c_attn.weight.data
            bias = getattr(old_attn.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.dim:
                weight = weight.t()
            split_size = weight.shape[0] // 3
            q_w = weight[:split_size, :]
            k_w = weight[split_size : 2 * split_size, :]
            v_w = weight[2 * split_size :, :]
            q_b = bias_data[:split_size] if bias_data is not None else None
            k_b = bias_data[split_size : 2 * split_size] if bias_data is not None else None
            v_b = bias_data[2 * split_size :] if bias_data is not None else None
            extracted["q"] = (q_w, q_b)
            extracted["k"] = (k_w, k_b)
            extracted["v"] = (v_w, v_b)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            w_o = old_attn.c_proj.weight.data
            if w_o.dim() == 2 and w_o.shape[0] != self.dim and w_o.shape[1] == self.dim:
                w_o = w_o.t()
            b_o = getattr(old_attn.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(old_attn, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(old_attn, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense") and extracted["o"][0] is None:
            dense = old_attn.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(old_attn, name):
                        layer = getattr(old_attn, name)
                        base = getattr(layer, "base_layer", layer)
                        if hasattr(base, "weight"):
                            extracted[proj] = (base.weight.data, getattr(base, "bias", None))
                            break

        # Detect nested encoder architecture
        cls_name = old_attn.__class__.__name__
        has_out_proj = any(hasattr(old_attn, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj", "out"))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"ViGAttention missing required weights for projections: {missing}")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t()
            return w

        with torch.no_grad():
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.dim
                    w_oriented = _orient(w, in_dim)
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w_oriented.shape
                    target_w.data[:min(out_d, src_out), :min(in_d, src_in)].copy_(w_oriented[:min(out_d, src_out), :min(in_d, src_in)])
                    if b is not None and target.bias is not None:
                        b_val = b.data if isinstance(b, nn.Parameter) else b
                        target.bias.data[:min(len(target.bias), len(b_val))].copy_(b_val[:min(len(target.bias), len(b_val))])
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        # Preserve source LoRA adapters
        source_submodules = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            for name in source_submodules[proj_key]:
                if hasattr(old_attn, name):
                    sub = getattr(old_attn, name)
                    if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
                        src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
                        src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
                        if hasattr(src_lA, "data") and target.lora_A is not None:
                            r_min = min(target.lora_A.shape[0], src_lA.shape[0])
                            in_min = min(target.lora_A.shape[1], src_lA.shape[1])
                            target.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                            target.lora_A.requires_grad = True
                        if hasattr(src_lB, "data") and target.lora_B is not None:
                            out_min = min(target.lora_B.shape[0], src_lB.shape[0])
                            r_min = min(target.lora_B.shape[1], src_lB.shape[1])
                            target.lora_B.data[:out_min, :r_min].copy_(src_lB.data[:out_min, :r_min])
                            target.lora_B.requires_grad = True
                        if hasattr(sub, "lora_scaling"):
                            target.lora_scaling = sub.lora_scaling
                        break

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)

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
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        B, N, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        H: int = max(1, int(math.isqrt(N)))
        W: int = (N + H - 1) // H
        total: int = H * W
        pad_len: int = total - N

        if pad_len > 0:
            x = torch.cat(
                [
                    hidden_states,
                    torch.zeros(B, pad_len, D, device=device, dtype=dtype),
                ],
                dim=1,
            )
            N_pad = total
        else:
            x = hidden_states
            N_pad = N

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if mask is not None:
            if mask.dim() == 4:
                valid_2d = (mask[:, 0, 0, :] > -10000.0).to(device)
            elif mask.dim() == 3:
                valid_2d = (mask[:, 0, :] > -10000.0).to(device)
            elif mask.dim() == 2:
                valid_2d = (mask > 0).to(device)
            else:
                valid_2d = mask.bool().to(device)
            if pad_len > 0:
                valid = torch.cat([valid_2d, torch.zeros(B, pad_len, dtype=torch.bool, device=device)], dim=1)
            else:
                valid = valid_2d
        else:
            valid = torch.ones(B, N_pad, dtype=torch.bool, device=device)

        Hh = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim

        q = self.q_proj(x).view(B, N_pad, Hh, Dh).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, N_pad, H_kv, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, N_pad, H_kv, Dh).permute(0, 2, 1, 3)

        # Rotary position embeddings
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            if pad_len > 0:
                cos_pad = torch.cat([cos, torch.zeros(cos.size(0), pad_len, cos.size(-1), device=cos.device, dtype=cos.dtype)], dim=1)
                sin_pad = torch.cat([sin, torch.zeros(sin.size(0), pad_len, sin.size(-1), device=sin.device, dtype=sin.dtype)], dim=1)
            else:
                cos_pad, sin_pad = cos, sin
            q, k = apply_rotary_pos_emb(q, k, cos_pad, sin_pad)
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                from framework.utils.compat import maybe_apply_rotary
                q, k = maybe_apply_rotary(q, k, getattr(self, "adapter", None), self.rotary_emb, kwargs.get("position_ids"))
            except Exception:
                pass

        # GQA expansion
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        gates = torch.sigmoid(self.gate_net(x)).view(B, N_pad, 2, Hh, 1)

        q = q.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)

        # Row Linear Attention
        q_row = q.reshape(B, Hh * W, H, Dh).transpose(2, 3)
        k_row = k.reshape(B, Hh * W, H, Dh).transpose(2, 3)
        v_row = v.reshape(B, Hh * W, H, Dh).transpose(2, 3)

        q_row_lin = F.elu(q_row.float()) + 1.0
        k_row_lin = F.elu(k_row.float()) + 1.0

        valid_grid = valid.reshape(B, H, W).permute(0, 2, 1)
        valid_row = (
            valid_grid.unsqueeze(1)
            .expand(B, Hh, W, H)
            .reshape(B, Hh * W, H)
            .unsqueeze(-2)
            .to(k_row_lin.dtype)
        )
        k_row_lin = k_row_lin * valid_row
        v_row_f = v_row.float() * valid_row

        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )

        if is_decoder:
            attn_r = q_row_lin.transpose(-1, -2) @ k_row_lin
            tril = torch.tril(torch.ones(H, H, device=q.device, dtype=torch.bool))
            attn_r = attn_r * tril.unsqueeze(0).unsqueeze(0)
            denom_r = attn_r.sum(dim=-1, keepdim=True) + 1e-5
            out_row = (attn_r @ v_row_f.transpose(-1, -2)) / denom_r
            out_row = out_row.transpose(-1, -2).transpose(2, 3).reshape(B, Hh, W, H, Dh).permute(0, 1, 3, 2, 4)
        else:
            kv_row = torch.einsum("b m n d, b m n e -> b m d e", k_row_lin, v_row_f)
            z_row = k_row_lin.sum(dim=2, keepdim=True)
            denom_row = torch.einsum("b m n d, b m d -> b m n", q_row_lin, z_row.squeeze(2)).unsqueeze(-1)
            out_row = torch.einsum("b m n d, b m d e -> b m n e", q_row_lin, kv_row) / (denom_row.clamp(min=1e-5))
            out_row = out_row.transpose(2, 3).reshape(B, Hh, W, H, Dh).permute(0, 1, 3, 2, 4)

        # Column Linear Attention
        q_col = q.permute(0, 1, 2, 4, 3)
        k_col = k.permute(0, 1, 2, 4, 3)
        v_col = v.permute(0, 1, 2, 4, 3)
        q_col_flat = q_col.reshape(B * Hh * H, W, Dh)
        k_col_flat = k_col.reshape(B * Hh * H, W, Dh)
        v_col_flat = v_col.reshape(B * Hh * H, W, Dh)

        q_col_lin = F.elu(q_col_flat.float()) + 1.0
        k_col_lin = F.elu(k_col_flat.float()) + 1.0

        valid_col = (
            valid.reshape(B, H, W)
            .unsqueeze(1)
            .expand(B, Hh, H, W)
            .reshape(B * Hh * H, W)
            .unsqueeze(-1)
            .to(k_col_lin.dtype)
        )
        k_col_lin = k_col_lin * valid_col
        v_col_f = v_col_flat.float() * valid_col

        if is_decoder:
            attn_c = q_col_lin @ k_col_lin.transpose(-1, -2)
            tril_c = torch.tril(torch.ones(W, W, device=q.device, dtype=torch.bool))
            attn_c = attn_c * tril_c.unsqueeze(0)
            denom_c = attn_c.sum(dim=-1, keepdim=True) + 1e-5
            out_col = (attn_c @ v_col_f) / denom_c
            out_col = out_col.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        else:
            kv_col = torch.einsum("b n d, b n e -> b d e", k_col_lin, v_col_f)
            z_col = k_col_lin.sum(dim=1, keepdim=True)
            denom_col = torch.einsum("b n d, b d -> b n", q_col_lin, z_col.squeeze(1)).unsqueeze(-1)
            out_col = torch.einsum("b n d, b d e -> b n e", q_col_lin, kv_col) / (denom_col.clamp(min=1e-5))
            out_col = out_col.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)

        # Gated fusion
        out_row = out_row.permute(0, 1, 3, 2, 4).reshape(B, Hh, N_pad, Dh).permute(0, 2, 1, 3)
        out_col = out_col.permute(0, 1, 3, 2, 4).reshape(B, Hh, N_pad, Dh).permute(0, 2, 1, 3)

        gate_row = gates[:, :, 0, :, :]
        gate_col = gates[:, :, 1, :, :]

        out = (out_row * gate_row + out_col * gate_col).to(dtype)

        # Head mask broadcasting
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, 1, -1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), 1, head_mask.size(1), 1)
            out = out * head_mask.to(dtype)

        out = out.reshape(B, N_pad, self.all_head_size)

        if pad_len > 0:
            out = out[:, :N, :]

        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)

        return (output, None)


@ATTENTION_REGISTRY.register("vig_inject")
@ATTENTION_REGISTRY.register("vig_attention_inject")
@ATTENTION_REGISTRY.register("inject_vig")
def inject_vig(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    **kwargs: Any,
) -> Optional[ViGAttention]:
    """Safe injection helper for ViG attention."""
    if old_attn is None:
        return None

    if model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    if model_config is None and hasattr(old_attn, "config"):
        model_config = old_attn.config

    hidden_size = (
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
    num_heads = (
        _cfg_get(model_config, "num_attention_heads")
        or _cfg_get(model_config, "num_heads")
        or _cfg_get(model_config, "n_heads")
        or getattr(old_attn, "num_heads", None)
        or getattr(old_attn, "num_attention_heads", None)
        or 12
    )

    if model_config is None:
        import types
        model_config = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=getattr(old_attn, "num_key_value_heads", num_heads),
            is_decoder=getattr(old_attn, "is_decoder", False),
        )

    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

    try:
        new_attn = ViGAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=old_attn,
            **kwargs,
        )

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_attn)
                return new_attn

        layer.attention = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_vig failed at layer %d: %s", layer_idx, e)
        return None
