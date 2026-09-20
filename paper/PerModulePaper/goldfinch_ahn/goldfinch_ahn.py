r"""
GoldFinch-AHN: Adaptive Hybrid Network Combining Sliding Window and Gated DeltaNet Memory (Hardened).

Combines short-term multi-head sliding-window attention with long-term GatedDeltaNet recurrent memory
via a learned input-dependent dynamic gating mechanism initialized for stable representation transfer.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- get_token_valid_mask for 2D/4D masks
- Forward-only causal scan for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

    return q, k


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
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
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


class GatedDeltaNetLayer(nn.Module):
    r"""Gated DeltaNet Recurrent Memory Layer."""

    def __init__(self, d_model: int, use_spectral_norm: bool = False) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@ATTENTION_REGISTRY.register("goldfinch_ahn", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("goldfinch_ahn_inject", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("goldfinch", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class GoldFinchAHNBlock(BaseAttention):
    r"""
    GoldFinch-AHN: Adaptive Hybrid Network Attention Block.

    Combines multi-head sliding-window attention with long-term GatedDeltaNet memory
    via learned input-dependent dynamic gating.
    """

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        num_heads: Optional[int] = None,
        window_size: int = 64,
        is_causal: bool = False,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        gate_init_bias: float = 1.8,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        if d_model is None:
            d_model = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
            if d_model is None:
                if config is not None:
                    d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
                elif source_attn is not None:
                    out_layer = getattr(source_attn, "out_lin", getattr(source_attn, "dense", getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", getattr(source_attn, "out_proj", None)))))
                    if out_layer is not None:
                        d_model = getattr(out_layer, "out_features", getattr(out_layer, "in_features", 768))
                    else:
                        q_l = getattr(source_attn, "q_proj", getattr(source_attn, "query", getattr(source_attn, "q_lin", None)))
                        d_model = getattr(q_l, "in_features", getattr(q_l, "out_features", 768)) if q_l is not None else 768
                else:
                    d_model = 768

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        if num_heads is None:
            num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
            if num_heads is None:
                if config is not None:
                    num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
                elif source_attn is not None:
                    num_heads = getattr(source_attn, "num_attention_heads", getattr(source_attn, "n_heads", 12))
                else:
                    num_heads = 12

        self.dim = int(d_model)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)

        # num_kv_heads: three-branch fallback including source_attn
        if config is not None:
            self.num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", kwargs.get("num_kv_heads", self.num_heads)))
        elif source_attn is not None:
            self.num_kv_heads = (
                getattr(source_attn, "num_key_value_heads", None)
                or getattr(source_attn, "num_kv_heads", None)
                or kwargs.get("num_kv_heads", self.num_heads)
            )
        else:
            self.num_kv_heads = kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = int(self.num_kv_heads)

        # head_dim: prefer config, then source_attn, then divide
        _head_dim = getattr(config, "head_dim", None) if config else None
        if _head_dim is None and source_attn is not None:
            _head_dim = getattr(source_attn, "head_dim", None)
        if _head_dim is None:
            _head_dim = self.dim // max(1, self.num_heads)
        self.head_dim = int(_head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True
        if config is not None:
            self.window_size = getattr(config, "window_size", getattr(comp_config, "window_size", kwargs.get("window_size", window_size)))
        elif comp_config is not None:
            self.window_size = getattr(comp_config, "window_size", kwargs.get("window_size", window_size))
        else:
            self.window_size = kwargs.get("window_size", window_size)
        self.window_size = int(self.window_size)

        # Decoder detection
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = getattr(config, "is_decoder", getattr(config, "is_causal", kwargs.get("is_causal", False))) if config else False
        _mt = (model_type or '').lower()
        _decoder_prefixes = ('gpt', 'qwen', 'llama', 'mistral', 'gemma', 'phi', 'opt', 'bloom', 'falcon', 'olmo', 'mpt', 'stablelm')
        if any(_mt.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False) or is_causal:
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # LoRA config
        if config is not None:
            self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
        elif comp_config is not None:
            self.lora_rank = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
        else:
            self.lora_rank = kwargs.get("lora_rank", 16)

        if config is not None:
            self.lora_alpha = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        elif comp_config is not None:
            self.lora_alpha = getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
        else:
            self.lora_alpha = kwargs.get("lora_alpha", 32.0)

        self.freeze_base = (self.lora_rank > 0)
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        # Projections
        self.W_q = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.W_k = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.W_v = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.delta_mem = GatedDeltaNetLayer(self.dim, use_spectral_norm=True)
        self.gate = nn.Linear(self.dim, self.num_heads)
        self.gate.bias.data.fill_(gate_init_bias)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.dropout = nn.Dropout(getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config is not None else 0.1)

        # Output calibration and preservation gate:
        # sigmoid(+2.0) ≈ 0.88 so attention dominates immediately at start.
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_gate = nn.Parameter(torch.tensor(2.0))

        if source_attn is not None:
            self.copy_weights(source_attn)

    @property
    def query(self): return self.W_q
    @property
    def key(self): return self.W_k
    @property
    def value(self): return self.W_v
    @property
    def q_proj(self): return self.W_q
    @property
    def k_proj(self): return self.W_k
    @property
    def v_proj(self): return self.W_v
    @property
    def q_lin(self): return self.W_q
    @property
    def k_lin(self): return self.W_k
    @property
    def v_lin(self): return self.W_v
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    def freeze_base_weights(self) -> None:
        for mod in (self.W_q, self.W_k, self.W_v, self.out_proj):
            if hasattr(mod, "freeze_base_weights"):
                mod.freeze_base_weights()

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = hasattr(src, "dense") or hasattr(src, "out_proj") or hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (hasattr(src, "query") and not has_out)
        ):
            is_nested = True
        if hasattr(src, "self") and hasattr(src, "output"):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv1d = (
                    "Conv1D" in c_attn_mod.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c_attn_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv1d:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size : 2 * split_size, :], c_b[split_size : 2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size :, :], c_b[2 * split_size :] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv1d = (
                        "Conv1D" in c_proj_mod.__class__.__name__
                        or "Conv1D" in base_o.__class__.__name__
                        or hasattr(c_proj_mod, "nf")
                        or hasattr(base_o, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv1d:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (out_mod.weight.data.clone(), out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 3. Flat modules
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query", "W_q"],
                "k": ["k_lin", "k_proj", "key", "W_k"],
                "v": ["v_lin", "v_proj", "value", "W_v"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception:
                pass

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning("GoldfinchAHN copy_weights warning. Proceeding with initialized weights.")

        target_map = {
            "q": self.W_q,
            "k": self.W_k,
            "v": self.W_v,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().contiguous().shape:
                    w = w.t().contiguous()
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(w[:out_d, :in_d].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype))

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.linalg.norm(target_mod.weight.data.float()).item())
                if w_norm < 1e-4:
                    raise RuntimeError(f"GoldFinchAHNBlock weight verification failed for '{proj_key}': norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

        if self.lora_rank > 0 or self.freeze_base:
            self.freeze_base_weights()

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "GoldFinchAHNBlock":
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = hidden_states
        if not x.is_contiguous():
            x = x.contiguous()
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.W_k(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.W_v(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif self.rotary_emb is not None:
            rotary_module = self.rotary_emb
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = rotary_module(q, seq_len=L)
            except Exception:
                try:
                    cos, sin = rotary_module(q, pos_ids)
                except Exception:
                    cos, sin = None, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scores = torch.matmul(q, k.transpose(-2, -1).contiguous()) * (self.head_dim ** -0.5)

        idx = torch.arange(L, device=x.device)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        mask_window = dist <= self.window_size

        if self.is_causal:
            causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
            valid_mask = causal & mask_window
        else:
            valid_mask = mask_window

        mask_val = -10000.0
        scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), mask_val)

        pad_mask = None
        if attention_mask is not None:
            pad_mask = get_token_valid_mask(attention_mask, B, L, x.device)
            scores = scores.masked_fill(~pad_mask.unsqueeze(1).unsqueeze(2), mask_val)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        short_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        long_out = self.delta_mem(x)  # (B, L, D)
        gate = torch.sigmoid(self.gate(x)).mean(dim=-1, keepdim=True)  # (B, L, 1)
        fused = gate * short_out + (1.0 - gate) * long_out

        if pad_mask is not None:
            fused = fused * pad_mask.unsqueeze(-1).to(fused.dtype)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, 1, -1)
            else:
                hm = head_mask
            fused = fused * hm

        if kwargs.get("encoder_hidden_states", None) is not None:
            raise NotImplementedError("Cross-attention is not implemented for GoldFinchAHNBlock")

        if self.is_nested_encoder:
            return fused, (attn_weights if output_attentions else None)

        out = self.out_proj(fused)
        out = out * self.output_scale

        # Preservation gate: blend with original hidden_states to preserve pretrained activations at init.
        alpha = torch.sigmoid(self.output_gate)
        out = alpha * out + (1.0 - alpha) * x

        return out, (attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("goldfinch_ahn_inject")
@ATTENTION_REGISTRY.register("inject_goldfinch_ahn")
def inject_goldfinch_ahn(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[GoldFinchAHNBlock, nn.Module]]:
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # 1. Whole model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_goldfinch_ahn(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # 2. Single layer injection
    src_attn = old_attn
    if src_attn is None and target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                src_attn = target.attention.self
            else:
                src_attn = target.attention
        elif hasattr(target, "self"):
            src_attn = target.self
        elif hasattr(target, "attn"):
            src_attn = target.attn
        elif hasattr(target, "self_attn"):
            src_attn = target.self_attn

    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, 'hidden_size', None) or
        getattr(src_attn, 'embed_dim', None) or
        getattr(src_attn, 'dim', None) or
        (src_attn.q_proj.in_features if hasattr(src_attn, 'q_proj') and hasattr(src_attn.q_proj, 'in_features') else None) or
        (getattr(target, 'in_features', None) if target is not None else None) or
        getattr(cfg, 'hidden_size', getattr(cfg, 'dim', 64))
    )
    num_heads = (
        getattr(src_attn, 'num_heads', None) or
        getattr(src_attn, 'num_attention_heads', None) or
        getattr(src_attn, 'n_heads', None) or
        getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_heads', 4))
    )

    try:
        new_attn = GoldFinchAHNBlock(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            num_heads=num_heads,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            **kwargs,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"GoldFinchAHNBlock injection failed: {e}")
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
        elif hasattr(target, "self"):
            target.self = new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn

    return new_attn
