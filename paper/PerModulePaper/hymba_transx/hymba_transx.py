"""HymbaTransX: Hybrid Differential Transformer & State Space Model Block (Production-Hardened).

Differential attention with rotary embeddings coupled with a State Space Model branch.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings (both external position_embeddings and internal HeadRoPE fallback)
- Causal masking for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.attentions.simple_ssm import SimpleSSM
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


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


class HeadRoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(1), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, d = x.shape
        if L > self.max_seq_len:
            self._build_cache(L)
            self.max_seq_len = L

        cos = self.cos_cached[:, :, :L, :d]
        sin = self.sin_cached[:, :, :L, :d]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos[..., 0::2] - x_odd * sin[..., 0::2]
        rot_odd = x_odd * cos[..., 0::2] + x_even * sin[..., 0::2]
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


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


@ATTENTION_REGISTRY.register("hymba_transx", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class HymbaTransXBlock(BaseAttention):
    r"""HymbaTransX: Hybrid Differential Transformer & State Space Model Block."""

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
        if hidden_size is None:
            if config is not None:
                hidden_size = getattr(config, "hidden_size", getattr(config, "dim", getattr(config, "d_model", getattr(config, "n_embd", None))))
            if hidden_size is None and source_attn is not None:
                for attr in ["hidden_size", "embed_dim", "dim", "d_model"]:
                    if hasattr(source_attn, attr) and getattr(source_attn, attr) is not None:
                        hidden_size = getattr(source_attn, attr)
                        break
                if hidden_size is None:
                    out_layer = getattr(source_attn, "out_lin", getattr(source_attn, "dense", getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None))))
                    hidden_size = getattr(out_layer, "out_features", None) if out_layer is not None else None
            if hidden_size is None:
                hidden_size = 768

        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", getattr(config, "num_heads", None)))
            if num_heads is None and source_attn is not None:
                num_heads = getattr(source_attn, "num_attention_heads", getattr(source_attn, "n_heads", getattr(source_attn, "num_heads", None)))
            if num_heads is None:
                num_heads = hidden_size // 64 if (hidden_size and hidden_size % 64 == 0) else 12

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", self.num_heads)) if config else self.num_heads

        cfg_head_dim = getattr(config, "head_dim", kwargs.get("head_dim", None)) if config else kwargs.get("head_dim", None)
        if cfg_head_dim is None and source_attn is not None:
            for attr in ("q_proj", "query", "q_lin", "W_q1"):
                mod = getattr(source_attn, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    cfg_head_dim = mod.weight.shape[0] // max(1, self.num_heads)
                    break

        if cfg_head_dim is not None:
            self.head_dim = int(cfg_head_dim)
        else:
            self.head_dim = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # Decoder detection
        _mt = str(getattr(config, "model_type", "") or "").lower() if config else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        is_dec = (
            getattr(config, "is_decoder", False)
            or getattr(config, "causal", False)
            or any(_mt.startswith(p) for p in _decoder_prefixes)
        ) if config else False
        if hasattr(config, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # LoRA config
        self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))) if config else 16
        self.lora_alpha = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))) if config else 32.0
        self.freeze_base = (self.lora_rank > 0)

        self.rope = HeadRoPE(self.head_dim)
        self.W_q1 = LoRALinear(self.dim, self.all_head_size, bias=False, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.W_k1 = LoRALinear(self.dim, self.all_kv_head_size, bias=False, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.W_q2 = LoRALinear(self.dim, self.all_head_size, bias=False, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.W_k2 = LoRALinear(self.dim, self.all_kv_head_size, bias=False, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.W_v = LoRALinear(self.dim, self.all_kv_head_size, bias=False, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.mamba = SimpleSSM(d_model=self.dim, config=config, is_decoder=self.is_decoder)
        self.alpha = nn.Parameter(torch.tensor(-1.0))

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    @property
    def query(self) -> nn.Module:
        return self.W_q1

    @property
    def key(self) -> nn.Module:
        return self.W_k1

    @property
    def value(self) -> nn.Module:
        return self.W_v

    @property
    def q_proj(self) -> nn.Module:
        return self.W_q1

    @property
    def k_proj(self) -> nn.Module:
        return self.W_k1

    @property
    def v_proj(self) -> nn.Module:
        return self.W_v

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(getattr(src, "self"), "query")
            and hasattr(src, "output")
            and hasattr(getattr(src, "output"), "dense")
        ):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
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
                cls_c = getattr(c_attn_mod, "__class__", type(c_attn_mod)).__name__
                base_cls_c = getattr(base_c, "__class__", type(base_c)).__name__
                if "Conv1D" in cls_c or "Conv1D" in base_cls_c or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1] and "Linear" not in cls_c and "Linear" not in base_cls_c):
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
                    cls_o = getattr(c_proj_mod, "__class__", type(c_proj_mod)).__name__
                    base_cls_o = getattr(base_o, "__class__", type(base_o)).__name__
                    if "Conv1D" in cls_o or "Conv1D" in base_cls_o or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1] and "Linear" not in cls_o and "Linear" not in base_cls_o):
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
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
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
            except Exception as e:
                logger.warning(f"UniversalAdapter extraction in HymbaTransXBlock failed: {e}")

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                f"HymbaTransXBlock copy_weights: missing projections {missing_after} from {src.__class__.__name__}; keeping defaults."
            )

        with torch.no_grad():
            w_q, _ = extracted["q"]
            w_k, _ = extracted["k"]
            w_v, _ = extracted["v"]
            w_o, b_o = extracted["o"]

            # Query 1 & 2
            if w_q is not None:
                if w_q.ndim != 2:
                    w_q = w_q.reshape(w_q.size(0), -1)
                if w_q.shape != self.W_q1.weight.shape and self.W_q1.weight.shape == w_q.t().contiguous().shape:
                    w_q = w_q.t().contiguous()
                out_d = min(self.W_q1.weight.size(0), w_q.size(0))
                in_d = min(self.W_q1.weight.size(1), w_q.size(1))
                self.W_q1.weight.data[:out_d, :in_d].copy_(w_q[:out_d, :in_d].to(device=self.W_q1.weight.device, dtype=self.W_q1.weight.dtype))

            # Decorrelate W_q2 and W_k2 via independent Kaiming init
            nn.init.kaiming_uniform_(self.W_q2.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_k2.weight, a=math.sqrt(5))

            # Key 1
            if w_k is not None:
                if w_k.ndim != 2:
                    w_k = w_k.reshape(w_k.size(0), -1)
                if w_k.shape != self.W_k1.weight.shape and self.W_k1.weight.shape == w_k.t().contiguous().shape:
                    w_k = w_k.t().contiguous()
                out_d = min(self.W_k1.weight.size(0), w_k.size(0))
                in_d = min(self.W_k1.weight.size(1), w_k.size(1))
                self.W_k1.weight.data[:out_d, :in_d].copy_(w_k[:out_d, :in_d].to(device=self.W_k1.weight.device, dtype=self.W_k1.weight.dtype))

            # Value
            if w_v is not None:
                if w_v.ndim != 2:
                    w_v = w_v.reshape(w_v.size(0), -1)
                if w_v.shape != self.W_v.weight.shape and self.W_v.weight.shape == w_v.t().contiguous().shape:
                    w_v = w_v.t().contiguous()
                out_d = min(self.W_v.weight.size(0), w_v.size(0))
                in_d = min(self.W_v.weight.size(1), w_v.size(1))
                self.W_v.weight.data[:out_d, :in_d].copy_(w_v[:out_d, :in_d].to(device=self.W_v.weight.device, dtype=self.W_v.weight.dtype))

            # Output
            if w_o is not None:
                if w_o.ndim != 2:
                    w_o = w_o.reshape(w_o.size(0), -1)
                if w_o.shape != self.out_proj.weight.shape and self.out_proj.weight.shape == w_o.t().contiguous().shape:
                    w_o = w_o.t().contiguous()
                out_d = min(self.out_proj.weight.size(0), w_o.size(0))
                in_d = min(self.out_proj.weight.size(1), w_o.size(1))
                self.out_proj.weight.data[:out_d, :in_d].copy_(w_o[:out_d, :in_d].to(device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype))
                if b_o is not None and self.out_proj.bias is not None:
                    min_b = min(self.out_proj.bias.numel(), b_o.numel())
                    self.out_proj.bias.data[:min_b].copy_(b_o[:min_b].to(device=self.out_proj.bias.device, dtype=self.out_proj.bias.dtype))
            elif self.is_nested_encoder:
                nn.init.eye_(self.out_proj.weight)
                if self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)

            # Norm checks
            for name, mod in [("W_q1", self.W_q1), ("W_k1", self.W_k1), ("W_v", self.W_v), ("out_proj", self.out_proj)]:
                w_norm = float(torch.linalg.norm(mod.weight.data.float()).item())
                if w_norm < 1e-4:
                    logger.warning(f"HymbaTransXBlock weight verification warning for '{name}': norm too small ({w_norm:.2e}), reinitializing")
                    nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
                if self.freeze_base:
                    mod.freeze_base_weights()

            if self.freeze_base:
                self.W_q2.freeze_base_weights()
                self.W_k2.freeze_base_weights()

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention is not supported by HymbaTransXBlock.")

        x = hidden_states if hidden_states is not None else kwargs.get("query", kwargs.get("x", None))
        if x is None and len(args) > 0:
            x = args[0]
        if attention_mask is None:
            attention_mask = kwargs.get("mask", kwargs.get("attn_mask", None))

        B, L, D = x.shape

        q1 = self.W_q1(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k1 = self.W_k1(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        q2 = self.W_q2(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k2 = self.W_k2(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.W_v(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
            q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)
        else:
            q1 = self.rope(q1)
            k1 = self.rope(k1)
            q2 = self.rope(q2)
            k2 = self.rope(k2)

        if self.kv_repeat > 1:
            k1 = repeat_kv(k1, self.kv_repeat)
            k2 = repeat_kv(k2, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores1 = torch.matmul(q1.float(), k1.float().transpose(-2, -1).contiguous()) * scale
        scores2 = torch.matmul(q2.float(), k2.float().transpose(-2, -1).contiguous()) * scale

        if self.is_decoder:
            causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            scores1 = scores1.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
            scores2 = scores2.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid_mask = get_token_valid_mask(attention_mask, B, L, x.device)
                if valid_mask is None:
                    valid_mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
                scores1 = scores1.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)
                scores2 = scores2.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)
            elif attention_mask.dim() == 4:
                scores1 = scores1 + attention_mask.clamp(min=-1e4)
                scores2 = scores2 + attention_mask.clamp(min=-1e4)

        attn1 = F.softmax(scores1, dim=-1)
        attn2 = F.softmax(scores2, dim=-1)
        diff_attn = F.relu(attn1 - 0.2 * attn2)
        denom = diff_attn.sum(dim=-1, keepdim=True)
        diff_attn = diff_attn / (denom + 1e-6)
        diff_attn = self.dropout(diff_attn).to(v.dtype)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            diff_attn = diff_attn * hm

        A_out = torch.matmul(diff_attn, v).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        if self.is_decoder:
            mamba_res = self.mamba(x, attention_mask=attention_mask)
            M_out = mamba_res[0] if isinstance(mamba_res, tuple) else mamba_res
        else:
            x_rev = torch.flip(x, dims=[1])
            mamba_fwd = self.mamba(x, attention_mask=attention_mask)
            mamba_bwd = self.mamba(x_rev, attention_mask=attention_mask)
            mf = mamba_fwd[0] if isinstance(mamba_fwd, tuple) else mamba_fwd
            mb = mamba_bwd[0] if isinstance(mamba_bwd, tuple) else mamba_bwd
            M_out = 0.5 * (mf + torch.flip(mb, dims=[1]))

        alpha = torch.sigmoid(self.alpha)
        combined = alpha * A_out + (1.0 - alpha) * M_out

        if self.is_nested_encoder:
            out = combined
        else:
            out = self.out_proj(combined)

        return out, (diff_attn if output_attentions else None)


@ATTENTION_REGISTRY.register("hymba_transx_inject")
@ATTENTION_REGISTRY.register("inject_hymba_transx")
def inject_hymba_transx(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Any]:
    """Safe injection hook for HymbaTransX hybrid attention module."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None)

    # 1. Whole model traversal support
    if (
        old_attn is None
        and target is not None
        and (
            hasattr(target, "transformer")
            or hasattr(target, "distilbert")
            or hasattr(target, "roberta")
            or hasattr(target, "bert")
            or (hasattr(target, "model") and hasattr(target.model, "layers"))
            or (hasattr(target, "encoder") and hasattr(target.encoder, "layer"))
        )
        and not (
            hasattr(target, "q_lin")
            or hasattr(target, "q_proj")
            or hasattr(target, "c_attn")
            or hasattr(target, "query")
        )
    ):
        try:
            layer_list = None
            if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
                layer_list = target.distilbert.transformer.layer
            elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
                layer_list = target.roberta.encoder.layer
            elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
                layer_list = target.bert.encoder.layer
            elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
                layer_list = target.transformer.h
            elif hasattr(target, "transformer") and hasattr(target.transformer, "layer"):
                layer_list = target.transformer.layer
            elif hasattr(target, "model") and hasattr(target.model, "layers"):
                layer_list = target.model.layers
            elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
                layer_list = target.encoder.layer

            if layer_list is not None:
                tot = len(layer_list)
                for idx, lyr in enumerate(layer_list):
                    inject_hymba_transx(
                        layer=lyr,
                        model_config=cfg,
                        comp_config=comp_config,
                        layer_idx=idx,
                        total_layers=tot,
                        **kwargs,
                    )
                return target
        except Exception as e:
            logger.error("inject_hymba_transx whole-model traversal failed: %s", e)
            return None

    # 2. Named modules whole-model fallback
    if (
        old_attn is None
        and target is not None
        and hasattr(target, "named_modules")
        and not (
            hasattr(target, "attention")
            or hasattr(target, "self")
            or hasattr(target, "attn")
            or hasattr(target, "q_proj")
            or hasattr(target, "q_lin")
            or hasattr(target, "query")
            or hasattr(target, "c_attn")
        )
    ):
        try:
            replaced = 0
            for name, module in list(target.named_modules()):
                if hasattr(module, "q_lin") or hasattr(module, "query") or hasattr(module, "q_proj") or hasattr(module, "c_attn"):
                    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                    child_name = name.rsplit(".", 1)[-1] if "." in name else name
                    parent = target.get_submodule(parent_name) if parent_name else target
                    new_attn = inject_hymba_transx(
                        layer=None,
                        model_config=cfg,
                        comp_config=comp_config,
                        old_attn=module,
                        **kwargs,
                    )
                    if new_attn is not None:
                        setattr(parent, child_name, new_attn)
                        replaced += 1
            if replaced > 0:
                return target
        except Exception as e:
            logger.error("inject_hymba_transx named_modules replacement failed: %s", e)
            return None

    src_attn = old_attn if old_attn is not None else (
        getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None)))
        if layer is not None else None
    )

    hidden_size = kwargs.get("hidden_size")
    if hidden_size is None and cfg is not None:
        hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "dim", getattr(cfg, "d_model", getattr(cfg, "n_embd", None))))
    if hidden_size is None and src_attn is not None:
        for attr in ["hidden_size", "embed_dim", "dim", "d_model"]:
            if hasattr(src_attn, attr) and getattr(src_attn, attr) is not None:
                hidden_size = getattr(src_attn, attr)
                break
        if hidden_size is None:
            for sub in ["q_lin", "q_proj", "query", "in_proj", "c_attn"]:
                if hasattr(src_attn, sub):
                    mod = getattr(src_attn, sub)
                    if hasattr(mod, "in_features"):
                        hidden_size = mod.in_features
                        break
                    elif hasattr(mod, "weight") and mod.weight is not None:
                        if hasattr(mod, "nf"):
                            hidden_size = mod.weight.shape[0]
                        else:
                            hidden_size = mod.weight.shape[1] if mod.weight.ndim == 2 else None
                        if hidden_size is not None:
                            break
    if hidden_size is None and layer is not None:
        hidden_size = getattr(layer, "in_features", None)
    if hidden_size is None:
        hidden_size = 768

    num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
    if num_heads is None and cfg is not None:
        num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", getattr(cfg, "num_heads", None)))
    if num_heads is None and src_attn is not None:
        num_heads = getattr(src_attn, "num_attention_heads", getattr(src_attn, "n_heads", getattr(src_attn, "num_heads", None)))
    if num_heads is None:
        num_heads = hidden_size // 64 if (hidden_size and hidden_size % 64 == 0) else 12

    try:
        new_attn = HymbaTransXBlock(
            config=cfg,
            comp_config=comp_config,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
        if layer is not None:
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                else:
                    layer.attention = new_attn
                    new_attn.is_nested_encoder = False
            elif hasattr(layer, "self"):
                layer.self = new_attn
                new_attn.is_nested_encoder = True
            elif hasattr(layer, "attn"):
                layer.attn = new_attn
                new_attn.is_nested_encoder = False
            elif hasattr(layer, "self_attn"):
                layer.self_attn = new_attn
                new_attn.is_nested_encoder = False
            else:
                new_attn.is_nested_encoder = False
        return new_attn
    except Exception as e:
        logger.error("inject_hymba_transx failed: %s", e)
        return None

