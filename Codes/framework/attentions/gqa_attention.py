r"""
Grouped-Query Attention (GQA).

Interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA) by dividing
$H_Q$ query heads into $G$ groups, where each group shares a single Key-Value head pair ($H_{KV} = G \le H_Q$).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if module is None:
        return None, None
    if hasattr(module, "base_layer"):
        module = module.base_layer
    w = getattr(module, "weight", None)
    b = getattr(module, "bias", None)
    if w is not None:
        w = w.data
        if hasattr(module, "nf") or module.__class__.__name__ == "Conv1D":
            w = w.t().contiguous()
    if b is not None:
        b = b.data
    return w, b


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

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = min(cos.shape[-1], q.shape[-1])
    rotary_dim = (rotary_dim // 2) * 2
    if rotary_dim <= 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos_cut = cos[..., :rotary_dim].to(q_rot.dtype)
    sin_cut = sin[..., :rotary_dim].to(q_rot.dtype)

    q_rot_embed = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot_embed = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
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


@ATTENTION_REGISTRY.register("gqa", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("gqa_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("grouped_query_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class GroupedQueryAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else old_attn

        if dim is None:
            if config is not None:
                dim = (
                    _cfg_get(config, "hidden_size", None)
                    or _cfg_get(config, "d_model", None)
                    or _cfg_get(config, "n_embd", None)
                    or _cfg_get(config, "dim", None)
                )
        if dim is None and src_attn is not None:
            dim = getattr(src_attn, "hidden_size", getattr(src_attn, "embed_dim", None))
            if dim is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(src_attn, cand, None)
                    if m is not None:
                        w, _ = _unpack_weight_bias(m)
                        if w is not None and w.dim() == 2:
                            dim = w.shape[1]
                            break
        self.hidden_size: int = int(dim or kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768))))

        if num_heads is None:
            if config is not None:
                num_heads = _cfg_get(config, "num_attention_heads", _cfg_get(config, "num_heads", _cfg_get(config, "n_head", None)))
        if num_heads is None and src_attn is not None:
            num_heads = getattr(src_attn, "num_heads", getattr(src_attn, "num_attention_heads", getattr(src_attn, "n_head", None)))
        self.num_heads: int = int(num_heads or kwargs.get("num_attention_heads", kwargs.get("num_heads", 8)))

        if num_kv_heads is None:
            if config is not None:
                num_kv_heads = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        if num_kv_heads is None and src_attn is not None:
            num_kv_heads = getattr(src_attn, "num_key_value_heads", getattr(src_attn, "num_kv_heads", None))
        if num_kv_heads is None:
            num_kv_heads = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", max(1, self.num_heads // 4)))
        self.num_kv_heads: int = int(min(self.num_heads, max(1, int(num_kv_heads))))

        self.head_dim: int = self.hidden_size // max(1, self.num_heads)
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        model_type = str(_cfg_get(config, "model_type", "")).lower()
        is_dec = _cfg_get(config, "is_decoder", _cfg_get(config, "is_causal", kwargs.get("is_causal", kwargs.get("is_decoder", False))))
        _mt = (model_type or '').lower()
        _decoder_prefixes = ('gpt', 'qwen', 'llama', 'mistral', 'gemma', 'phi', 'opt', 'bloom', 'falcon', 'olmo', 'mpt', 'stablelm')
        if any(_mt.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_nested_encoder: bool = False

        comp_config = getattr(config, "compression_config", None)
        self.lora_rank: int = int(_cfg_get(comp_config, "lora_rank", _cfg_get(config, "lora_rank", kwargs.get("lora_rank", lora_rank))))
        self.lora_alpha: float = float(_cfg_get(comp_config, "lora_alpha", _cfg_get(config, "lora_alpha", kwargs.get("lora_alpha", lora_alpha))))
        self.freeze_base: bool = (self.lora_rank > 0)

        self.query = LoRALinear(self.hidden_size, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.key = LoRALinear(self.hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.value = LoRALinear(self.hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.hidden_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(src_attn, "rotary_emb", None)

        if src_attn is not None:
            self.copy_weights(src_attn)

    def copy_weights(self, src: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            weight, bias_data = _unpack_weight_bias(src.c_attn)
            if weight is not None:
                split_size = weight.shape[0] // 3
                extracted["q"] = (weight[:split_size, :], bias_data[:split_size] if bias_data is not None else None)
                extracted["k"] = (weight[split_size : 2 * split_size, :], bias_data[split_size : 2 * split_size] if bias_data is not None else None)
                extracted["v"] = (weight[2 * split_size :, :], bias_data[2 * split_size :] if bias_data is not None else None)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            w_o, b_o = _unpack_weight_bias(src.c_proj)
            extracted["o"] = (w_o, b_o)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(src, "self")
            if hasattr(s, "query"):
                extracted["q"] = _unpack_weight_bias(s.query)
            if hasattr(s, "key"):
                extracted["k"] = _unpack_weight_bias(s.key)
            if hasattr(s, "value"):
                extracted["v"] = _unpack_weight_bias(s.value)

        if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense") and extracted["o"][0] is None:
            extracted["o"] = _unpack_weight_bias(src.output.dense)

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        w, b = _unpack_weight_bias(getattr(src, name))
                        if w is not None:
                            extracted[proj] = (w, b)
                            break

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("GroupedQueryAttention: missing weights for projections: %s, using initialized weights", missing)

        def _orient(w: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
            if w.dim() != 2:
                return w
            if w.shape == target_shape:
                return w
            if w.shape == (target_shape[1], target_shape[0]):
                return w.t().contiguous()
            if w.shape[1] != target_shape[1] and w.shape[0] == target_shape[1]:
                return w.t().contiguous()
            return w

        # Copy into projection modules
        with torch.no_grad():
            for proj_key, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    target_w = target.weight
                    w_oriented = _orient(w, (target_w.shape[0], target_w.shape[1]))
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

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

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
        if encoder_hidden_states is not None:
            return hidden_states, None

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        key_value_states = kwargs.get("key_value_states", None)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K = self.key(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.value(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # Apply RoPE
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))

        # GQA repeat
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = d ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale  # [B, H, N_q, N_k]

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if mask is not None:
            if mask.dim() == 4:
                scores = scores + mask.to(scores.dtype)
            else:
                valid_m = get_token_valid_mask(mask, B, N_k, device)
                if valid_m is not None:
                    scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)

        # Explicit Causal Masking for decoders
        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in ("gpt2", "gpt", "qwen", "qwen2", "qwen2.5", "qwen3.5", "llama")
        )
        if is_decoder and N_q == N_k:
            causal_tril = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_tril, -10000.0)

        attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        if output.shape[-1] != hidden_states.shape[-1]:
            if output.shape[-1] < hidden_states.shape[-1]:
                output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1]))
            else:
                output = output[..., : hidden_states.shape[-1]]

        return output, (attn if output_attentions else None)


def _is_single_layer(obj: Any) -> bool:
    """Return True if obj is a transformer layer, not a full model."""
    if obj is None:
        return False
    single_layer_classes = (
        "TransformerBlock", "BertLayer", "RobertaLayer",
        "GPT2Block", "Qwen2DecoderLayer", "LlamaDecoderLayer",
    )
    cls_name = getattr(obj, "__class__", type(obj)).__name__
    if cls_name in single_layer_classes:
        return True
    if (
        hasattr(obj, "attention")
        or hasattr(obj, "attn")
        or hasattr(obj, "self_attn")
        or hasattr(obj, "self")
    ) and not (
        hasattr(obj, "transformer")
        or hasattr(obj, "distilbert")
        or hasattr(obj, "roberta")
        or hasattr(obj, "bert")
        or hasattr(obj, "encoder")
        or hasattr(obj, "model")
    ):
        return True
    return False


_INJECT_DEPTH = 0
_MAX_DEPTH = 4


@ATTENTION_REGISTRY.register("gqa_inject")
@ATTENTION_REGISTRY.register("gqa")
@ATTENTION_REGISTRY.register("grouped_query_attention")
def inject_gqa(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    lora_rank: int = 16,
    **kwargs: Any,
) -> Optional[Any]:
    r"""Injection hook for GroupedQueryAttention supporting single layers and whole models."""
    global _INJECT_DEPTH
    if _INJECT_DEPTH >= _MAX_DEPTH:
        logger.error("Max injection depth reached in inject_gqa - possible recursion")
        return None
    _INJECT_DEPTH += 1
    try:
        target = layer if layer is not None else model
        cfg = model_config or getattr(target, "config", None) or kwargs.get("config", None)

        # 1. Whole model traversal support
        if _is_single_layer(target):
            pass
        elif target is not None and (
            hasattr(target, "transformer")
            or hasattr(target, "distilbert")
            or hasattr(target, "roberta")
            or hasattr(target, "bert")
            or (hasattr(target, "model") and not hasattr(target, "forward"))
        ):
            layer_list = None
            if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
                layer_list = target.distilbert.transformer.layer
            elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
                layer_list = target.roberta.encoder.layer
            elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
                layer_list = target.bert.encoder.layer
            elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
                layer_list = target.transformer.h
            elif hasattr(target, "model") and hasattr(target.model, "layers"):
                layer_list = target.model.layers
            elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
                layer_list = target.encoder.layer

            if layer_list is not None:
                num_injected = 0
                tot = len(layer_list)
                for idx, lyr in enumerate(layer_list):
                    res = inject_gqa(
                        layer=lyr,
                        model_config=cfg,
                        comp_config=comp_config,
                        layer_idx=idx,
                        total_layers=tot,
                        lora_rank=lora_rank,
                        **kwargs,
                    )
                    if res is not None:
                        num_injected += 1
                return target if num_injected > 0 else None

        # 2. Single layer injection
        curr_attn = old_attn
        if curr_attn is None and target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    curr_attn = target.attention.self
                else:
                    curr_attn = target.attention
            elif hasattr(target, "self"):
                curr_attn = target.self
            elif hasattr(target, "attn"):
                curr_attn = target.attn
            elif hasattr(target, "self_attn"):
                curr_attn = target.self_attn

        if curr_attn is None and target is not None and hasattr(target, "forward"):
            curr_attn = target

        if curr_attn is None and cfg is None:
            return None

        try:
            new_layer = GroupedQueryAttention(
                config=cfg,
                old_attn=curr_attn,
                lora_rank=lora_rank,
                **kwargs,
            )
        except Exception as ex:
            logger.warning("Failed to instantiate GroupedQueryAttention: %s", ex)
            return None

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_layer
                else:
                    target.attention = new_layer
            elif hasattr(target, "self"):
                target.self = new_layer
            elif hasattr(target, "attn"):
                target.attn = new_layer
            elif hasattr(target, "self_attn"):
                target.self_attn = new_layer

        return new_layer
    finally:
        _INJECT_DEPTH -= 1
