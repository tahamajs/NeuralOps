"""
MHA-SDPA: A thin Multi-Head Attention wrapper around PyTorch's 
scaled_dot_product_attention.

DISCLAIMER:
    This module was previously named "SPECTRA-v2" and claimed to 
    implement a sparse per-head efficient context transformer 
    with intra-block attention, top-k inter-block routing, 
    adaptive spans, and RFF rectification. NONE of these 
    features are implemented here. What this module actually 
    does is:
    
      1. Project Q, K, V via LoRALinear (frozen base + adapters).
      2. Apply RoPE if position_embeddings are provided.
      3. Repeat KV heads for GQA.
      4. Call scaled_dot_product_attention.
      5. Apply head_mask.
      6. Project the output.
    
    This is functionally equivalent to a standard MHA with SDPA.
    It provides NO sparsity, NO top-k, NO adaptive span, and NO 
    RFF features.

If you need the actual SPECTRA architecture, see paper/_SPECTRA_V2_REPORT.md.

Registered as "mha_sdpa" (canonical) and "spectra_v2" (deprecated).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

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
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        if (module.__class__.__name__ == "Conv1D") or (base.__class__.__name__ == "Conv1D") or (w.ndim == 2 and w.shape[0] < w.shape[1]):
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    rotary_dim = x.shape[-1]
    rotary_dim = rotary_dim - (rotary_dim % 2)
    x_rot = x[..., :rotary_dim]
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    if x.shape[-1] > rotary_dim:
        return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)
    return rotated


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
    elif cos.dim() == 4:
        pass

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class LoRALinear(nn.Module):
    """Linear layer augmented with Low-Rank Adaptation (LoRA) and base freezing."""

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
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("mha_sdpa")
@ATTENTION_REGISTRY.register("spectra_v2")
@ATTENTION_REGISTRY.register("spectra_v2_attention")
class SpectraV2Attention(BaseAttention):
    """
    MHA wrapper around PyTorch's scaled_dot_product_attention.
    Previously named spectra_v2 (deprecated alias).
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        model_config: Optional[Any] = None,
        layer_idx: int = 0,
        total_layers: int = 12,
        bert_mode: bool = False,
        pretrained_attn: Optional[nn.Module] = None,
        rank: Optional[int] = 64,
        **kwargs: Any,
    ) -> None:
        cfg = config or model_config or kwargs.get("config", None)
        super().__init__(cfg, rank or 64)
        self.config = cfg
        comp_config = _cfg_get(cfg, "compression_config", kwargs.get("comp_config", None))

        self.hidden_size = kwargs.get(
            "hidden_size",
            _cfg_get(cfg, "hidden_size", _cfg_get(cfg, "dim", _cfg_get(cfg, "d_model", 768))),
        )
        self.dim = self.hidden_size
        self.num_heads = kwargs.get(
            "num_heads",
            kwargs.get(
                "num_attention_heads",
                _cfg_get(cfg, "num_attention_heads", _cfg_get(cfg, "num_heads", _cfg_get(cfg, "n_heads", 12))),
            ),
        )
        self.num_kv_heads = kwargs.get(
            "num_kv_heads",
            kwargs.get(
                "num_key_value_heads",
                _cfg_get(cfg, "num_key_value_heads", _cfg_get(cfg, "num_kv_heads", self.num_heads)),
            ),
        )
        self.head_dim = kwargs.get(
            "head_dim",
            _cfg_get(cfg, "head_dim", self.dim // self.num_heads if self.num_heads > 0 else 64),
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.bert_mode = bert_mode

        is_dec = _cfg_get(cfg, "is_decoder", kwargs.get("is_decoder", False))
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        self.lora_rank = _cfg_get(cfg, "lora_rank", _cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha = _cfg_get(cfg, "lora_alpha", _cfg_get(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base = (self.lora_rank > 0)

        # Projections
        self.query = LoRALinear(
            self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base
        )
        self.key = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base
        )
        self.value = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base
        )

        dropout_p = _cfg_get(cfg, "dropout", _cfg_get(cfg, "attention_probs_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn", None)
        if source_attn is not None:
            self.copy_weights(source_attn)

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_proj = hasattr(src, "out_proj") or hasattr(src, "dense") or hasattr(src, "c_proj") or hasattr(src, "out_lin")
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
        ) and not has_out_proj:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused c_attn / c_proj
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size : 2 * split_size, :], c_b[split_size : 2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size :, :], c_b[2 * split_size :] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                extracted["o"] = _unpack_weight_bias(src.c_proj)

        # 2. DistilBERT style: q_lin, k_lin, v_lin, out_lin
        if hasattr(src, "q_lin") and src.q_lin is not None:
            extracted["q"] = _unpack_weight_bias(src.q_lin)
            extracted["k"] = _unpack_weight_bias(getattr(src, "k_lin", None))
            extracted["v"] = _unpack_weight_bias(getattr(src, "v_lin", None))
            extracted["o"] = _unpack_weight_bias(getattr(src, "out_lin", None))

        # 3. BERT / RoBERTa style: query, key, value, dense
        if hasattr(src, "query") and src.query is not None and extracted["q"][0] is None:
            extracted["q"] = _unpack_weight_bias(src.query)
            extracted["k"] = _unpack_weight_bias(getattr(src, "key", None))
            extracted["v"] = _unpack_weight_bias(getattr(src, "value", None))
            extracted["o"] = _unpack_weight_bias(getattr(src, "dense", getattr(src, "out_proj", None)))

        # 4. Nested structures (RoBERTa / BERT container objects)
        if hasattr(src, "self") and hasattr(getattr(src, "self"), "query") and extracted["q"][0] is None:
            s = getattr(src, "self")
            extracted["q"] = _unpack_weight_bias(getattr(s, "query", None))
            extracted["k"] = _unpack_weight_bias(getattr(s, "key", None))
            extracted["v"] = _unpack_weight_bias(getattr(s, "value", None))
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                extracted["o"] = _unpack_weight_bias(getattr(getattr(src, "output"), "dense", None))

        # 5. Qwen / LLaMA / HF Attention style: q_proj, k_proj, v_proj, o_proj
        if hasattr(src, "q_proj") and src.q_proj is not None and extracted["q"][0] is None:
            extracted["q"] = _unpack_weight_bias(src.q_proj)
            extracted["k"] = _unpack_weight_bias(getattr(src, "k_proj", None))
            extracted["v"] = _unpack_weight_bias(getattr(src, "v_proj", None))
            extracted["o"] = _unpack_weight_bias(getattr(src, "o_proj", getattr(src, "dense", getattr(src, "out_proj", None))))

        # 6. UniversalAdapter fallback for custom names
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
                logger.debug("UniversalAdapter fallback failed: %s", e)

        target_map = {
            "q": self.query,
            "k": self.key,
            "v": self.value,
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
                    else:
                        logger.warning("Attention module copy_weights warning. Proceeding with initialized weights.")
                        continue

                src_norm = torch.linalg.norm(w.float()).item()

                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif w.shape[0] < target_mod.weight.shape[0] and target_mod.weight.shape[0] % w.shape[0] == 0:
                    repeats = target_mod.weight.shape[0] // w.shape[0]
                    target_mod.weight.copy_(w.repeat(repeats, 1))
                elif w.shape[0] > target_mod.weight.shape[0] and w.shape[0] % target_mod.weight.shape[0] == 0:
                    repeats = w.shape[0] // target_mod.weight.shape[0]
                    # For GQA: average the query head groups
                    w_grouped = w.view(target_mod.weight.shape[0], repeats, w.shape[1]).mean(dim=1)
                    target_mod.weight.copy_(w_grouped)
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    target_mod.weight.zero_()
                    target_mod.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])

                if b is not None and target_mod.bias is not None:
                    if b.shape == target_mod.bias.shape:
                        target_mod.bias.copy_(b)
                    elif b.shape[0] < target_mod.bias.shape[0] and target_mod.bias.shape[0] % b.shape[0] == 0:
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))
                    elif b.shape[0] > target_mod.bias.shape[0] and b.shape[0] % target_mod.bias.shape[0] == 0:
                        repeats = b.shape[0] // target_mod.bias.shape[0]
                        b_grouped = b.view(target_mod.bias.shape[0], repeats).mean(dim=1)
                        target_mod.bias.copy_(b_grouped)
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                fnorm = torch.linalg.norm(target_mod.weight.float()).item()
                if src_norm > 1e-3 and fnorm < 1e-4:
                    raise RuntimeError(
                        f"SpectraV2Attention: copy_weights for {proj_key} produced near-zero norm "
                        f"({fnorm:.2e}) from a valid source ({src_norm:.2e}). Aborting to prevent silent "
                        f"model corruption."
                    )

        if self.freeze_base:
            for mod in [self.query, self.key, self.value, self.out_proj]:
                mod.freeze_base_weights()

    _copy_pretrained = copy_weights

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "SpectraV2Attention":
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape

        is_cross = encoder_hidden_states is not None
        kv_states = encoder_hidden_states if is_cross else hidden_states
        N_k = kv_states.shape[1]

        eff_mask = encoder_attention_mask if is_cross else attention_mask

        Q = self.query(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(kv_states).view(B, N_k, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(kv_states).view(B, N_k, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply RoPE if provided
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # Repeat KV for GQA
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            K = repeat_kv(K, repeats)
            V = repeat_kv(V, repeats)

        scale = self.head_dim ** -0.5

        # Causal determination
        _is_causal = (
            getattr(self, "is_decoder", False)
            and not is_cross
            and N > 1
        )

        attn_out = None
        attn_weights = None

        if not output_attentions:
            try:
                _qs = Q.half() if Q.dtype == torch.float32 else Q
                _ks = K.half() if K.dtype == torch.float32 else K
                _vs = V.half() if V.dtype == torch.float32 else V

                attn_mask_for_sdpa = None
                if eff_mask is not None and not _is_causal:
                    if eff_mask.dim() == 2:
                        attn_mask_for_sdpa = eff_mask.bool().view(
                            eff_mask.shape[0], 1, 1, eff_mask.shape[1]
                        )
                    elif eff_mask.dim() == 4:
                        attn_mask_for_sdpa = eff_mask.bool()

                _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    attn_mask=attn_mask_for_sdpa,
                    is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )

                _sdpa_out = _sdpa_out.to(hidden_states.dtype)

                # Apply head_mask
                if head_mask is not None and head_mask.dim() >= 1:
                    hm = head_mask.view(-1, self.num_heads)
                    if hm.shape[0] == 1:
                        hm = hm.expand(_sdpa_out.shape[0], -1)
                    _sdpa_out = _sdpa_out * hm.view(
                        _sdpa_out.shape[0], self.num_heads, 1, 1
                    ).to(_sdpa_out.dtype)

                _sdpa_out = _sdpa_out.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
                _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)

                context = _sdpa_out
                if self.is_nested_encoder:
                    if context.shape[-1] != hidden_states.shape[-1]:
                        raise RuntimeError(
                            f"SpectraV2Attention: output dimension mismatch. "
                            f"Got {context.shape[-1]}, expected {hidden_states.shape[-1]}."
                        )
                    return context, None

                output = self.out_proj(context)
                if output.shape[-1] != hidden_states.shape[-1]:
                    raise RuntimeError(
                        f"SpectraV2Attention: output dimension mismatch. "
                        f"Got {output.shape[-1]}, expected {hidden_states.shape[-1]}."
                    )
                return output, None

            except Exception as e:
                logger.debug("SDPA path failed or fell back: %s", e)
                pass

        # Fallback path (manual attention)
        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        if _is_causal:
            causal_mask = torch.triu(torch.full((N, N_k), -1e4, device=scores.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if eff_mask is not None:
            scores = apply_attention_mask(scores, eff_mask)

        attn_weights = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                attn_weights = attn_weights * head_mask.view(1, self.num_heads, 1, 1).to(attn_weights.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                attn_weights = attn_weights * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(attn_weights.dtype)
            elif head_mask.dim() == 4:
                attn_weights = attn_weights * head_mask.to(attn_weights.dtype)

        context = torch.matmul(attn_weights, V).permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            if context.shape[-1] != hidden_states.shape[-1]:
                raise RuntimeError(
                    f"SpectraV2Attention: output dimension mismatch. "
                    f"Got {context.shape[-1]}, expected {hidden_states.shape[-1]}."
                )
            return context, (attn_weights if output_attentions else None)

        output = self.out_proj(context)
        if output.shape[-1] != hidden_states.shape[-1]:
            raise RuntimeError(
                f"SpectraV2Attention: output dimension mismatch. "
                f"Got {output.shape[-1]}, expected {hidden_states.shape[-1]}."
            )
        return output, (attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("mha_sdpa_inject")
@ATTENTION_REGISTRY.register("spectra_v2_inject")
@ATTENTION_REGISTRY.register("inject_spectra_v2")
def inject_spectra_v2(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    target_layer_idx: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Any]:
    """Safely replace attention in model or layer with SpectraV2Attention (mha_sdpa)."""
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None)

    if target is not None and (
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
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
            layer_list = target.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
            layer_list = target.layers

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                if target_layer_idx is not None and idx != target_layer_idx:
                    continue
                res = inject_spectra_v2(
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
        else:
            return None

    src_attn = old_attn
    if src_attn is None and target is not None:
        src_attn = getattr(target, "attention", getattr(target, "self", getattr(target, "attn", None)))

    hidden_size = (
        kwargs.get("hidden_size")
        or getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or (src_attn.q_proj.in_features if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features") else None)
        or (src_attn.query.in_features if hasattr(src_attn, "query") and hasattr(src_attn.query, "in_features") else None)
        or (src_attn.q_lin.in_features if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features") else None)
        or (target.in_features if hasattr(target, "in_features") else None)
        or _cfg_get(cfg, "hidden_size", None)
        or _cfg_get(cfg, "d_model", 64)
    )
    num_heads = (
        kwargs.get("num_heads")
        or kwargs.get("num_attention_heads")
        or getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or _cfg_get(cfg, "num_attention_heads", None)
        or _cfg_get(cfg, "num_heads", 4)
    )
    try:
        new_attn = SpectraV2Attention(
            config=cfg,
            pretrained_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            comp_config=comp_config,
            layer_idx=layer_idx,
            total_layers=total_layers,
            **kwargs,
        )
        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_attn
                else:
                    target.attention = new_attn
            elif hasattr(target, "self"):
                target.self = new_attn
            elif hasattr(target, "attn"):
                target.attn = new_attn
            else:
                target.attention = new_attn
        return new_attn
    except Exception:
        return None
