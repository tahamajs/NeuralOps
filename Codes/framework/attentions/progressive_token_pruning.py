"""Progressive Token Pruning (PTP) – Production-Hardened.

Learnable Progressive Token Pruning module with guaranteed [CLS] retention,
multi-head attention capabilities, and full multi-architecture support.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
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
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("progressive_token_pruning")
@ATTENTION_REGISTRY.register("ptp")
class ProgressiveTokenPruning(BaseAttention):
    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        keep_ratio: float = 0.8,
        rank: Optional[int] = 64,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        if rank is None:
            rank = 64
        super().__init__(config, rank)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        self.d_model = getattr(config, "hidden_size", d_model or kwargs.get("hidden_size", 768))
        self.dim = self.d_model
        self.num_heads = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.num_kv_heads = getattr(
            config,
            "num_key_value_heads",
            kwargs.get("num_key_value_heads", self.num_heads),
        )
        self.head_dim = getattr(
            config,
            "head_dim",
            kwargs.get("head_dim", self.dim // self.num_heads),
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.keep_ratio = getattr(comp_config, "keep_ratio", kwargs.get("keep_ratio", keep_ratio))

        is_dec = getattr(config, "is_decoder", kwargs.get("is_decoder", False))
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))) if config else 16
        self.lora_alpha = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))) if config else 32.0
        self.freeze_base = (self.lora_rank > 0)
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # Projections
        self.q_lin = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_lin = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_lin = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_lin = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin
        self.query = self.q_lin
        self.key = self.k_lin
        self.value = self.v_lin

        self.score_network = nn.Sequential(
            nn.Linear(self.dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

        src = pretrained_attn if pretrained_attn is not None else (old_attn if old_attn is not None else kwargs.get("source_attn", None))
        if src is not None:
            self.copy_weights(src)

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
                if (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
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
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
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
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"ProgressiveTokenPruning copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_lin,
            "k": self.k_lin,
            "v": self.v_lin,
            "o": self.out_lin,
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

                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif w.shape[0] < target_mod.weight.shape[0] and target_mod.weight.shape[0] % w.shape[0] == 0:
                    repeats = target_mod.weight.shape[0] // w.shape[0]
                    target_mod.weight.copy_(w.repeat(repeats, 1))
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
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                fnorm = torch.linalg.norm(target_mod.weight.float()).item()
                if fnorm < 1e-4:
                    logger.warning("Attention module copy_weights warning. Proceeding with initialized weights.")

        if self.freeze_base:
            for mod in [self.q_lin, self.k_lin, self.v_lin, self.out_lin]:
                mod.freeze_base_weights()

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
        x = hidden_states
        B, L, D = x.shape

        valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)

        scores = torch.sigmoid(self.score_network(x)).squeeze(-1)  # (B, L)
        # Always guarantee preservation of [CLS] / initial token
        scores[:, 0] = 1e4
        scores = scores.masked_fill(~valid_mask, -1e4)
        scores[:, 0] = 1e4

        k_tokens = max(1, min(L, int(math.ceil(L * self.keep_ratio))))
        _, top_indices = torch.topk(scores, k_tokens, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        # Projections
        q = self.q_lin(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_all = self.k_lin(x).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v_all = self.v_lin(x).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply RoPE if provided
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k_all = apply_rotary_pos_emb(q, k_all, cos, sin)

        # Repeat KV for GQA
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k_all = repeat_kv(k_all, repeats)
            v_all = repeat_kv(v_all, repeats)

        # Prune keys and values according to top_indices: (B, H, k_tokens, d)
        top_idx_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim)
        k_pruned = torch.gather(k_all, 2, top_idx_exp)
        v_pruned = torch.gather(v_all, 2, top_idx_exp)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q.float(), k_pruned.float().transpose(-2, -1).contiguous()) * scale  # (B, H, L, k_tokens)

        # Causal mask: query i cannot attend to key whose original position > i
        is_causal = self.is_decoder or kwargs.get("is_causal", False)
        if is_causal:
            q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
            k_orig_pos = top_indices.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k_tokens)
            causal_mask = k_orig_pos > q_pos
            attn_scores = attn_scores.masked_fill(causal_mask, -1e4)

        # Valid mask for pruned keys
        valid_keys = torch.gather(valid_mask, 1, top_indices)  # (B, k_tokens)
        attn_scores = attn_scores.masked_fill(~valid_keys.unsqueeze(1).unsqueeze(2), -1e4)

        probs = F.softmax(attn_scores, dim=-1).to(x.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # head_mask handling
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                probs = probs * head_mask.view(1, self.num_heads, 1, 1).to(probs.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                probs = probs * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(probs.dtype)
            elif head_mask.dim() == 4:
                probs = probs * head_mask.to(probs.dtype)

        attn_out = torch.matmul(probs, v_pruned).permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            return attn_out, (probs if output_attentions else None)

        out = self.out_lin(attn_out)
        return out, (probs if output_attentions else None)


def inject_progressive_token_pruning(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    target_layer_idx: Optional[int] = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
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
            res = inject_progressive_token_pruning(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    """Safely replace attention in model or layer with ProgressiveTokenPruning."""
    try:
        if layer is not None:
            config = model_config or getattr(layer, "config", None)
            target = old_attn
            if target is None:
                for attr in ("attention", "self_attn", "attn", "self"):
                    if hasattr(layer, attr):
                        cand = getattr(layer, attr)
                        if isinstance(cand, nn.Module) and cand is not layer:
                            target = cand
                            break
            if target is None:
                target = layer

            new_attn = ProgressiveTokenPruning(
                config=config,
                pretrained_attn=target if target is not layer else None,
                **kwargs,
            )
            for attr in ("self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    setattr(layer, attr, new_attn)
                    return layer
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                else:
                    layer.attention = new_attn
                return layer
            layer.attention = new_attn
            return layer

        if model is None:
            return None

        replaced = 0
        for name, module in model.named_modules():
            is_attn = False
            if hasattr(module, "q_lin") or hasattr(module, "query") or hasattr(module, "q_proj") or hasattr(module, "c_attn"):
                is_attn = True

            if is_attn:
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                child_name = name.rsplit(".", 1)[-1] if "." in name else name
                parent = model.get_submodule(parent_name) if parent_name else model

                new_attn = ProgressiveTokenPruning(
                    config=getattr(model, "config", None),
                    pretrained_attn=module,
                    **kwargs,
                )
                setattr(parent, child_name, new_attn)
                replaced += 1
                if target_layer_idx is not None and replaced > target_layer_idx:
                    break
        return model if replaced > 0 else None
    except Exception:
        return None


inject_ptp = inject_progressive_token_pruning
inject_progressive_token_pruning_attention = inject_progressive_token_pruning
