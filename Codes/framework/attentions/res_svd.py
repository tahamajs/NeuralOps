"""Spectral Residual Low-Rank SVD Attention (Res-SVD Attention) – Production-Hardened.

Combines base truncated SVD low-rank factorizations with trainable residual adapters:
W = U_base V_base^T + alpha * B A
where B A is initialized directly from the top-k residual singular directions of W - W_base.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA residual adapters with frozen base SVD factors
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

from framework.compression_engine.res_svd import ResSVDLinear
from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
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


@ATTENTION_REGISTRY.register("res_svd")
@ATTENTION_REGISTRY.register("ressvd")
@ATTENTION_REGISTRY.register("res_svd_attention")
@ATTENTION_REGISTRY.register("spectral_residual_svd")
class ResSVDAttention(BaseAttention):
    is_custom_attention: bool = True
    def __init__(
        self,
        config: Any,
        base_rank: Optional[int] = None,
        residual_rank: Optional[int] = None,
        rank: Optional[int] = 16,
        scaling: float = 1.0,
        factorize_base: bool = True,
        freeze_base: bool = True,
        dropout: Optional[float] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        effective_rank = rank if rank is not None else 16
        super().__init__(config, effective_rank)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        self.hidden_size: int = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.dim = self.hidden_size
        self.num_heads: int = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.num_kv_heads: int = getattr(
            config,
            "num_key_value_heads",
            kwargs.get("num_key_value_heads", self.num_heads),
        )

        cfg_head_dim = getattr(config, "head_dim", None)
        adapter_head_dim = getattr(getattr(config, "adapter", None), "head_dim", None)
        kw_head_dim = kwargs.get("head_dim", None)
        inferred = cfg_head_dim or adapter_head_dim or kw_head_dim
        if inferred is None and pretrained_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                if hasattr(pretrained_attn, attr):
                    layer_q = getattr(pretrained_attn, attr)
                    if hasattr(layer_q, "weight"):
                        inferred = layer_q.weight.shape[0] // max(1, self.num_heads)
                        break

        self.head_dim = int(inferred or (self.dim // max(1, self.num_heads)))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim

        self.base_rank: int = base_rank if base_rank is not None else getattr(comp_config, "base_rank", effective_rank)
        self.residual_rank: int = residual_rank if residual_rank is not None else getattr(comp_config, "residual_rank", max(4, effective_rank // 4))
        self.scaling = float(scaling)
        self.factorize_base = factorize_base
        self.freeze_base = freeze_base

        _mt = str(getattr(config, "model_type", "") or "").lower()
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        self.is_decoder = (
            bool(getattr(config, "is_decoder", False))
            or bool(getattr(config, "is_causal", False))
            or any(_mt.startswith(p) for p in _decoder_prefixes)
            or (pretrained_attn is not None and hasattr(pretrained_attn, "c_attn"))
            or bool(kwargs.get("is_causal", False))
            or bool(kwargs.get("is_decoder", False))
        )
        self.is_causal = bool(self.is_decoder)
        self.is_nested_encoder = False
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        self.query = ResSVDLinear(
            in_features=self.hidden_size,
            out_features=self.all_head_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )
        self.key = ResSVDLinear(
            in_features=self.hidden_size,
            out_features=self.all_kv_head_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )
        self.value = ResSVDLinear(
            in_features=self.hidden_size,
            out_features=self.all_kv_head_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )
        self.out_proj = ResSVDLinear(
            in_features=self.all_head_size,
            out_features=self.hidden_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )

        dropout_p = (
            dropout
            if dropout is not None
            else getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
        )
        self.dropout = nn.Dropout(dropout_p)

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn", None)
        if source_attn is not None:
            self.copy_weights(source_attn)

        logger.info(
            "ResSVDAttention initialized: hidden=%d, heads=%d, head_dim=%d, all_head_size=%d, is_decoder=%s, is_nested=%s, base_rank=%d, residual_rank=%d",
            self.hidden_size,
            self.num_heads,
            self.head_dim,
            self.all_head_size,
            self.is_decoder,
            self.is_nested_encoder,
            self.base_rank,
            self.residual_rank,
        )

    @property
    def q_proj(self) -> ResSVDLinear:
        return self.query

    @property
    def k_proj(self) -> ResSVDLinear:
        return self.key

    @property
    def v_proj(self) -> ResSVDLinear:
        return self.value

    @property
    def q_lin(self) -> ResSVDLinear:
        return self.query

    @property
    def k_lin(self) -> ResSVDLinear:
        return self.key

    @property
    def v_lin(self) -> ResSVDLinear:
        return self.value

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
                p_c_cls = getattr(c_attn_mod, "__class__", type(c_attn_mod)).__name__
                p_base_cls = getattr(base_c, "__class__", type(base_c)).__name__
                if ("Conv1D" in p_c_cls) or ("Conv1D" in p_base_cls) or hasattr(c_attn_mod, "nf") or hasattr(base_c, "nf") or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
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
                    p_proj_cls = getattr(c_proj_mod, "__class__", type(c_proj_mod)).__name__
                    p_base_o_cls = getattr(base_o, "__class__", type(base_o)).__name__
                    if ("Conv1D" in p_proj_cls) or ("Conv1D" in p_base_o_cls) or hasattr(c_proj_mod, "nf") or hasattr(base_o, "nf") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
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
            logger.warning(
                f"ResSVDAttention copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}. Proceeding with initialized projections."
            )

        target_map = {
            "q": self.query,
            "k": self.key,
            "v": self.value,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target in target_map.items():
                if proj_key == "o" and self.is_nested_encoder:
                    temp_linear = nn.Linear(target.in_features, target.out_features, bias=True)
                    nn.init.eye_(temp_linear.weight)
                    nn.init.zeros_(temp_linear.bias)
                    new_layer = ResSVDLinear.from_linear(
                        temp_linear,
                        base_rank=self.base_rank,
                        residual_rank=self.residual_rank,
                        scaling=self.scaling,
                        factorize_base=self.factorize_base,
                        freeze_base=self.freeze_base,
                    )
                    target.load_state_dict(new_layer.state_dict())
                    continue

                w, b = extracted[proj_key]
                if w is None:
                    logger.warning(f"Missing required weights for {proj_key}. Skipping copy.")
                    continue

                if w.shape != (target.out_features, target.in_features):
                    if w.t().contiguous().shape == (target.out_features, target.in_features):
                        w = w.t().contiguous()
                    elif w.shape[0] < target.out_features and target.out_features % w.shape[0] == 0:
                        repeats = target.out_features // w.shape[0]
                        w = w.repeat(repeats, 1)
                    else:
                        full_w = torch.zeros((target.out_features, target.in_features), dtype=w.dtype, device=w.device)
                        min_0 = min(target.out_features, w.shape[0])
                        min_1 = min(target.in_features, w.shape[1])
                        full_w[:min_0, :min_1].copy_(w[:min_0, :min_1])
                        w = full_w

                temp_linear = nn.Linear(target.in_features, target.out_features, bias=(b is not None))
                temp_linear.weight.copy_(w)
                if b is not None and temp_linear.bias is not None:
                    if b.shape == temp_linear.bias.shape:
                        temp_linear.bias.copy_(b)
                    elif b.shape[0] < temp_linear.bias.shape[0] and temp_linear.bias.shape[0] % b.shape[0] == 0:
                        repeats = temp_linear.bias.shape[0] // b.shape[0]
                        temp_linear.bias.copy_(b.repeat(repeats))
                    else:
                        min_b = min(temp_linear.bias.shape[0], b.shape[0])
                        temp_linear.bias.zero_()
                        temp_linear.bias[:min_b].copy_(b[:min_b])

                new_layer = ResSVDLinear.from_linear(
                    temp_linear,
                    base_rank=self.base_rank,
                    residual_rank=self.residual_rank,
                    scaling=self.scaling,
                    factorize_base=self.factorize_base,
                    freeze_base=self.freeze_base,
                )
                target.load_state_dict(new_layer.state_dict())

                # Check Frobenius norm
                sample_x = torch.eye(min(target.in_features, 64), target.in_features)
                y = target(sample_x)
                fnorm = torch.linalg.norm(y.float()).item()
                if fnorm < 1e-4:
                    logger.warning(f"Weight copy warning for {proj_key}: norm {fnorm:.6f} < 1e-4")

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
        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention with encoder_hidden_states is not supported in ResSVDAttention")

        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.query(hidden_states).view(B, N, H, D).permute(0, 2, 1, 3)
        K = self.key(hidden_states).view(B, N, self.num_kv_heads, D).permute(0, 2, 1, 3)
        V = self.value(hidden_states).view(B, N, self.num_kv_heads, D).permute(0, 2, 1, 3)

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

        scale = D ** -0.5
        # Memory-efficient: use SDPA to avoid OOM on large seqs; fallback to explicit if needed

        try:

            _qs = Q.half() if Q.dtype == torch.float32 else Q

            _ks = K.half() if K.dtype == torch.float32 else K

            _vs = V.half() if V.dtype == torch.float32 else V

            _is_causal = getattr(self, 'is_decoder', False) and not is_cross and N > 1 if 'is_cross' in dir() else getattr(self, 'is_decoder', False) and N > 1

            _sdpa_out = torch.nn.functional.scaled_dot_product_attention(_qs, _ks, _vs, is_causal=_is_causal, dropout_p=self.dropout.p if self.training else 0.0)

            _sdpa_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(B, N, self.all_head_size)

            _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)

            return (self.out_proj(_sdpa_out) if not getattr(self, 'is_nested_encoder', False) else _sdpa_out), None

        except Exception:

            pass

        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        # Causal masking
        is_causal = (self.is_decoder or getattr(self, "is_causal", False) or kwargs.get("is_causal", False)) and (encoder_hidden_states is None)
        if is_causal:
            causal_mask = torch.triu(torch.full((N, N), -1e4, device=scores.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Attention mask
        if attention_mask is not None:
            scores = apply_attention_mask(scores, attention_mask)

        attn_weights = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # head_mask handling
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                attn_weights = attn_weights * head_mask.view(1, self.num_heads, 1, 1).to(attn_weights.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                attn_weights = attn_weights * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(attn_weights.dtype)
            elif head_mask.dim() == 4:
                attn_weights = attn_weights * head_mask.to(attn_weights.dtype)

        context = torch.matmul(attn_weights, V).permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            return context, (attn_weights if output_attentions else None)

        output = self.out_proj(context)
        return output, (attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("res_svd_inject")
@ATTENTION_REGISTRY.register("inject_res_svd")
@ATTENTION_REGISTRY.register("inject_res_svd_attention")
def inject_res_svd(
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
            res = inject_res_svd(
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

    """Safely replace attention in model or layer with ResSVDAttention."""
    target_obj = layer if layer is not None else model
    try:
        if target_obj is not None and (
            hasattr(target_obj, "transformer")
            or hasattr(target_obj, "distilbert")
            or hasattr(target_obj, "roberta")
            or hasattr(target_obj, "bert")
        ):
            layers = None
            if hasattr(target_obj, "roberta") and hasattr(target_obj.roberta, "encoder"):
                layers = target_obj.roberta.encoder.layer
            elif hasattr(target_obj, "bert") and hasattr(target_obj.bert, "encoder"):
                layers = target_obj.bert.encoder.layer
            elif hasattr(target_obj, "distilbert") and hasattr(target_obj.distilbert, "transformer"):
                layers = target_obj.distilbert.transformer.layer
            elif hasattr(target_obj, "transformer") and hasattr(target_obj.transformer, "h"):
                layers = target_obj.transformer.h
            if layers is not None:
                for idx, lyr in enumerate(layers):
                    if target_layer_idx is not None and idx != target_layer_idx:
                        continue
                    inject_res_svd(
                        layer=lyr,
                        model_config=model_config or getattr(target_obj, "config", None),
                        layer_idx=idx,
                        total_layers=len(layers),
                        **kwargs,
                    )
                return target_obj

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

            new_attn = ResSVDAttention(
                config=config,
                pretrained_attn=target if target is not layer else None,
                **kwargs,
            )
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                    return layer
                else:
                    layer.attention = new_attn
                    return layer
            for attr in ("self_attn", "attn"):
                if hasattr(layer, attr):
                    setattr(layer, attr, new_attn)
                    return layer
            if hasattr(layer, "self"):
                layer.self = new_attn
                new_attn.is_nested_encoder = True
                return layer

            return new_attn

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

                new_attn = ResSVDAttention(
                    config=getattr(model, "config", None),
                    pretrained_attn=module,
                    **kwargs,
                )
                if child_name == "self" or "self" in parent_name:
                    new_attn.is_nested_encoder = True
                setattr(parent, child_name, new_attn)
                replaced += 1
                if target_layer_idx is not None and replaced > target_layer_idx:
                    break
        return model if replaced > 0 else None
    except Exception as e:
        logger.error("inject_res_svd failed at layer %d: %s", layer_idx, e)
        return None


inject_res_svd_attention = inject_res_svd
