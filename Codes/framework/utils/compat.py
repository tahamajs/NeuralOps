"""Family-aware, opt-in compatibility helpers for attention modules."""
from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn as nn

from .model_family import ModelAdapter, detect_family
from .qwen_compat import QwenIncompatibleError


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query K/V heads: [B, Hkv, L, D] -> [B, H, L, D]."""
    if n_rep <= 1:
        return x
    if x.dim() != 4:
        raise ValueError(f"repeat_kv expects [B,H,L,D], got {tuple(x.shape)}")
    return x[:, :, None, :, :].expand(x.size(0), x.size(1), n_rep, x.size(2), x.size(3)).reshape(
        x.size(0), x.size(1) * n_rep, x.size(2), x.size(3)
    )


def is_decoder_family(family: str) -> bool:
    return family in {"gpt", "gpt2", "qwen", "qwen3_5", "llama", "mistral", "falcon"}


def is_decoder_projection_name(name: str) -> bool:
    """Whether a parameter name denotes a decoder attention projection."""
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in {"weight", "bias"}:
        parts = name.rsplit(".", 2)
        leaf = parts[-2].lower() if len(parts) >= 2 else leaf
    return leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "c_attn", "c_proj"}


SENSITIVE_PATTERNS = [
    "attn", "attention", "delta", "delta_net", "gate", "expert",
    "router", "norm", "rope", "embed", "lm_head", "score", "classifier",
    "conv1d", "in_proj", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "lora"
]
ALLOWED_FFN_PATTERNS = ["mlp", "ffn", "dense", "feed_forward", "down_proj", "up_proj", "wi", "wo"]


def should_compress_qwen3_5(name: str) -> bool:
    n = name.lower()
    if any(p in n for p in SENSITIVE_PATTERNS):
        return False
    return any(p in n for p in ALLOWED_FFN_PATTERNS)


def should_compress_linear(model: Any, name: str, family: Optional[str] = None) -> bool:
    """
    Family-aware check for whether a linear layer should be compressed.
    
    Task 1: For qwen3_5, sensitive layers (attention, DeltaNet, GQA, MoE routers/experts,
    norms, RoPE, embeddings, heads) are protected, compressing only ordinary FFN dense layers.
    For other families, original safe linear layer compression behavior is preserved.
    """
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in {"weight", "bias"}:
        parts = name.rsplit(".", 2)
        leaf = parts[-2].lower() if len(parts) >= 2 else leaf

    name_low = name.lower()
    # Exclude head/classifier/output layers across all models
    if leaf in {"classifier", "score", "lm_head", "pooler", "pre_classifier"} or any(
        k in name_low for k in ["classifier", "score", "lm_head", "pooler", "pre_classifier", "classification_head"]
    ):
        return False

    fam = family or (detect_family(model) if not isinstance(model, str) else model)

    if fam == "qwen3_5":
        return should_compress_qwen3_5(name)

    if is_decoder_family(fam):
        return is_decoder_projection_name(name) or any(kw in name_low for kw in ["mlp", "ffn", "dense", "feed_forward"])

    return True


def get_dynamic_compression_rank(
    in_features: int,
    out_features: int,
    model_or_family: Any = None,
    default_rank: int = 16,
    comp_config: Optional[dict] = None,
) -> int:
    """
    Compute architecture-aware dynamic compression rank (Task 2).
    
    For Qwen3.5: rank = min(256, max(128, min(in_features, out_features) // 4))
    For other families: preserves default fixed/configured rank.
    """
    comp_config = comp_config or {}
    if "qwen3_5_rank" in comp_config and comp_config["qwen3_5_rank"] is not None:
        return int(comp_config["qwen3_5_rank"])
    if "dynamic_rank" in comp_config and comp_config["dynamic_rank"] is not None:
        return int(comp_config["dynamic_rank"])

    fam = model_or_family if isinstance(model_or_family, str) else (detect_family(model_or_family) if model_or_family is not None else "unknown")
    if fam == "qwen3_5":
        return min(256, max(128, min(in_features, out_features) // 4))
    return default_rank


def get_recovery_params(model: nn.Module) -> list[nn.Parameter]:
    """
    Task 3: Recovery parameter selection.
    
    Returns LoRA adapter parameters, classification head parameters,
    and all parameters of injected attention/adapter/compressor modules to allow them
    to adapt to the backbone representation.
    """
    has_lora = any(any(tag in n.lower() for tag in (".lora_", ".lora_a", ".lora_b")) for n, _ in model.named_parameters())
    for name, param in model.named_parameters():
        n_low = name.lower()
        if any(tag in n_low for tag in (".lora_", ".lora_a", ".lora_b", ".adapter", ".prompt")):
            trainable_params.append(param)
        elif any(h in n_low for h in ("score.", "classifier.", "classification_head", "pre_classifier", "pooler")):
            trainable_params.append(param)
        elif any(tag in n_low for tag in (".inner.", "attention.", "self_attn.", ".attn.", "swa_mamba", "kernel_linear")):
            if has_lora and (any(n_low.endswith(sfx) for sfx in (".u", ".s", ".v", ".weight_u", ".weight_v")) or ".lin1." in n_low or ".lin2." in n_low):
                continue
            trainable_params.append(param)
        elif any(tag in n_low for tag in (".down.", ".up.", ".u_proj", ".v_proj", ".u_matrix", ".v_matrix",
                                          ".w_a", ".w_b", ".weight_u", ".weight_v", ".factor", ".factors",
                                          ".comp_", ".svd_", ".cur_", ".salr", ".direct_svd", ".sparse_proj",
                                          ".cores", ".centroids", ".weight_table", ".alpha", ".gate",
                                          ".proj_k", ".proj_v")):
            if has_lora and any(n_low.endswith(sfx) for sfx in (".u", ".s", ".v", ".weight_u", ".weight_v")):
                continue
            trainable_params.append(param)

    for name, param in model.named_parameters():
        param.requires_grad = any(param is p for p in trainable_params)

    return trainable_params


def maybe_apply_rotary(q: torch.Tensor, k: torch.Tensor, adapter: ModelAdapter,
                       rotary_emb: Any, position_ids: Optional[torch.Tensor]):
    """Apply a native rotary module when required; never silently drop RoPE."""
    if adapter.position_embedding != "rotary":
        return q, k
    if rotary_emb is None:
        raise QwenIncompatibleError("rotary_position_embedding_required")
    try:
        rotary = rotary_emb(q, position_ids)
    except TypeError:
        try:
            rotary = rotary_emb(q, position_ids=position_ids)
        except TypeError as exc:
            raise QwenIncompatibleError(f"rotary_forward_signature: {exc}") from exc
    if hasattr(rotary, "cos") and hasattr(rotary, "sin"):
        cos, sin = rotary.cos, rotary.sin
    elif isinstance(rotary, (tuple, list)) and len(rotary) >= 2:
        cos, sin = rotary[0], rotary[1]
    else:
        raise QwenIncompatibleError("rotary_forward_invalid_output")
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    # HF rotary outputs are commonly [B,L,D] or [B,1,L,D], while custom
    # modules conventionally use [B,H,L,D].
    if cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.shape[-2] != q.shape[-2]:
        cos, sin = cos[..., :q.shape[-2], :], sin[..., :q.shape[-2], :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def causal_mask_from_attention_mask(attention_mask: Optional[torch.Tensor], adapter: ModelAdapter):
    """Return the mask without discarding decoder causal visibility semantics."""
    if attention_mask is None:
        return None
    if adapter.causal and attention_mask.dim() == 4:
        return attention_mask
    if adapter.causal and attention_mask.dim() == 2:
        bsz, seqlen = attention_mask.shape
        causal = torch.ones((seqlen, seqlen), dtype=torch.bool, device=attention_mask.device).tril()
        return causal[None, None, :, :] & attention_mask.bool()[:, None, None, :]
    if not adapter.causal and attention_mask.dim() == 2:
        return attention_mask.bool()[:, None, None, :]
    if not adapter.causal:
        return attention_mask
    raise QwenIncompatibleError(f"causal_mask_format_{attention_mask.dim()}d_unsupported")


def install_projection_aliases(module: nn.Module) -> nn.Module:
    """Installs bidirectional attribute aliases across all canonical projection naming schemes."""
    from framework.core.adapters import CANONICAL_PROJECTION_ALIASES
    for child in module.modules():
        # Bridge nested encoder attention blocks (RoBERTa, BERT, DeBERTa) to expose leaf projections
        if hasattr(child, "self") and hasattr(child.self, "query"):
            for attr_name in ["query", "key", "value"]:
                val = getattr(child.self, attr_name, None)
                if val is not None:
                    try:
                        object.__setattr__(child, attr_name, val)
                    except (AttributeError, TypeError):
                        pass
        if hasattr(child, "output") and hasattr(child.output, "dense"):
            val = getattr(child.output, "dense", None)
            if val is not None:
                try:
                    object.__setattr__(child, "dense", val)
                    object.__setattr__(child, "out_proj", val)
                    object.__setattr__(child, "out_lin", val)
                    object.__setattr__(child, "o_proj", val)
                except (AttributeError, TypeError):
                    pass

        for group in CANONICAL_PROJECTION_ALIASES.values():
            present_attr = next((name for name in group if hasattr(child, name)), None)
            if present_attr is not None:
                mod = getattr(child, present_attr)
                for alias in group:
                    if not hasattr(child, alias):
                        try:
                            object.__setattr__(child, alias, mod)
                        except (AttributeError, TypeError):
                            pass
    return module


def convert_mask(mask: Optional[torch.Tensor], adapter: ModelAdapter,
                 expects_bidirectional: bool = False) -> Optional[torch.Tensor]:
    if mask is None or not expects_bidirectional:
        return mask
    if mask.dim() == 2:
        return mask[:, None, None, :].bool()
    if mask.dim() == 3:
        return mask[:, None, :, :].bool()
    if mask.dim() == 4:
        # A causal additive mask cannot truthfully become bidirectional.
        # Preserve visibility semantics and let the caller decide whether to
        # skip a module whose contract requires non-causal attention.
        return mask >= 0
    return mask


def fp32_linalg(op, *tensors: torch.Tensor, **kwargs: Any):
    dtypes = [x.dtype for x in tensors if isinstance(x, torch.Tensor) and x.is_floating_point()]
    dtype = dtypes[0] if dtypes else torch.float32
    result = op(*[(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x) for x in tensors], **kwargs)
    if isinstance(result, tuple):
        return tuple(x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in result)
    return result.to(dtype) if isinstance(result, torch.Tensor) and result.is_floating_point() else result


class FamilyModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module, family: str):
        super().__init__()
        self.module = install_projection_aliases(module)
        self.adapter = ModelAdapter(module, family)

    def forward(self, *args: Any, **kwargs: Any):
        mask = kwargs.get("attention_mask", kwargs.get("mask", kwargs.get("attn_mask")))
        # A replacement that receives rotary tensors must explicitly support
        # them.  A bare legacy call with no positional inputs remains usable
        # for backwards-compatible unit/injection paths; the runner's strict
        # family policy is responsible for classifying such modules.
        if (self.adapter.position_embedding == "rotary"
                and not getattr(self.module, "supports_rope", False)
                and ("position_embeddings" in kwargs or "position_ids" in kwargs
                     or getattr(self.module, "requires_rope", False))):
            raise QwenIncompatibleError("module_does_not_support_rope")
        if mask is not None:
            adapted = convert_mask(mask, self.adapter, bool(getattr(self.module, "expects_bidirectional", False)))
            if "attention_mask" in kwargs: kwargs["attention_mask"] = adapted
            elif "mask" in kwargs: kwargs["mask"] = adapted
            elif "attn_mask" in kwargs: kwargs["attn_mask"] = adapted
        return self.module(*args, **kwargs)


def wrap_for_family(module: nn.Module, model_family: Optional[str] = None) -> nn.Module:
    family = model_family or detect_family(module)
    install_projection_aliases(module)
    # Keep BERT/DistilBERT's original call path byte-for-byte compatible.
    return module if family in {"bert", "roberta", "unknown"} else FamilyModuleWrapper(module, family)


def module_family_support(name: str) -> dict[str, str]:
    return {
        "bert": "encoder_compatible",
        "roberta": "encoder_compatible",
        "gpt": "adapted",
        "qwen": "adapted",
        "qwen3_5": "adapted",
        "t5": "encoder_decoder_compatible",
        "ssm": "adapted",
        "unknown": "adapted",
    }
