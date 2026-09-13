"""Family-aware, opt-in compatibility helpers for attention modules."""
from __future__ import annotations

import logging
from typing import Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from .model_family import ModelAdapter, detect_family
from .qwen_compat import QwenIncompatibleError
from framework.core.model_utils import replace_module


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query K/V heads: [B, Hkv, L, D] -> [B, H, L, D]."""
    if n_rep <= 1:
        return x
    if x.dim() != 4:
        raise ValueError(f"repeat_kv expects [B,H,L,D], got {tuple(x.shape)}")
    return x[:, :, None, :, :].expand(x.size(0), x.size(1), n_rep, x.size(2), x.size(3)).reshape(
        x.size(0), x.size(1) * n_rep, x.size(2), x.size(3)
    )


def init_output_scale(module: nn.Module, name: str = "output_scale") -> None:
    """Register output_scale learnable parameter if not already present."""
    if not hasattr(module, name):
        module.register_parameter(name, nn.Parameter(torch.ones(1)))



def is_decoder_family(family: str) -> bool:
    return family in {"gpt", "gpt2", "qwen", "qwen3_5", "llama", "mistral", "falcon"}


def is_decoder_projection_name(name: str) -> bool:
    """Whether a parameter name denotes a decoder attention projection."""
    name_low = name.lower()
    if any(p in name_low for p in ["c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]):
        if "attn" in name_low or "attention" in name_low or "self_attn" in name_low:
            return True
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in {"weight", "bias", "base_layer"}:
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
    if leaf in {"weight", "bias", "base_layer"}:
        parts = name.rsplit(".", 2)
        leaf = parts[-2].lower() if len(parts) >= 2 else leaf

    name_low = name.lower()
    # Exclude head/classifier/output layers across all models
    if leaf in {"classifier", "score", "lm_head", "pooler", "pre_classifier"} or any(
        k in name_low for k in ["classifier", "score", "lm_head", "pooler", "pre_classifier", "classification_head"]
    ):
        return False

    if "c_attn" in name_low or "c_proj" in name_low:
        if "attn" in name_low or "attention" in name_low:
            return True

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
    if isinstance(comp_config, dict):
        if "qwen3_5_rank" in comp_config and comp_config["qwen3_5_rank"] is not None:
            return int(comp_config["qwen3_5_rank"])
        if "dynamic_rank" in comp_config and comp_config["dynamic_rank"] is not None:
            return int(comp_config["dynamic_rank"])
        if "rank" in comp_config and comp_config["rank"] is not None:
            return int(comp_config["rank"])
    elif comp_config is not None:
        if hasattr(comp_config, "qwen3_5_rank") and getattr(comp_config, "qwen3_5_rank") is not None:
            return int(getattr(comp_config, "qwen3_5_rank"))
        if hasattr(comp_config, "dynamic_rank") and getattr(comp_config, "dynamic_rank") is not None:
            return int(getattr(comp_config, "dynamic_rank"))
        if hasattr(comp_config, "rank") and getattr(comp_config, "rank") is not None:
            return int(getattr(comp_config, "rank"))

    if isinstance(model_or_family, str):
        fam = model_or_family
    elif model_or_family is not None:
        fam = detect_family(model_or_family)
    else:
        fam = "unknown"
    if fam == "qwen3_5":
        target = min(in_features, out_features) // 4
        return max(128, min(256, target))
    return default_rank


def get_recovery_params(model: nn.Module) -> list[nn.Parameter]:
    """
    Task 3: Recovery parameter selection.
    
    Returns LoRA adapter parameters, classification head parameters,
    and all parameters of injected attention/adapter/compressor modules.
    Ensures that fallback and uncompressed original linear/Conv1D layers remain strictly frozen.
    """
    svd_modules = [m for m in model.modules() if m.__class__.__name__ in ("SVDLinear", "SVDLinearLayer")]
    has_svd = len(svd_modules) > 0
    svd_param_ids = {id(p) for m in svd_modules for p in m.parameters()}
    has_lora = any(any(t in n.lower() for t in (".lora_", ".lora_a", ".lora_b")) for n, _ in model.named_parameters())
    fallback_layers = set()
    if hasattr(model, "compression_stats") and isinstance(model.compression_stats, dict):
        fb = model.compression_stats.get("fallback_layers", [])
        if isinstance(fb, (list, tuple, set)):
            fallback_layers = set(fb)

    # Map parameter names to containing module types for strict recurrent module checking
    module_by_name = dict(model.named_modules())
    recurrent_type_keywords = ("hgrn", "delta", "mamba", "ssm", "recurrent", "jamba")

    trainable_params = []
    for name, param in model.named_parameters():
        n_low = name.lower()
        parent_module_prefix = name.split(".base_layer.")[0] if ".base_layer." in name else name.rsplit(".", 1)[0]
        layer_has_lora = any(k.startswith(parent_module_prefix) and any(t in k.lower() for t in (".lora_", ".lora_a", ".lora_b")) for k, _ in model.named_parameters())

        # Check if parent or enclosing module is a recurrent/hybrid module
        is_recurrent_module = False
        parts = name.split(".")
        for i in range(len(parts)):
            sub_prefix = ".".join(parts[:i])
            if sub_prefix in module_by_name:
                mod_cls = module_by_name[sub_prefix].__class__.__name__.lower()
                if any(kw in mod_cls for kw in recurrent_type_keywords):
                    is_recurrent_module = True
                    break

        # Fallback layers must remain strictly frozen during recovery fine-tuning
        if fallback_layers and (parent_module_prefix in fallback_layers or any(name.startswith(fb + ".") or name == fb for fb in fallback_layers)):
            continue

        # For recurrent/hybrid modules (HGRN, Delta, Mamba, SSM, etc.), allow lora adapters, output scales, gates, and decays
        if is_recurrent_module:
            if any(tag in n_low for tag in (".lora_a", ".lora_b", "lora_a", "lora_b", ".output_scale", ".gate", ".alpha", ".beta", ".decay")):
                trainable_params.append(param)
            continue

        # 1. LoRA adapter, gate, calibration scale, decay, and span parameters are always trainable
        if any(tag in n_low for tag in (".lora_", ".lora_a", ".lora_b", ".adapter", ".prompt", ".gate", "gate.", ".alpha", ".beta", ".output_scale", ".decay", "span_logits", ".span_logits")):
            trainable_params.append(param)
            continue

        # 2. Classification / sequence classification heads are trainable, but when LoRA is used
        # on backbones, sequence classification heads (classifier., pre_classifier, score.)
        # must remain frozen to prevent catastrophic overfitting during short recovery fine-tuning.
        if any(h in n_low for h in ("score.", "classifier.", "classification_head", "pre_classifier", "pooler")):
            if (has_lora and any(d in n_low for d in ("classifier.", "pre_classifier.", "score."))) or not param.requires_grad:
                continue
            trainable_params.append(param)
            continue

        # Explicitly exclude standard base linear parameters unless inside a LoRALinear adapter
        if any(n_low.endswith(sfx) for sfx in (
            ".weight", ".bias", ".u", ".v", ".a", ".b", ".beta", ".alpha", ".gate",
            ".in_proj", ".out_proj", ".x_proj", ".q_proj", ".k_proj", ".v_proj", ".z_proj"
        )):
            if not any(tag in n_low for tag in (".lora_", ".lora_a", ".lora_b")):
                if layer_has_lora or has_lora:
                    continue

        # If this layer has LoRA adapters, its base weight/bias and base factors remain frozen
        if layer_has_lora and (any(n_low.endswith(sfx) for sfx in (".u", ".s", ".v", ".weight_u", ".weight_v", ".bias")) or ".base_layer." in n_low or ".lin1." in n_low or ".lin2." in n_low):
            continue

        # 3. If model contains factorized SVDLinear modules:
        # Only compressed SVDLinear parameters (U, V, bias) are trainable;
        # Fallback original layers (Conv1D.weight, Linear.weight, etc.) remain strictly frozen.
        if has_svd:
            if id(param) in svd_param_ids:
                trainable_params.append(param)
            continue

        # 4. If model is configured in LoRA mode without SVDLinear:
        # Only LoRA parameters and heads are trainable; backbone/factor weights remain frozen.
        # EXCEPTION: injected attention module parameters that are NOT part of the LoRA base
        # layers must also be trainable (e.g. preservation gates, Linformer E/F projections,
        # GoldFinch delta memory, etc.).
        if has_lora:
            # Allow preservation gate (output_gate) — always trainable when present
            if n_low.endswith(".output_gate"):
                trainable_params.append(param)
                continue
            # Allow Linformer E / F projection matrices (named .inner.e and .inner.f)
            if ".inner.e" in n_low or ".inner.f" in n_low:
                trainable_params.append(param)
                continue
            # Allow any custom injected parameter inside an .inner. or ._wrapped_attention. attention submodule
            # but NOT the base weight/bias of a linear layer (those stay frozen)
            if (".inner." in n_low or "._wrapped_attention." in n_low) and not (n_low.endswith(".weight") or n_low.endswith(".bias")):
                trainable_params.append(param)
                continue
            # Allow GoldFinch delta-memory / gate parameters
            if any(tag in n_low for tag in (".delta_mem.", ".beta_gate.", ".forget_gate.",
                                             ".input_gate.", ".output_proj.", ".mem_proj.")):
                trainable_params.append(param)
                continue
            continue

        # 5. Non-LoRA, non-SVD models (custom attention replacements, SSM, CUR, etc.):
        if any(tag in n_low for tag in (".inner.", "._wrapped_attention.", "swa_mamba", "kernel_linear", ".u", ".v")):
            trainable_params.append(param)
        elif any(tag in n_low for tag in (".down.", ".up.", ".u_proj", ".v_proj", ".u_matrix", ".v_matrix",
                                          ".w_a", ".w_b", ".weight_u", ".weight_v", ".factor", ".factors",
                                          ".comp_", ".svd_", ".cur_", ".salr", ".direct_svd", ".sparse_proj",
                                          ".cores", ".centroids", ".weight_table", ".alpha",
                                          ".proj_k", ".proj_v")):
            trainable_params.append(param)
        elif not (n_low.endswith(".weight") or n_low.endswith(".bias")) and any(tag in n_low for tag in ("attention.", "self_attn.", ".attn.")):
            trainable_params.append(param)
        # In hybrid recurrent attention modules (HGRN, DeltaNet, etc.), only keep gates and adapters trainable, not full base projections
        elif any(tag in n_low for tag in (".gate.", ".beta.", ".input_gate.", ".forget_gate.", ".alpha_gate.")):
            trainable_params.append(param)

    # Set requires_grad
    for name, param in model.named_parameters():
        param.requires_grad = any(param is p for p in trainable_params)

    # Safety check on total trainable parameters to prevent OOM
    total_trainable = sum(p.numel() for p in trainable_params)
    logger.info("Recovery fine-tuning selected %d trainable parameters across %d tensors.", total_trainable, len(trainable_params))
    if total_trainable > 2_000_000:
        logger.warning(
            "Total trainable parameters (%d) exceeds 2,000,000 threshold for recovery. Pruning heavy base weights to prevent OOM.",
            total_trainable,
        )
        pruned_params = []
        for p in trainable_params:
            if p.ndim <= 1:
                pruned_params.append(p)
            elif p.numel() <= 32768:
                pruned_params.append(p)
            else:
                p.requires_grad = False
        trainable_params = pruned_params
        for name, param in model.named_parameters():
            param.requires_grad = any(param is p for p in trainable_params)
        total_trainable = sum(p.numel() for p in trainable_params)
        logger.info("Reduced recovery parameters to %d trainable parameters across %d tensors.", total_trainable, len(trainable_params))

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
    if name.lower() in {"mamba"}:
        return {
            "bert": "encoder_compatible",
            "roberta": "encoder_compatible",
            "gpt": "adapted",
            "qwen": "incompatible",
            "qwen3_5": "incompatible",
            "t5": "encoder_decoder_compatible",
            "ssm": "adapted",
            "unknown": "adapted",
        }
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

