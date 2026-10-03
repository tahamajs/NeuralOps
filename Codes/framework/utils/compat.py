"""Family-aware, opt-in compatibility helpers for attention modules.

Bug fixes (2026-09-25):
- Fix CP-01: get_recovery_params now preserves SVDLinear U/V factor matrices
  during the parameter-budget pruning step, even if their numel exceeds the
  32768 threshold. Previously, recovery was reduced to bias-only training,
  crippling recovery on compressed models.
- Fix CP-02: Parameter-budget pruning now uses priority-based allocation:
  (1) protected params (LoRA/gates/classifier), (2) SVDLinear U/V factors,
  (3) small params (1D, or numel <= 32768), (4) other large 2D params.
- Fix CP-03: The 5M-rank-clamp now correctly preserves the LoRA's alpha/r
  scaling by adjusting both lora_rank and lora_alpha together.
- Fix CP-04: Better diagnostics: log how many params were kept per priority
  bucket so recovery behavior is observable.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from .model_family import CAUSAL_FAMILIES, ModelAdapter, detect_family
from .qwen_compat import QwenIncompatibleError


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query K/V heads: [B, Hkv, L, D] -> [B, H, L, D]."""
    if n_rep <= 1:
        return x
    if x.dim() != 4:
        raise ValueError(f"repeat_kv expects [B,H,L,D], got {tuple(x.shape)}")
    return (
        x[:, :, None, :, :]
        .expand(x.size(0), x.size(1), n_rep, x.size(2), x.size(3))
        .reshape(x.size(0), x.size(1) * n_rep, x.size(2), x.size(3))
    )


def init_output_scale(module: nn.Module, name: str = "output_scale") -> None:
    """Register output_scale learnable parameter if not already present."""
    if not hasattr(module, name):
        module.register_parameter(name, nn.Parameter(torch.ones(1)))


def is_decoder_family(family: str) -> bool:
    """Check whether a family (canonical or raw) is a causal decoder.

    Fix CP-00: Now uses the canonical CAUSAL_FAMILIES set from model_family,
    so all GPT variants (gpt2, llama, mistral, ...) are correctly classified.
    """
    return family in CAUSAL_FAMILIES


def is_decoder_projection_name(name: str) -> bool:
    """Whether a parameter name denotes a decoder attention projection."""
    name_low = name.lower()
    if any(
        p in name_low
        for p in [
            "c_attn",
            "c_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
        ]
    ):
        if "attn" in name_low or "attention" in name_low or "self_attn" in name_low:
            return True
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in {"weight", "bias", "base_layer"}:
        parts = name.rsplit(".", 2)
        leaf = parts[-2].lower() if len(parts) >= 2 else leaf
    return leaf in {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "c_attn",
        "c_proj",
    }


SENSITIVE_PATTERNS = [
    "attn",
    "attention",
    "delta",
    "delta_net",
    "gate",
    "expert",
    "router",
    "norm",
    "rope",
    "embed",
    "lm_head",
    "score",
    "classifier",
    "conv1d",
    "in_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "lora",
]
ALLOWED_FFN_PATTERNS = [
    "mlp",
    "ffn",
    "dense",
    "feed_forward",
    "down_proj",
    "up_proj",
    "wi",
    "wo",
]


def should_compress_qwen3_5(name: str) -> bool:
    n = name.lower()
    if any(p in n for p in SENSITIVE_PATTERNS):
        return False
    return any(p in n for p in ALLOWED_FFN_PATTERNS)


def should_compress_linear(model: Any, name: str, family: str | None = None) -> bool:
    """
    Family-aware check for whether a linear layer should be compressed.

    For qwen3_5: sensitive layers (attention, DeltaNet, GQA, MoE routers/experts,
    norms, RoPE, embeddings, heads) are protected; only ordinary FFN dense layers
    are compressed. For other families, safe linear layer compression is preserved.
    """
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in {"weight", "bias", "base_layer"}:
        parts = name.rsplit(".", 2)
        leaf = parts[-2].lower() if len(parts) >= 2 else leaf

    name_low = name.lower()
    if leaf in {"classifier", "score", "lm_head", "pooler", "pre_classifier"} or any(
        k in name_low
        for k in [
            "classifier",
            "score",
            "lm_head",
            "pooler",
            "pre_classifier",
            "classification_head",
        ]
    ):
        return False

    if "c_attn" in name_low or "c_proj" in name_low:
        if "attn" in name_low or "attention" in name_low:
            return True

    fam = family or (detect_family(model) if not isinstance(model, str) else model)

    if fam == "qwen3_5":
        return should_compress_qwen3_5(name)

    recognized_attn = [
        "q_lin",
        "k_lin",
        "v_lin",
        "out_lin",  # DistilBERT
        "query",
        "key",
        "value",
        "dense",
        "output.dense",  # RoBERTa / BERT
        "c_attn",
        "c_proj",  # GPT-2
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",  # Qwen / LLaMA
    ]

    if any(leaf == k or k in name_low.split(".") for k in recognized_attn):
        return True

    if is_decoder_family(fam):
        return is_decoder_projection_name(name) or any(
            kw in name_low for kw in ["mlp", "ffn", "dense", "feed_forward"]
        )

    return True


def get_dynamic_compression_rank(
    in_features: int,
    out_features: int,
    model_or_family: Any = None,
    default_rank: int = 16,
    comp_config: dict | None = None,
) -> int:
    """
    Compute architecture-aware dynamic compression rank.

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
        if getattr(comp_config, "qwen3_5_rank", None) is not None:
            return int(comp_config.qwen3_5_rank)
        if getattr(comp_config, "dynamic_rank", None) is not None:
            return int(comp_config.dynamic_rank)
        if getattr(comp_config, "rank", None) is not None:
            return int(comp_config.rank)

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


# ---------------------------------------------------------------------------
# Priority buckets for recovery pruning (Fix CP-01/CP-02)
# ---------------------------------------------------------------------------
PROTECTED_KEYWORDS = (
    "lora_a",
    "lora_b",
    "lora_",
    "classifier",
    "score",
    "gate",
    "scale",
    "alpha",
    "beta",
)
SVD_FACTOR_KEYWORDS = (
    ".u",
    ".v",
    ".weight_u",
    ".weight_v",
    ".svd_u",
    ".svd_v",
    ".factor_u",
    ".factor_v",
    ".q_down",
    ".q_up",
    ".down.",
    ".up.",
)


def _is_svd_factor_param(param_id: int, svd_param_ids: set) -> bool:
    return param_id in svd_param_ids


def _is_protected_param(name: str) -> bool:
    n_l = name.lower()
    return any(k in n_l for k in PROTECTED_KEYWORDS)


def get_recovery_params(model: nn.Module) -> list[nn.Parameter]:
    """
    Select recovery parameters for LoRA / SVD / injected attention modules.

    Fix CP-01: SVDLinear U/V factor matrices are now preserved during the
    parameter budget pruning step. Previously, their numel (e.g. 64 * 768 =
    49152) exceeded the 32768 threshold and they were pruned, reducing recovery
    to bias-only training.

    Fix CP-02: Pruning now uses priority-based allocation:
        Priority 0 (highest): protected params (LoRA / gates / classifier / alpha / beta)
        Priority 1: SVDLinear U/V factors and factorized parameters
        Priority 2: small params (1D, or numel <= 32768)
        Priority 3 (lowest): other large 2D params
    """
    # Collect SVD module parameter ids and LoRA presence.
    svd_modules = [
        m
        for m in model.modules()
        if m.__class__.__name__ in ("SVDLinear", "SVDLinearLayer", "FactorisedLinear")
    ]
    svd_param_ids = {id(p) for m in svd_modules for p in m.parameters()}
    has_svd = len(svd_modules) > 0

    LORA_KEYWORDS = ("lora_a", "lora_b", "lora_")
    has_lora = any(
        any(t in n.lower() for t in LORA_KEYWORDS) for n, _ in model.named_parameters()
    )

    fallback_layers = set()
    if hasattr(model, "compression_stats") and isinstance(
        model.compression_stats, dict
    ):
        fb = model.compression_stats.get("fallback_layers", [])
        if isinstance(fb, (list, tuple, set)):
            fallback_layers = set(fb)

    module_by_name = dict(model.named_modules())
    recurrent_type_keywords = ("hgrn", "delta", "mamba", "ssm", "recurrent", "jamba")

    trainable_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        n_low = name.lower()
        parent_module_prefix = (
            name.split(".base_layer.")[0]
            if ".base_layer." in name
            else name.rsplit(".", 1)[0]
        )
        layer_has_lora = any(
            k.startswith(parent_module_prefix)
            and any(t in k.lower() for t in LORA_KEYWORDS)
            for k, _ in model.named_parameters()
        )

        # Recurrent/hybrid module detection.
        is_recurrent_module = False
        parts = name.split(".")
        for i in range(len(parts)):
            sub_prefix = ".".join(parts[:i])
            if sub_prefix in module_by_name:
                mod_cls = module_by_name[sub_prefix].__class__.__name__.lower()
                if any(kw in mod_cls for kw in recurrent_type_keywords):
                    is_recurrent_module = True
                    break

        # Fallback layers remain strictly frozen.
        if fallback_layers and (
            parent_module_prefix in fallback_layers
            or any(name.startswith(fb + ".") or name == fb for fb in fallback_layers)
        ):
            continue

        # For recurrent/hybrid modules, only LoRA params are trainable.
        if is_recurrent_module:
            if any(tag in n_low for tag in ("lora_a", "lora_b")):
                trainable_params.append(param)
            continue

        # Base layer inside a LoRA wrapper stays frozen unless it is a LoRA param.
        if ".base_layer." in n_low and not any(
            tag in n_low for tag in ("lora_a", "lora_b", "lora_")
        ):
            continue

        # Always trainable: LoRA / gates / scales / decay / span / alpha / beta.
        if any(
            tag in n_low
            for tag in (
                "lora_a",
                "lora_b",
                "lora_",
                ".adapter",
                ".prompt",
                ".gate",
                "gate.",
                ".alpha",
                ".beta",
                ".output_scale",
                ".decay",
                "span_logits",
                ".span_logits",
            )
        ):
            trainable_params.append(param)
            continue

        # Classification head handling.
        if any(
            h in n_low
            for h in (
                "score.",
                "classifier.",
                "classification_head",
                "pre_classifier",
                "pooler",
            )
        ):
            if (
                has_lora
                and any(
                    d in n_low for d in ("classifier.", "pre_classifier.", "score.")
                )
            ) or not param.requires_grad:
                continue
            trainable_params.append(param)
            continue

        # Standard base weight/bias handling.
        if any(
            n_low.endswith(sfx)
            for sfx in (
                ".weight",
                ".bias",
                ".u",
                ".v",
                ".a",
                ".b",
                ".beta",
                ".alpha",
                ".gate",
                ".in_proj",
                ".out_proj",
                ".x_proj",
                ".q_proj",
                ".k_proj",
                ".v_proj",
                ".z_proj",
            )
        ):
            if not any(tag in n_low for tag in LORA_KEYWORDS):
                if layer_has_lora or has_lora:
                    continue

        # If this layer has LoRA, its base weight/bias and base factors stay frozen.
        if layer_has_lora and (
            any(
                n_low.endswith(sfx)
                for sfx in (
                    ".u",
                    ".s",
                    ".v",
                    ".weight_u",
                    ".weight_v",
                    ".bias",
                )
            )
            or ".base_layer." in n_low
            or ".lin1." in n_low
            or ".lin2." in n_low
        ):
            continue

        # Factorized SVDLinear modules: only their U/V params are trainable.
        if has_svd:
            if id(param) in svd_param_ids:
                trainable_params.append(param)
            continue

        # LoRA mode without SVDLinear: only LoRA params and injected attention extras.
        if has_lora:
            if n_low.endswith(".output_gate"):
                trainable_params.append(param)
                continue
            if ".inner.e" in n_low or ".inner.f" in n_low:
                trainable_params.append(param)
                continue
            if (".inner." in n_low or "._wrapped_attention." in n_low) and not (
                n_low.endswith(".weight") or n_low.endswith(".bias")
            ):
                trainable_params.append(param)
                continue
            if any(
                tag in n_low
                for tag in (
                    ".delta_mem.",
                    ".beta_gate.",
                    ".forget_gate.",
                    ".input_gate.",
                    ".output_proj.",
                    ".mem_proj.",
                )
            ):
                trainable_params.append(param)
                continue
            continue

        # Non-LoRA / non-SVD path.
        if (
            (
                any(
                    tag in n_low
                    for tag in (
                        ".inner.",
                        "._wrapped_attention.",
                        "swa_mamba",
                        "kernel_linear",
                    )
                )
                or any(
                    f".{x}." in n_low or n_low.endswith(f".{x}")
                    for x in (
                        "u",
                        "v",
                        "weight_u",
                        "weight_v",
                    )
                )
            )
            or any(
                tag in n_low
                for tag in (
                    ".down.",
                    ".up.",
                    ".u_proj",
                    ".v_proj",
                    ".u_matrix",
                    ".v_matrix",
                    ".w_a",
                    ".w_b",
                    ".weight_u",
                    ".weight_v",
                    ".factor",
                    ".factors",
                    ".comp_",
                    ".svd_",
                    ".cur_",
                    ".salr",
                    ".direct_svd",
                    ".sparse_proj",
                    ".cores",
                    ".centroids",
                    ".weight_table",
                    ".alpha",
                    ".proj_k",
                    ".proj_v",
                )
            )
            or not (n_low.endswith(".weight") or n_low.endswith(".bias"))
            and any(tag in n_low for tag in ("attention.", "self_attn.", ".attn."))
            or any(
                tag in n_low
                for tag in (
                    ".gate.",
                    ".beta.",
                    ".input_gate.",
                    ".forget_gate.",
                    ".alpha_gate.",
                )
            )
        ):
            trainable_params.append(param)

    # Deduplicate preserving order.
    seen_ids = set()
    unique_trainable = []
    for p in trainable_params:
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            unique_trainable.append(p)
    trainable_params = unique_trainable

    # Set requires_grad across the model.
    trainable_id_set = {id(p) for p in trainable_params}
    for _, param in model.named_parameters():
        param.requires_grad = id(param) in trainable_id_set

    total_trainable = sum(p.numel() for p in trainable_params)
    logger.info(
        "Recovery fine-tuning selected %d trainable parameters across %d tensors.",
        total_trainable,
        len(trainable_params),
    )

    is_nested_or_deep = any(
        "encoder" in n.lower() or "roberta" in n.lower() or "bert" in n.lower()
        for n, _ in model.named_parameters()
    )
    prune_threshold = 10_000_000 if (has_lora or is_nested_or_deep) else 2_000_000

    # Fix CP-03: Shrink LoRA rank properly, preserving the r/alpha scaling.
    if total_trainable > 5_000_000:
        for m in model.modules():
            if hasattr(m, "lora_rank") and getattr(m, "lora_rank", 0) > 8:
                old_r = m.lora_rank
                m.lora_rank = 8
                if hasattr(m, "lora_alpha"):
                    # Preserve effective scaling: new_alpha / 8 == old_alpha / old_r
                    m.lora_alpha = m.lora_alpha * (8.0 / max(1, old_r))
                if hasattr(m, "lora_scaling") and hasattr(m, "lora_alpha"):
                    m.lora_scaling = m.lora_alpha / 8

    # Fix CP-01/CP-02: Priority-based pruning.
    if total_trainable > prune_threshold:
        logger.warning(
            "Total trainable parameters (%d) exceeds %d threshold for recovery. "
            "Pruning with priority-based allocation.",
            total_trainable,
            prune_threshold,
        )

        # Bucket every trainable param by priority.
        priority_buckets: dict[int, list[tuple[str, nn.Parameter]]] = {
            0: [],  # protected (LoRA/gates/classifier/alpha/beta)
            1: [],  # SVDLinear U/V factors
            2: [],  # small params (1D or numel <= 32768)
            3: [],  # other large 2D params
        }
        for name, param in model.named_parameters():
            if id(param) not in trainable_id_set:
                continue
            n_l = name.lower()
            if _is_protected_param(n_l):
                priority_buckets[0].append((name, param))
            elif id(param) in svd_param_ids or any(
                kw in n_l for kw in SVD_FACTOR_KEYWORDS
            ):
                priority_buckets[1].append((name, param))
            elif param.ndim <= 1 or param.numel() <= 32768:
                priority_buckets[2].append((name, param))
            else:
                priority_buckets[3].append((name, param))

        # Allocate budget in priority order.
        budget = prune_threshold
        kept_params: list[nn.Parameter] = []
        curr_count = 0
        for priority in sorted(priority_buckets.keys()):
            bucket = priority_buckets[priority]
            bucket_kept = 0
            bucket_pruned = 0
            for name, p in bucket:
                if curr_count + p.numel() <= budget:
                    kept_params.append(p)
                    curr_count += p.numel()
                    bucket_kept += 1
                else:
                    p.requires_grad = False
                    bucket_pruned += 1
            logger.info(
                "  priority=%d: kept=%d pruned=%d (running total=%d)",
                priority,
                bucket_kept,
                bucket_pruned,
                curr_count,
            )

        trainable_params = kept_params
        kept_id_set = {id(p) for p in kept_params}
        for _, param in model.named_parameters():
            param.requires_grad = id(param) in kept_id_set
        total_trainable = sum(p.numel() for p in trainable_params)
        logger.info(
            "Reduced recovery parameters to %d trainable parameters across %d tensors.",
            total_trainable,
            len(trainable_params),
        )

    logger.info(
        "Discovered %d trainable tensors (%d params)",
        len(trainable_params),
        total_trainable,
    )
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug(
                "  %s: %s (%d params)", name, tuple(param.shape), param.numel()
            )

    return trainable_params


def maybe_apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    adapter: ModelAdapter,
    rotary_emb: Any,
    position_ids: torch.Tensor | None,
):
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
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]),
            dim=-1,
        )

    if cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.shape[-2] != q.shape[-2]:
        cos, sin = cos[..., : q.shape[-2], :], sin[..., : q.shape[-2], :]
    return (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )


def causal_mask_from_attention_mask(
    attention_mask: torch.Tensor | None,
    adapter: ModelAdapter,
):
    """Return the mask without discarding decoder causal visibility semantics."""
    if attention_mask is None:
        return None
    if adapter.causal and attention_mask.dim() == 4:
        return attention_mask
    if adapter.causal and attention_mask.dim() == 2:
        bsz, seqlen = attention_mask.shape
        causal = torch.ones(
            (seqlen, seqlen),
            dtype=torch.bool,
            device=attention_mask.device,
        ).tril()
        return causal[None, None, :, :] & attention_mask.bool()[:, None, None, :]
    if not adapter.causal and attention_mask.dim() == 2:
        return attention_mask.bool()[:, None, None, :]
    if not adapter.causal:
        return attention_mask
    raise QwenIncompatibleError(
        f"causal_mask_format_{attention_mask.dim()}d_unsupported"
    )


def install_projection_aliases(module: nn.Module) -> nn.Module:
    """Install bidirectional attribute aliases across canonical projection names."""
    from framework.core.adapters import CANONICAL_PROJECTION_ALIASES

    for child in module.modules():
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


def convert_mask(
    mask: torch.Tensor | None,
    adapter: ModelAdapter,
    expects_bidirectional: bool = False,
) -> torch.Tensor | None:
    if mask is None or not expects_bidirectional:
        return mask
    if mask.dim() == 2:
        return mask[:, None, None, :].bool()
    if mask.dim() == 3:
        return mask[:, None, :, :].bool()
    if mask.dim() == 4:
        return mask >= 0
    return mask


def fp32_linalg(op, *tensors: torch.Tensor, **kwargs: Any):
    dtypes = [
        x.dtype
        for x in tensors
        if isinstance(x, torch.Tensor) and x.is_floating_point()
    ]
    dtype = dtypes[0] if dtypes else torch.float32
    result = op(
        *[
            (x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x)
            for x in tensors
        ],
        **kwargs,
    )
    if isinstance(result, tuple):
        return tuple(
            x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
            for x in result
        )
    return (
        result.to(dtype)
        if isinstance(result, torch.Tensor) and result.is_floating_point()
        else result
    )


class FamilyModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module, family: str):
        super().__init__()
        self.module = install_projection_aliases(module)
        self.adapter = ModelAdapter(module, family)

    def forward(self, *args: Any, **kwargs: Any):
        mask = kwargs.get("attention_mask", kwargs.get("mask", kwargs.get("attn_mask")))
        if (
            self.adapter.position_embedding == "rotary"
            and not getattr(self.module, "supports_rope", False)
            and (
                "position_embeddings" in kwargs
                or "position_ids" in kwargs
                or getattr(self.module, "requires_rope", False)
            )
        ):
            raise QwenIncompatibleError("module_does_not_support_rope")
        if mask is not None:
            adapted = convert_mask(
                mask,
                self.adapter,
                bool(getattr(self.module, "expects_bidirectional", False)),
            )
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = adapted
            elif "mask" in kwargs:
                kwargs["mask"] = adapted
            elif "attn_mask" in kwargs:
                kwargs["attn_mask"] = adapted
        return self.module(*args, **kwargs)


def wrap_for_family(module: nn.Module, model_family: str | None = None) -> nn.Module:
    family = model_family or detect_family(module)
    install_projection_aliases(module)
    family_aliases = globals().get("FAMILY_ALIASES", {})
    canonical = (
        family_aliases.get(family, family)
        if isinstance(family_aliases, dict)
        else family
    )
    return (
        module
        if canonical in {"bert", "roberta", "unknown"}
        else FamilyModuleWrapper(module, family)
    )


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
