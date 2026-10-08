"""Parameter recovery selection and budget pruning for compressed models."""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

# Priority buckets for recovery pruning (Fix CP-01/CP-02)
PROTECTED_KEYWORDS: tuple[str, ...] = (
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

SVD_FACTOR_KEYWORDS: tuple[str, ...] = (
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

LORA_KEYWORDS: tuple[str, ...] = ("lora_a", "lora_b", "lora_")
RECURRENT_TYPE_KEYWORDS: tuple[str, ...] = ("hgrn", "delta", "mamba", "ssm", "recurrent", "jamba")


def _is_protected_param(name: str) -> bool:
    """Check if parameter name matches any protected recovery keyword."""
    n_l = name.lower()
    return any(k in n_l for k in PROTECTED_KEYWORDS)


def _collect_trainable_params(
    model: nn.Module,
    svd_param_ids: set[int],
    has_svd: bool,
    has_lora: bool,
) -> list[nn.Parameter]:
    """Inspect model parameters and filter candidates for fine-tuning recovery.

    Args:
        model: Target PyTorch model.
        svd_param_ids: Set of memory ids for SVD module parameters.
        has_svd: Whether SVD modules exist in the model.
        has_lora: Whether LoRA parameters exist in the model.

    Returns:
        Deduplicated list of candidate trainable parameters.
    """
    fallback_layers: set[str] = set()
    if hasattr(model, "compression_stats") and isinstance(model.compression_stats, dict):
        fb = model.compression_stats.get("fallback_layers", [])
        if isinstance(fb, (list, tuple, set)):
            fallback_layers = set(fb)

    module_by_name = dict(model.named_modules())
    trainable_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        n_low = name.lower()
        parent_module_prefix = (
            name.split(".base_layer.")[0] if ".base_layer." in name else name.rsplit(".", 1)[0]
        )
        layer_has_lora = any(
            k.startswith(parent_module_prefix) and any(t in k.lower() for t in LORA_KEYWORDS)
            for k, _ in model.named_parameters()
        )

        # Recurrent/hybrid module detection.
        is_recurrent_module = False
        parts = name.split(".")
        for i in range(len(parts)):
            sub_prefix = ".".join(parts[:i])
            if sub_prefix in module_by_name:
                mod_cls = module_by_name[sub_prefix].__class__.__name__.lower()
                if any(kw in mod_cls for kw in RECURRENT_TYPE_KEYWORDS):
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
                has_lora and any(d in n_low for d in ("classifier.", "pre_classifier.", "score."))
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
    seen_ids: set[int] = set()
    unique_trainable: list[nn.Parameter] = []
    for p in trainable_params:
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            unique_trainable.append(p)
    return unique_trainable


def _prune_to_budget(
    model: nn.Module,
    trainable_id_set: set[int],
    svd_param_ids: set[int],
    prune_threshold: int,
) -> list[nn.Parameter]:
    """Allocate parameter budget according to strict priority tiers.

    Args:
        model: PyTorch model.
        trainable_id_set: Set of trainable parameter ids.
        svd_param_ids: Set of parameter ids belonging to SVD modules.
        prune_threshold: Maximum parameter count allowed.

    Returns:
        List of retained trainable parameters.
    """
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
        elif id(param) in svd_param_ids or any(kw in n_l for kw in SVD_FACTOR_KEYWORDS):
            priority_buckets[1].append((name, param))
        elif param.ndim <= 1 or param.numel() <= 32768:
            priority_buckets[2].append((name, param))
        else:
            priority_buckets[3].append((name, param))

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

    kept_id_set = {id(p) for p in kept_params}
    for _, param in model.named_parameters():
        param.requires_grad = id(param) in kept_id_set
    return kept_params


def get_recovery_params(model: nn.Module) -> list[nn.Parameter]:
    """Select recovery parameters for LoRA / SVD / injected attention modules.

    Fix CP-01: SVDLinear U/V factor matrices are now preserved during the
    parameter budget pruning step. Previously, their numel (e.g. 64 * 768 =
    49152) exceeded the 32768 threshold and they were pruned, reducing recovery
    to bias-only training.

    Fix CP-02: Pruning now uses priority-based allocation:
        Priority 0 (highest): protected params (LoRA / gates / classifier / alpha / beta)
        Priority 1: SVDLinear U/V factors and factorized parameters
        Priority 2: small params (1D, or numel <= 32768)
        Priority 3 (lowest): other large 2D params

    Args:
        model: The PyTorch neural network module.

    Returns:
        List of parameters chosen for gradient optimization.
    """
    svd_modules = [
        m
        for m in model.modules()
        if m.__class__.__name__ in ("SVDLinear", "SVDLinearLayer", "FactorisedLinear")
    ]
    svd_param_ids = {id(p) for m in svd_modules for p in m.parameters()}
    has_svd = len(svd_modules) > 0

    has_lora = any(any(t in n.lower() for t in LORA_KEYWORDS) for n, _ in model.named_parameters())

    trainable_params = _collect_trainable_params(
        model, svd_param_ids=svd_param_ids, has_svd=has_svd, has_lora=has_lora
    )

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
        trainable_params = _prune_to_budget(
            model=model,
            trainable_id_set=trainable_id_set,
            svd_param_ids=svd_param_ids,
            prune_threshold=prune_threshold,
        )
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
            logger.debug("  %s: %s (%d params)", name, tuple(param.shape), param.numel())

    return trainable_params
