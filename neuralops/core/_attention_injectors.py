"""
Architecture-specific attention layer preparation, instantiation, and injection engine.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from neuralops.core._attention_surgery import (
    FUSED_QKV_REGEX,
    PROJ_REGEX_PATTERNS,
    _copy_projection_weights,
    _find_submodule_by_regex,
    _slice_fused_qkv_weights,
    _svd_factorize_weights,
)
from neuralops.core.adapters import get_adapter
from neuralops.core.architecture import detect_architecture, get_attention_modules
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.model_family import ModelAdapter, detect_family
from neuralops.utils.qwen_compat import QwenIncompatibleError
from neuralops.utils.replace_attention import UniversalAttentionWrapper

logger = logging.getLogger(__name__)


def _prepare_layer_config(
    old_attn: nn.Module,
    config: Any,
    family_adapter: Any,
    family: str,
    is_dec: bool,
    adapter: Any,
) -> dict[str, Any]:
    """Resolve hidden_size, num_heads, head_dim, num_kv_heads, and projection aliases."""
    hidden_size = None
    num_heads = None
    head_dim = None
    num_kv_heads = None

    if family_adapter is not None:
        hidden_size = getattr(family_adapter, "hidden_size", None)
        num_heads = getattr(family_adapter, "num_heads", None)
        num_kv_heads = getattr(family_adapter, "num_key_value_heads", None)
        head_dim = getattr(family_adapter, "head_dim", None)

    if hidden_size is None or num_heads is None:
        for _, child in old_attn.named_children():
            if (
                hasattr(child, "weight")
                and isinstance(child.weight, torch.Tensor)
                and child.weight.ndim == 2
            ):
                if hidden_size is None:
                    hidden_size = child.weight.shape[1]
                break
        if num_heads is None:
            num_heads = getattr(
                old_attn,
                "num_heads",
                getattr(old_attn, "num_attention_heads", None),
            )
        if num_kv_heads is None:
            num_kv_heads = getattr(old_attn, "num_key_value_heads", num_heads)
        if head_dim is None and hidden_size is not None and num_heads is not None:
            head_dim = getattr(old_attn, "head_dim", hidden_size // max(1, num_heads))

    # Ensure config exists or construct minimal namespace config
    if config is None:
        is_decoder_family_bool = bool(
            is_dec or family in ("gpt", "gpt2", "qwen", "qwen3_5", "llama", "mistral")
        )
        config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_kv_heads or num_heads,
            dropout=0.1,
            is_decoder=is_decoder_family_bool,
            is_causal=is_decoder_family_bool,
            model_type=family,
        )
    else:
        if family in ("gpt", "gpt2", "qwen", "qwen3_5", "llama", "mistral") or is_dec:
            config.is_decoder = True
            config.is_causal = True

    # Ensure old_attn has canonical projection attributes for custom modules inspecting pretrained_attn
    for proj_key, alias_list in [
        ("q", ["query", "q_proj", "q_lin", "Wq", "wq", "q"]),
        ("k", ["key", "k_proj", "k_lin", "Wk", "wk", "k"]),
        ("v", ["value", "v_proj", "v_lin", "Wv", "wv", "v"]),
        ("o", ["dense", "out_proj", "o_proj", "out_lin", "c_proj", "wo", "o"]),
    ]:
        try:
            existing_mod = next(
                (getattr(old_attn, a) for a in alias_list if hasattr(old_attn, a)),
                None,
            )
            if existing_mod is None and hasattr(old_attn, "self"):
                existing_mod = next(
                    (
                        getattr(old_attn.self, a)
                        for a in alias_list
                        if hasattr(old_attn.self, a)
                    ),
                    None,
                )

            if existing_mod is not None:
                for a in alias_list:
                    if not hasattr(old_attn, a):
                        try:
                            setattr(old_attn, a, existing_mod)
                        except Exception:
                            pass
            elif adapter is not None:
                w, b = adapter.get_weight_bias(old_attn, proj_key)
                if w is not None:
                    proxy = nn.Linear(
                        w.shape[1],
                        w.shape[0],
                        bias=(b is not None),
                        device=w.device,
                        dtype=w.dtype,
                    )
                    proxy.weight.data.copy_(w)
                    if b is not None:
                        proxy.bias.data.copy_(b)
                    for a in alias_list:
                        if not hasattr(old_attn, a):
                            try:
                                setattr(old_attn, a, proxy)
                            except Exception:
                                pass
        except Exception as ex:
            logger.debug("Failed to set projection aliases for %s: %s", proj_key, ex)

    return {
        "config": config,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
    }


def _instantiate_attention_layer(
    attn_class: type,
    config: Any,
    effective_rank: int,
    old_attn: nn.Module,
    family_adapter: Any,
    family: str,
    attention_type: str,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate attention layer trying supported signatures in order and checking GQA compatibility."""
    attempts = [
        lambda: attn_class(config, rank=effective_rank, pretrained_attn=old_attn, **kwargs),
        lambda: attn_class(config, pretrained_attn=old_attn, **kwargs),
        lambda: attn_class(config=config, pretrained_attn=old_attn, **kwargs),
        lambda: attn_class(config, rank=effective_rank, **kwargs),
        lambda: attn_class(config, **kwargs),
        lambda: attn_class(rank=effective_rank, **kwargs),
        lambda: attn_class(**kwargs),
    ]
    new_attn = None
    for attempt in attempts:
        try:
            new_attn = attempt()
            break
        except TypeError:
            continue
        except Exception as e:
            logger.debug("Instantiation attempt failed: %s", e)
            continue

    if new_attn is None:
        raise RuntimeError(f"Could not instantiate attention class {attn_class.__name__}")

    # Check GQA geometry preservation
    is_gqa = (
        family_adapter is not None
        and family in ("qwen", "qwen3_5")
        and family_adapter.num_key_value_heads != family_adapter.num_heads
    )
    if is_gqa and not getattr(new_attn, "supports_gqa", False):
        expected = {
            "q": family_adapter.q_proj_out,
            "k": family_adapter.kv_proj_out,
            "v": family_adapter.kv_proj_out,
        }
        for key, patterns in (
            ("q", PROJ_REGEX_PATTERNS["q"]),
            ("k", PROJ_REGEX_PATTERNS["k"]),
            ("v", PROJ_REGEX_PATTERNS["v"]),
        ):
            target = _find_submodule_by_regex(new_attn, patterns)
            out_features = getattr(target, "out_features", None)
            if target is not None and out_features not in {expected[key], None}:
                if expected[key] is not None and out_features is not None:
                    if (
                        out_features % expected[key] == 0
                        or expected[key] % out_features == 0
                        or _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS["o"]) is not None
                    ):
                        pass
                    else:
                        raise QwenIncompatibleError(
                            f"{attention_type}: {key}_projection_out={out_features}, "
                            f"requires={expected[key]} (GQA)"
                        )

    return new_attn


def _transplant_weights(
    old_attn: nn.Module,
    new_attn: nn.Module,
    adapter: Any,
    num_heads: int | None,
    num_kv_heads: int | None,
    head_dim: int | None,
    is_dec: bool,
    hidden_size: int | None = None,
) -> None:
    """Transfer weights: custom copy_weights, fused QKV slicing + SVD, or decoupled projections."""
    old_param = next(old_attn.parameters(), None)
    if old_param is not None:
        new_attn.to(device=old_param.device, dtype=old_param.dtype)

    if (
        is_dec
        or hasattr(old_attn, "c_attn")
        or "gpt" in old_attn.__class__.__name__.lower()
    ):
        old_attn.is_decoder = True
        if hasattr(new_attn, "is_decoder"):
            new_attn.is_decoder = True

    # Branch 1: Custom copy_weights hook
    if hasattr(new_attn, "copy_weights") and callable(new_attn.copy_weights):
        try:
            new_attn.copy_weights(old_attn)
            if is_dec and hasattr(new_attn, "is_decoder"):
                new_attn.is_decoder = True
            if old_param is not None:
                new_attn.to(device=old_param.device, dtype=old_param.dtype)
            return
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.debug("Custom copy_weights failed: %s", e)
        except Exception as e:
            logger.warning("Unexpected error in custom copy_weights: %s", e)

    # Branch 2: Fused QKV projection module
    fused_mod = None
    for child_name, child_mod in old_attn.named_children():
        if FUSED_QKV_REGEX.match(child_name) and hasattr(child_mod, "weight"):
            fused_mod = child_mod
            break
    if fused_mod is None:
        for candidate in ["c_attn", "qkv_proj", "query_key_value", "in_proj", "Wqkv"]:
            mod = getattr(old_attn, candidate, None)
            if mod is not None and hasattr(mod, "weight"):
                fused_mod = mod
                break

    if fused_mod is not None:
        sliced = _slice_fused_qkv_weights(
            fused_mod.weight.data,
            (
                fused_mod.bias.data
                if hasattr(fused_mod, "bias") and fused_mod.bias is not None
                else None
            ),
            num_heads=num_heads if num_heads is not None else 12,
            num_kv_heads=num_kv_heads or num_heads,
            head_dim=head_dim,
            source_mod=fused_mod,
        )
        for proj_key in ["q", "k", "v"]:
            w, b = sliced[proj_key]
            target_proj = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS[proj_key])
            if target_proj is not None:
                _copy_projection_weights(target_proj, w, b)
            else:
                lin1 = getattr(new_attn, f"{proj_key}_proj1", None)
                lin2 = getattr(new_attn, f"{proj_key}_proj2", None)
                if lin1 is not None and lin2 is not None:
                    _copy_projection_weights(lin1, w, None)
                    _svd_factorize_weights(w, b, lin1, lin2)

        # Output projection
        try:
            src_w_o, src_b_o = adapter.get_weight_bias(old_attn, "o") if adapter else (None, None)
            target_o = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS["o"])
            if target_o is not None and src_w_o is not None:
                _copy_projection_weights(target_o, src_w_o, src_b_o)
            elif src_w_o is not None:
                lin1_o = getattr(new_attn, "out_proj1", getattr(new_attn, "o_proj1", None))
                lin2_o = getattr(new_attn, "out_proj2", getattr(new_attn, "o_proj2", None))
                if lin1_o is not None and lin2_o is not None:
                    _svd_factorize_weights(src_w_o, src_b_o, lin1_o, lin2_o)
        except (ValueError, AttributeError) as e:
            logger.debug("Could not copy output projection: %s", e)
        return

    # Branch 3: Standard decoupled Q, K, V, O projections
    for proj_key in ["q", "k", "v", "o"]:
        src_weight, src_bias = None, None
        if adapter is not None:
            try:
                src_weight, src_bias = adapter.get_weight_bias(old_attn, proj_key)
            except (ValueError, AttributeError) as e:
                logger.debug("Could not get %s projection weights: %s", proj_key, e)

        target_proj = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS[proj_key])
        if target_proj is not None:
            if src_weight is not None:
                _copy_projection_weights(target_proj, src_weight, src_bias)
            elif proj_key == "o":
                # Encoder self-attention without internal out_proj -> initialize to identity
                with torch.no_grad():
                    if hasattr(target_proj, "weight") and target_proj.weight.ndim == 2:
                        if target_proj.weight.shape[0] == target_proj.weight.shape[1]:
                            nn.init.eye_(target_proj.weight)
                        else:
                            nn.init.zeros_(target_proj.weight)
                            min_d = min(target_proj.weight.shape[0], target_proj.weight.shape[1])
                            target_proj.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    elif hasattr(target_proj, "set_svd_factors") or (
                        hasattr(target_proj, "U") and hasattr(target_proj, "V")
                    ):
                        d_out = getattr(target_proj, "out_features", getattr(target_proj, "d_out", hidden_size or 768))
                        d_in = getattr(target_proj, "in_features", getattr(target_proj, "d_in", hidden_size or 768))
                        eye_mat = torch.eye(
                            d_out,
                            d_in,
                            device=(target_proj.U.device if hasattr(target_proj, "U") else None),
                        )
                        _copy_projection_weights(target_proj, eye_mat, None)
                    if hasattr(target_proj, "bias") and target_proj.bias is not None:
                        nn.init.zeros_(target_proj.bias)
        else:
            lin1 = getattr(new_attn, f"{proj_key}_proj1", getattr(new_attn, f"{proj_key}1", None))
            lin2 = getattr(new_attn, f"{proj_key}_proj2", getattr(new_attn, f"{proj_key}2", None))
            if proj_key == "o":
                lin1 = lin1 or getattr(new_attn, "out_proj1", getattr(new_attn, "o_proj1", None))
                lin2 = lin2 or getattr(new_attn, "out_proj2", getattr(new_attn, "o_proj2", None))
            if lin1 is not None and lin2 is not None:
                if src_weight is not None:
                    _svd_factorize_weights(src_weight, src_bias, lin1, lin2)
                elif proj_key == "o":
                    with torch.no_grad():
                        if hasattr(lin1, "weight") and hasattr(lin2, "weight"):
                            nn.init.zeros_(lin1.weight)
                            nn.init.zeros_(lin2.weight)
                            r = min(
                                lin1.weight.shape[0],
                                lin2.weight.shape[1],
                                lin1.weight.shape[1],
                                lin2.weight.shape[0],
                            )
                            lin1.weight.data[:r, :r].fill_diagonal_(1.0)
                            lin2.weight.data[:r, :r].fill_diagonal_(1.0)
                            if getattr(lin2, "bias", None) is not None:
                                nn.init.zeros_(lin2.bias)


def _wrap_and_replace_layer(
    model: nn.Module,
    layer_name: str,
    old_attn: nn.Module,
    new_attn: nn.Module,
    family: str,
) -> None:
    """Wrap new attention in UniversalAttentionWrapper and perform dynamic module swap."""
    return_tuple = (
        "self_attn" in layer_name.lower()
        or "attention" in layer_name.lower()
        or getattr(old_attn, "is_decoder", False)
        or family == "gpt"
        or layer_name.endswith(".attn")
    )
    wrapped_attn = UniversalAttentionWrapper(
        new_attn, layer_name=layer_name, return_tuple=return_tuple
    )
    replace_module(model, layer_name, wrapped_attn)


def _inject_attention_loop(
    model: nn.Module,
    attention_type: str,
    rank: int | None = None,
    copy_weights: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Main injection orchestrator iterating over model attention modules."""
    attn_class = ATTENTION_REGISTRY.resolve(attention_type)
    if callable(attn_class) and not isinstance(attn_class, type):
        try:
            res = attn_class(model=model, rank=rank, copy_weights=copy_weights, **kwargs)
            if res is not None:
                return res
        except Exception as inj_err:
            logger.debug(
                "Dedicated injector %s failed, falling back to layer replacement: %s",
                attention_type,
                inj_err,
            )

    config = getattr(model, "config", None)
    if config is not None:
        cfg_source = getattr(config, "text_config", config)
        hidden_size = getattr(
            cfg_source,
            "hidden_size",
            getattr(
                cfg_source,
                "d_model",
                getattr(cfg_source, "n_embd", getattr(cfg_source, "dim", None)),
            ),
        )
        num_heads = getattr(
            cfg_source,
            "num_attention_heads",
            getattr(
                cfg_source,
                "num_heads",
                getattr(cfg_source, "n_heads", getattr(cfg_source, "n_head", None)),
            ),
        )
        head_dim = getattr(
            cfg_source,
            "head_dim",
            getattr(
                cfg_source,
                "d_kv",
                (hidden_size // max(1, num_heads) if (hidden_size and num_heads) else None),
            ),
        )
        num_kv_heads = getattr(cfg_source, "num_key_value_heads", num_heads)
        if hidden_size is not None:
            config.hidden_size = hidden_size
        if num_heads is not None:
            config.num_attention_heads = num_heads
        if head_dim is not None:
            config.head_dim = head_dim
        if num_kv_heads is not None:
            config.num_key_value_heads = num_kv_heads
        if not hasattr(config, "dropout"):
            dropout_val = getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
            config.dropout = dropout_val

    arch = detect_architecture(model)
    adapter = get_adapter(arch)
    family = detect_family(model)
    family_adapter = ModelAdapter(model, family)

    is_dec = family in ("gpt", "gpt2", "qwen", "qwen3_5", "llama", "mistral")
    if config is not None:
        config.is_decoder = bool(is_dec or getattr(config, "is_decoder", False))
        if not hasattr(config, "model_type") or not config.model_type:
            config.model_type = family

    replaced = 0
    modules_list = list(get_attention_modules(model))
    if not modules_list:
        logger.warning("No attention modules found to replace for architecture '%s'", arch)

    for layer_i, (name, old_attn) in enumerate(modules_list):
        if "linear_attn" in name and not getattr(attn_class, "supports_delta_net", False):
            continue

        prep = _prepare_layer_config(
            old_attn,
            config,
            family_adapter,
            family,
            is_dec,
            adapter,
        )
        layer_cfg = prep["config"]
        effective_rank = rank if rank is not None else getattr(layer_cfg, "rank", 32)

        layer_kwargs = dict(kwargs)
        layer_kwargs.setdefault("layer_idx", layer_i)
        layer_kwargs.setdefault("layer_id", layer_i)

        new_attn = _instantiate_attention_layer(
            attn_class,
            layer_cfg,
            effective_rank,
            old_attn,
            family_adapter,
            family,
            attention_type,
            **layer_kwargs,
        )

        if copy_weights:
            _transplant_weights(
                old_attn,
                new_attn,
                adapter,
                num_heads=prep["num_heads"],
                num_kv_heads=prep["num_kv_heads"],
                head_dim=prep["head_dim"],
                is_dec=is_dec,
                hidden_size=prep["hidden_size"],
            )

        _wrap_and_replace_layer(model, name, old_attn, new_attn, family)
        replaced += 1

    logger.info(
        "Successfully injected '%s' into %d transformer blocks.",
        attention_type,
        replaced,
    )
    if not hasattr(model, "compression_stats"):
        model.compression_stats = {}
    model.compression_stats["num_compressed"] = replaced
    model.compression_stats["total_params_after"] = sum(p.numel() for p in model.parameters())
    return model


__all__ = [
    "_prepare_layer_config",
    "_instantiate_attention_layer",
    "_transplant_weights",
    "_wrap_and_replace_layer",
    "_inject_attention_loop",
]
