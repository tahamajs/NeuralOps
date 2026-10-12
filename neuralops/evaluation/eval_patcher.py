"""Model patching and module injection framework for NeuralOps evaluation.

Handles base transformer model instantiations (BERT, RoBERTa, GPT-2, T5, Qwen),
state-dict caching, and module injection into target model graphs (attention replacement,
compressor dispatch, KV-cache tuning, and custom injectors).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    RobertaConfig,
    RobertaForSequenceClassification,
    T5Config,
    T5EncoderModel,
)

from neuralops.core.attention_utils import inject_attention
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from neuralops.evaluation.eval_types import DEPRECATED_MODULES
from neuralops.utils.replace_attention import UniversalAttentionWrapper

logger = logging.getLogger("EvalEngine")

# Cache storing (state_dict_cpu, config) for clean base model instantiation
_BASE_MODEL_TEMPLATE_CACHE: dict[
    tuple[str, int, int, int, int, str],
    tuple[dict[str, torch.Tensor], Any],
] = {}


def build_model_instance(
    model_family: str,
    num_labels: int = 2,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    vocab_size: int | None = None,
    seq_len: int = 32,
    device: torch.device | None = None,
) -> tuple[nn.Module, Any]:
    """Instantiate a baseline Transformer model and its configuration on the requested device.

    Args:
        model_family: Target architecture ('bert', 'roberta', 't5', 'gpt2', 'distilbert', 'qwen').
        num_labels: Number of output classification logits.
        hidden_size: Embedding and hidden dimension.
        num_heads: Multi-head attention head count.
        num_layers: Depth / number of transformer layers.
        vocab_size: Vocabulary size (overridden to canonical defaults if unspecified).
        seq_len: Max token sequence length.
        device: PyTorch device to move model to.

    Returns:
        Tuple of (model, config).
    """
    family = model_family.lower()
    if vocab_size is None or vocab_size == 30522:
        if family == "roberta":
            vocab_size = 50265
        elif family == "gpt2":
            vocab_size = 50257
        elif family == "t5":
            vocab_size = 32128
        else:
            vocab_size = 30522

    if family == "bert":
        cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_labels=num_labels,
            max_position_embeddings=512,
        )
        model = BertForSequenceClassification(cfg)
    elif family == "roberta":
        cfg = RobertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_labels=num_labels,
            max_position_embeddings=512,
        )
        model = RobertaForSequenceClassification(cfg)
    elif family == "t5":
        cfg = T5Config(
            vocab_size=vocab_size,
            d_model=hidden_size,
            d_kv=hidden_size // max(1, num_heads),
            d_ff=hidden_size * 4,
            num_layers=num_layers,
            num_decoder_layers=num_layers,
            num_heads=num_heads,
        )
        model = T5EncoderModel(cfg)
    elif family == "gpt2":
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=512,
        )
        model = GPT2LMHeadModel(cfg)
    elif family == "distilbert":
        cfg = DistilBertConfig(
            vocab_size=vocab_size,
            dim=hidden_size,
            n_layers=num_layers,
            n_heads=num_heads,
            hidden_dim=hidden_size * 4,
            num_labels=num_labels,
        )
        model = DistilBertForSequenceClassification(cfg)
    elif family in ("qwen", "qwen2", "qwen3.5", "qwen3_5"):
        from transformers import Qwen2Config, Qwen2ForSequenceClassification

        kv_heads = max(1, num_heads // 2)
        cfg = Qwen2Config(
            vocab_size=151936,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=kv_heads,
            head_dim=hidden_size // num_heads,
            num_labels=num_labels,
            max_position_embeddings=512,
            pad_token_id=0,
        )
        model = Qwen2ForSequenceClassification(cfg)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    model.to(device)
    model.eval()
    return model, cfg


def get_base_model_cached(
    model_family: str,
    num_labels: int = 2,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    device: torch.device | None = None,
) -> tuple[nn.Module, Any]:
    """Retrieve an isolated base model instance via cached state_dict reconstruction.

    Avoids Python deepcopy overhead while guaranteeing pristine weights for benchmarks.

    Args:
        model_family: Target architecture family name.
        num_labels: Number of output classification logits.
        hidden_size: Hidden dimension size.
        num_heads: Multi-head attention head count.
        num_layers: Transformer layer depth.
        device: Target device for execution.

    Returns:
        Tuple of (model, config).
    """
    family = model_family.lower()
    dev_str = str(device)
    cache_key = (family, num_labels, hidden_size, num_heads, num_layers, dev_str)

    if cache_key not in _BASE_MODEL_TEMPLATE_CACHE:
        with torch.device("cpu"):
            template_model, cfg = build_model_instance(
                model_family=family,
                num_labels=num_labels,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                device=torch.device("cpu"),
            )
        template_state = {
            k: v.detach().cpu().clone() for k, v in template_model.state_dict().items()
        }
        del template_model
        _BASE_MODEL_TEMPLATE_CACHE[cache_key] = (template_state, cfg)
    else:
        template_state, cfg = _BASE_MODEL_TEMPLATE_CACHE[cache_key]

    fresh_model, _ = build_model_instance(
        model_family=family,
        num_labels=num_labels,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        device=device,
    )
    fresh_model.load_state_dict(
        {k: v.to(device) for k, v in template_state.items()},
        strict=False,
    )
    return fresh_model, cfg


def _apply_attention_module(
    model: nn.Module,
    module_name: str,
    rank: int,
    device: torch.device,
) -> nn.Module:
    """Inject attention module using custom injector or standard inject_attention.

    Args:
        model: Target PyTorch model.
        module_name: Name of attention module in ATTENTION_REGISTRY.
        rank: Rank parameter for low-rank / linear attention variants.
        device: PyTorch device.

    Returns:
        Patched model instance.
    """
    if module_name not in ATTENTION_REGISTRY:
        try:
            importlib.import_module(f"neuralops.attentions.{module_name}")
        except ImportError as e:
            logger.warning("Could not import attention '%s': %s", module_name, e)

    resolved_mod = ATTENTION_REGISTRY.get(module_name)
    custom_injector = None
    if (
        resolved_mod is not None
        and callable(resolved_mod)
        and not isinstance(resolved_mod, type)
    ):
        custom_injector = resolved_mod
    if custom_injector is None:
        cand = (
            ATTENTION_REGISTRY.get(f"inject_{module_name}")
            or ATTENTION_REGISTRY.get(f"{module_name}_inject")
            or ATTENTION_REGISTRY.get(f"inject_{module_name}_attention")
        )
        if cand is not None and not isinstance(cand, type):
            custom_injector = cand

    if custom_injector is not None and callable(custom_injector) and not isinstance(custom_injector, type):
        try:
            res = custom_injector(model=model, rank=rank, copy_weights=True)
            is_single_layer = isinstance(res, (BaseAttention, nn.Linear)) or (
                "Attention" in type(res).__name__
                and not any(
                    hasattr(res, a)
                    for a in (
                        "encoder",
                        "transformer",
                        "bert",
                        "roberta",
                        "distilbert",
                        "gpt2",
                        "qwen2",
                    )
                )
            )
            is_whole_model = (
                res is not None
                and isinstance(res, nn.Module)
                and sum(p.numel() for p in res.parameters()) > 1000
            )
            if is_whole_model:
                model = res
                if device is not None:
                    model = model.to(device)
                return model
            elif is_single_layer:
                logger.debug(
                    "Injector %s returned a single layer; using inject_attention.",
                    custom_injector,
                )
        except Exception as e:
            logger.debug("Injector %s failed: %s", custom_injector, e)

    if module_name in ATTENTION_REGISTRY:
        inject_attention(model, module_name, rank=rank, copy_weights=True)
    else:
        mod = importlib.import_module(f"neuralops.attentions.{module_name}")
        target_cls = None
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, nn.Module)
                and ("Attention" in attr or "Block" in attr)
                and attr != "BaseAttention"
            ):
                target_cls = obj
                break
        if target_cls is None:
            raise ValueError(f"No valid attention class in '{module_name}'")
        cfg = getattr(model, "config", None)
        for name, sub in list(model.named_modules()):
            if "attention" in name.lower() and hasattr(sub, "self"):
                new_inner = target_cls(cfg)
                wrapped = UniversalAttentionWrapper(new_inner)
                sub.self = wrapped

    return model


def _invoke_compressor_instance(
    instance: Any,
    model: nn.Module,
    calibration_loader: Any | None,
    calibration_data: Any | None,
) -> nn.Module | None:
    """Invoke compression hooks (.compress, .apply, .prune) on a compressor instance."""
    for method_name in ("compress", "apply", "prune"):
        if not hasattr(instance, method_name):
            continue
        m = getattr(instance, method_name)
        if not callable(m):
            continue

        attempts = [lambda: m()]
        if calibration_loader is not None:
            attempts.append(lambda: m(calibration_loader=calibration_loader))
        if calibration_data is not None:
            attempts.append(lambda: m(calibration_data=calibration_data))
        attempts.append(lambda: m(model))
        if calibration_loader is not None:
            attempts.append(lambda: m(model, calibration_loader=calibration_loader))
        if calibration_data is not None:
            attempts.append(lambda: m(model, calibration_data=calibration_data))

        for fn in attempts:
            try:
                res = fn()
                if isinstance(res, tuple) and res and isinstance(res[0], nn.Module):
                    if len(res) > 1 and isinstance(res[1], dict):
                        try:
                            res[0].compression_stats = res[1]
                        except Exception:
                            pass
                    return res[0]
                if isinstance(res, nn.Module):
                    return res
            except (TypeError, ValueError, RuntimeError):
                continue

    m = getattr(instance, "model", None)
    if m is not None and hasattr(instance, "get_compression_stats"):
        try:
            m.compression_stats = instance.get_compression_stats()
        except Exception:
            pass
    return m


def _apply_compressor_module(
    model: nn.Module,
    module_name: str,
    rank: int,
    calibration_data: Any | None,
    calibration_loader: Any | None,
    tokenizer: Any | None,
) -> nn.Module:
    """Apply compression engine module to target model using registry instantiation strategies."""
    if module_name not in COMPRESSOR_REGISTRY:
        try:
            importlib.import_module(f"neuralops.compression_engine.{module_name}")
        except ImportError as e:
            logger.warning("Could not import compressor '%s': %s", module_name, e)

    comp_cls = COMPRESSOR_REGISTRY.get(module_name)
    if comp_cls is None:
        raise RuntimeError(f"Compressor '{module_name}' not in COMPRESSOR_REGISTRY")

    tok = (
        tokenizer
        or getattr(model, "tokenizer", None)
        or getattr(getattr(model, "config", None), "tokenizer", None)
    )
    cfg = getattr(model, "config", None)

    c_kwargs: dict[str, Any] = {}
    if cfg is not None:
        for k in (
            "rank_ratio",
            "max_error",
            "lora_rank",
            "target_attention_only",
            "use_jg_svd",
            "use_cholesky_jg",
            "energy_threshold",
            "group_size",
            "strict_rank",
            "n_calib",
            "calib_samples",
            "keep_ratio",
            "bits",
        ):
            if hasattr(cfg, k):
                c_kwargs[k] = getattr(cfg, k)

    if calibration_loader is not None:
        c_kwargs["calibration_loader"] = calibration_loader
        c_kwargs["calib_loader"] = calibration_loader
    if calibration_data is not None:
        c_kwargs["calibration_data"] = calibration_data

    # Strategy 1: instantiate with model
    candidate = None
    for init_fn in [
        lambda: comp_cls(model, rank=rank, tokenizer=tok, **c_kwargs),
        lambda: comp_cls(model, rank=rank, tokenizer=tok),
        lambda: comp_cls(model, rank=rank, **c_kwargs),
        lambda: comp_cls(model, rank=rank),
        lambda: comp_cls(model),
    ]:
        try:
            candidate = init_fn()
            if candidate is not None:
                break
        except (TypeError, ValueError):
            continue

    if candidate is not None:
        if isinstance(candidate, tuple) and candidate and isinstance(candidate[0], nn.Module):
            model = candidate[0]
            if len(candidate) > 1 and isinstance(candidate[1], dict):
                try:
                    model.compression_stats = candidate[1]
                except Exception:
                    pass
        elif isinstance(candidate, nn.Module) and not hasattr(candidate, "compress"):
            model = candidate
        else:
            res_mod = _invoke_compressor_instance(
                candidate, model, calibration_loader, calibration_data
            )
            if res_mod is not None and isinstance(res_mod, nn.Module):
                model = res_mod
    else:
        # Strategy 2: instantiate without model
        candidate = None
        for init_fn in [
            lambda: comp_cls(rank=rank, tokenizer=tok, **c_kwargs),
            lambda: comp_cls(rank=rank),
            lambda: comp_cls(),
        ]:
            try:
                candidate = init_fn()
                if candidate is not None:
                    break
            except (TypeError, ValueError):
                continue

        if candidate is None:
            raise RuntimeError(
                f"Could not instantiate compressor '{module_name}' with any signature."
            )
        res_mod = _invoke_compressor_instance(
            candidate, model, calibration_loader, calibration_data
        )
        if res_mod is not None and isinstance(res_mod, nn.Module):
            model = res_mod

    return model


def apply_module_to_model(
    model: nn.Module,
    module_name: str,
    module_type: str = "auto",
    rank: int = 16,
    device: torch.device | None = None,
    calibration_data: Any | None = None,
    calibration_loader: Any | None = None,
    tokenizer: Any | None = None,
) -> tuple[nn.Module, float]:
    """Inject attention or apply compression engine to model and calculate parameter ratio.

    Args:
        model: PyTorch model instance to patch.
        module_name: Name of module from ATTENTION_REGISTRY or COMPRESSOR_REGISTRY.
        module_type: 'attention', 'compressor', or 'auto' (determined dynamically).
        rank: Low-rank rank or bottleneck size.
        device: Target execution device.
        calibration_data: Optional tensor calibration batch.
        calibration_loader: Optional DataLoader containing real calibration samples.
        tokenizer: Optional AutoTokenizer matching the target model.

    Returns:
        Tuple of (patched_model, compression_ratio).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_params = sum(p.numel() for p in model.parameters())

    if not module_name or module_name.lower() in ("none", "baseline", "dense", "full"):
        return model, 1.0

    if module_name == "hla_inject":
        module_name = "hla_attention"

    if module_name in DEPRECATED_MODULES:
        logger.warning("Module %s is DEPRECATED; skipping.", module_name)
        return model, 1.0

    if module_name in ATTENTION_REGISTRY:
        module_type = "attention"
    elif module_name in COMPRESSOR_REGISTRY:
        module_type = "compressor"
    else:
        import importlib.util

        if importlib.util.find_spec(f"neuralops.compression_engine.{module_name}"):
            module_type = "compressor"
        elif importlib.util.find_spec(f"neuralops.attentions.{module_name}"):
            module_type = "attention"

    if module_type == "attention":
        model = _apply_attention_module(model, module_name, rank, device)
    elif module_type == "compressor":
        model = _apply_compressor_module(
            model, module_name, rank, calibration_data, calibration_loader, tokenizer
        )

    model.to(device)
    new_params = sum(p.numel() for p in model.parameters())
    ratio = orig_params / max(1, new_params)
    return model, ratio


__all__ = [
    "build_model_instance",
    "get_base_model_cached",
    "apply_module_to_model",
]
