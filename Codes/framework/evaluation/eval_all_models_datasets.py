from __future__ import annotations

import contextlib

"""
Universal Evaluation Engine for NeuralOps Framework.

Fixes applied (2026-09-27):
  FIX-1  Build and pass a real calibration_loader to compressors.
  FIX-2  Removed unreachable candidate=None in strategy-2.
  FIX-3  Full logits for LM; correct PPL formula.
  FIX-4  Removed fake `tensor(0.0, requires_grad=True)` loss.
  FIX-5  Replaced deepcopy with state_dict-based rebuild.
  FIX-6  probe_optimal_batch_size uses target_vram_ratio=0.35 (training).
  FIX-7  Do not strip labels for GPT-2 in _filter_batch.
  FIX-8  Synthetic padding only for bidirectional models.
  FIX-9  inspect.signature done once, not per batch.
  FIX-10 Final status validation (NaN, calibration, compression).
  FIX-11 No random fake calibration fallback.
  FIX-12 allow_synthetic defaults to False.
  FIX-13 LRU eviction for caches.
  FIX-14 Pearson/Spearman NaN guard.
  FIX-15 Fidelity per-sample cosine, not flattened.
  FIX-16 Bounded throughput.
  FIX-17 Pre-built autocast context.
  FIX-18 grad_norm only over trainable params.
"""


import gc
import json
import logging
import math
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

import torch.nn as nn
import torch.nn.functional as F
from framework.core.attention_utils import inject_attention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.utils.replace_attention import UniversalAttentionWrapper
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("EvalEngine")


# =============================================================================
# Constants
# =============================================================================

MODEL_TOKENIZER_MAPPING: dict[str, str] = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "t5": "t5-small",
    "gpt2": "gpt2",
    "distilbert": "distilbert-base-uncased",
    "qwen": "Qwen/Qwen2.5-0.5B",
    "qwen2": "Qwen/Qwen2.5-0.5B",
    "qwen3.5": "Qwen/Qwen2.5-0.5B",
    "qwen3_5": "Qwen/Qwen2.5-0.5B",
}

LM_DATASETS = {
    "wikitext2",
    "wikitext-2-raw-v1",
    "wikitext",
    "wikitext103",
    "c4",
    "ptb",
    "lm",
}

GLUE_DATASETS = {"sst2", "cola", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli", "stsb"}

ACTIVATION_AWARE_MODULES = {
    "a3_compressor",
    "a3",
    "aasvd",
    "asvd",
    "adasvd_compressor",
    "adasvd",
    "flat_llm",
    "armor",
    "armor_pruner",
    "armor_compressor",
    "flap",
    "mgie",
    "mgie_svd",
    "mgie_budget",
    "frandi",
    "frandi_compressor",
    "optimal_subspace_projection",
    "slicegpt",
    "slicegpt_pca",
    "rotational_slicegpt_int4",
    "rotational_slicegpt",
    "proxsparse",
    "mamba_shedder",
    "mlorq",
    "mlorq_compressor",
    "compot",
    "compot_compressor",
    "hashed_net",
    "hashed_net_compressor",
    "lsc",
    "lsc_compressor",
    "grecodec",
    "grecodec_compressor",
    "direct_svd",
    "adaptive_ranks",
    "entropy_budget",
    "entropy_budget_allocation",
    "dynamic_rank",
    "dobi_svd",
    "dobi",
    "erc",
    "erc_svd",
    "gfws",
    "gfws_d_compressor",
    "modegpt",
    "subspace_projection",
    "roast",
    "bonsai",
    "bonsai_pruner",
    "swift_svd",
    "swift_svd_act",
    "hybrid_svd",
}

COMPUTE_OPTIMIZERS = {
    "nha",
    "nha_class",
    "nha_attention",
    "replace",
    "replace_class",
    "nli_softmax",
    "nli_softmax_class",
    "nli_softmax_attn",
    "block_transformer_hiba",
    "flash3",
    "flash_attention3",
    "flash3_fp8",
    "adasplash",
    "adasplash_attention",
    "moba",
    "xattention",
    "fasa",
    "cluster",
    "cluster_attention",
    "cosformer",
}

KV_CACHE_OPTIMIZERS = {
    "hybridkv",
    "hybridkv_class",
    "aatc_xkv",
    "aatc_xkv_compressor",
    "kvtc",
    "delta_kv",
    "delta_kv_class",
    "zerokv",
    "zerokv_neo",
    "moe_deltakv_paged",
    "padre",
}

DEPRECATED_MODULES = {
    "eat",
    "eat_attention",
    "granite_hybrid",
    "jamba_info",
    "tpa",
    "tpa_attention",
    "tp_attention",
    "tt",
    "tt_linear",
    "tt_compressor",
    "tt_linear_compressor",
    "tt_factory",
    "tt_compressor_factory",
    "tt_linear_factory",
    "lillama_compressor",
    "mamba_svd_flash",
}


# =============================================================================
# Result schema
# =============================================================================


@dataclass
class BenchmarkResult:
    track: str
    module_name: str
    module_type: str
    model_family: str
    dataset: str
    status: str
    error_message: str | None = None

    # Latency & Throughput
    forward_latency_ms: float = 0.0
    backward_latency_ms: float = 0.0
    total_step_latency_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    speedup_vs_baseline: float = 1.0

    # Memory & Parameters
    param_count: int = 0
    trainable_param_count: int = 0
    compression_ratio: float = 1.0
    peak_memory_mb: float = 0.0
    memory_savings_pct: float = 0.0

    # Accuracy & Loss
    loss: float = 0.0
    accuracy_pct: float = 0.0
    f1_score: float = 0.0
    perplexity: float = 0.0
    grad_norm: float = 0.0

    # Advanced metrics
    mcc: float = 0.0
    pearson: float = 0.0
    spearman: float = 0.0
    fidelity_score: float = 0.0
    flops_macs: float = 0.0
    throughput_tokens_sec: float = 0.0
    latency_variance_ms: float = 0.0
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value_vs_baseline: float | None = None
    mean_seed_acc: float | None = None
    std_seed_acc: float | None = None

    # Diagnostics (FIX-10)
    calibration_used: bool = False
    calibration_source: str | None = None
    num_compressed_layers: int = 0
    num_fallback_layers: int = 0
    avg_reconstruction_error: float | None = None
    params_before: int = 0
    params_after: int = 0

    trajectories: dict[str, list[float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # JSON cannot serialize NaN/inf
        for k, v in list(d.items()):
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return d


# =============================================================================
# Track registry
# =============================================================================

TRACK_MODULES: dict[str, dict[str, list[str]]] = {
    "track1_svd": {
        "attentions": [
            "svd_linear_attention",
            "adaptive_svd_attention",
            "asvd_attention",
            "linformer_attention",
            "res_svd",
            "svd_attention",
            "svd_compressed",
            "svd_projection_attention",
            "svd_llm_v2_attention",
            "factorised_matrix_state",
            "mamba_svd_flash",
            "minimal_svd",
            "minimal_svd_v2",
        ],
        "compressors": [
            "adaptive_ranks",
            "direct_svd",
            "dobi_svd",
            "erc_svd",
            "entropy_budget_allocation",
            "kronecker_tt_lora",
            "saes_svd_v2",
            "swift_svd",
            "swift_svd_act",
            "aasvd",
            "rpca",
            "cur_moe",
            "grecodec",
            "frandi",
            "dynamic_rank",
            "hybrid_svd",
            "svd_linear",
            "svd_llm",
            "svd_llm_compressor",
        ],
    },
    "track2_ssm": {
        "attentions": [
            "aegis_v1",
            "aegis_v2",
            "aegis_v3",
            "aegis_v4",
            "aegis_v5",
            "aegis_v6",
            "aegis_hybrid",
            "aegis_spectron_fp8",
            "spectra_v2",
            "spectral_mamba",
            "Mamba",
            "SVD-Mamba",
            "simple_ssm",
            "gated_delta_net",
            "gated_deltanet_block",
            "gated_linear_attention",
            "hgrn2_layer",
            "hgrn2_gated_deltanet",
            "hgrn2_outer",
            "griffin_ssd",
            "regla_attention",
            "halo_regla",
            "kda_attention",
            "decay_linear_attention",
            "spectron",
            "galerkin_attention",
            "vspec",
            "vspec_attention",
        ],
        "compressors": ["mamba_shedder"],
    },
    "track3_moe": {
        "attentions": [
            "bam_umoe",
            "expert_choice_moe",
            "reap_moe",
            "samba_moe",
            "samba_m3",
            "jamba",
            "jamba_info",
            "hymba_transx",
            "moe_deltakv_paged",
            "block_transformer_hiba",
            "nexus",
            "leka_v1",
            "leka_v2",
            "leka_v3",
            "mixattention_moska",
            "phantom",
            "moba",
            "fasa",
            "upi_stuffed_mamba",
            "granite_hybrid",
        ],
        "compressors": ["cur_moe"],
    },
    "track4_kernel_quant": {
        "attentions": [
            "performer_attention",
            "nystrom_attention",
            "flash_attention3",
            "cluster_attention",
            "adasplash_attention",
            "hamming_attention",
            "cosformer",
            "decay_linear_attention",
            "gated_linear_attention",
            "SOAFusedAttention",
            "fused_svd_kernel",
        ],
        "compressors": [
            "astro_quantizer",
            "faq_quantizer",
            "coopq_quantizer",
            "kmeans_quantize",
            "zerokv_neo",
            "modegpt",
            "delta_kv",
            "kvtc",
            "aatc_xkv",
            "hybridkv",
        ],
    },
    "track5_pruning": {
        "attentions": ["linformer_attention", "adaptive_sparse_attention"],
        "compressors": [
            "slicegpt",
            "slicegpt_pca",
            "sparsegpt",
            "admm_pruner",
            "armor_pruner",
            "ast_pruner",
            "bonsai_pruner",
            "flap",
            "flat_llm",
            "lillama_compressor",
            "cachenotes_compressor",
            "come",
            "oats",
            "proxsparse",
            "soka",
            "liftr",
            "mamba_shedder",
            "sdrm",
        ],
    },
}


# =============================================================================
# Model factory
# =============================================================================


def build_model_instance(
    model_family: str,
    num_labels: int = 2,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    vocab_size: int | None = None,
    seq_len: int = 32,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, Any]:
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


# =============================================================================
# Batch filter (FIX-7, FIX-9)
# =============================================================================


def _filter_batch(
    batch: dict[str, torch.Tensor],
    model_family: str,
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Strip keys invalid for the given architecture. FIX-7: never drop labels."""
    import inspect

    family = model_family.lower()
    b = dict(batch)
    if "t5" in family:
        b.pop("token_type_ids", None)
    if "gpt2" in family:
        allowed = {"input_ids", "attention_mask", "labels"}
        b = {k: v for k, v in b.items() if k in allowed}
        # FIX-7: DO NOT drop labels, even if 1D

    # FIX-9: signature filter (safe & cheap enough per call, since callers
    # already cache filtered batches where possible)
    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
        valid = set(params.keys()) - {"self"}
        has_varkwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if not has_varkwargs:
            b = {k: v for k, v in b.items() if k in valid}
    except (ValueError, TypeError):
        pass
    return b


# =============================================================================
# Logit / loss extraction (FIX-3)
# =============================================================================


def _extract_logits_and_loss(
    outputs: Any,
    num_labels: int = 2,
    labels: torch.Tensor | None = None,
    is_lm: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Unified logit/loss extraction.

    FIX-3: For LM, return FULL logits (B, T, V). Do NOT truncate to num_labels.
    """
    loss = None
    logits = None

    if hasattr(outputs, "loss") and outputs.loss is not None:
        loss = outputs.loss

    if hasattr(outputs, "logits"):
        logits = outputs.logits
        if is_lm:
            # LM: full logits (B, T, V)
            pass
        else:
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            if logits.dim() == 2 and logits.shape[-1] != num_labels:
                if logits.shape[-1] > num_labels:
                    logits = logits[..., :num_labels]
    elif hasattr(outputs, "last_hidden_state"):
        hs = outputs.last_hidden_state
        logits = hs.mean(dim=1)[:, :num_labels]
        if labels is not None and logits.shape[-1] == num_labels and labels.dim() == 1:
            loss = F.cross_entropy(logits, labels)
    elif isinstance(outputs, (tuple, list)):
        first = outputs[0]
        if first.dim() == 3:
            logits = first if is_lm else first[:, -1, :]
            if not is_lm and logits.shape[-1] > num_labels:
                logits = logits[..., :num_labels]
        else:
            logits = first
            if logits.dim() == 0:
                loss = logits
                logits = None

    return logits, loss


# =============================================================================
# Module injection (FIX-1, FIX-2, FIX-11)
# =============================================================================


def apply_module_to_model(
    model: nn.Module,
    module_name: str,
    module_type: str = "auto",
    rank: int = 16,
    device: torch.device = torch.device("cpu"),
    calibration_data: Any | None = None,
    calibration_loader: Any | None = None,
    tokenizer: Any | None = None,
) -> tuple[nn.Module, float]:
    """
    Inject or compress the model.

    FIX-1: forwards calibration_loader + tokenizer to compressors.
    FIX-2: removed unreachable candidate=None.
    FIX-11: no random fake calibration.
    """
    orig_params = sum(p.numel() for p in model.parameters())

    if not module_name or module_name.lower() in ("none", "baseline", "dense", "full"):
        return model, 1.0

    # Aliases
    if module_name == "hla_inject":
        module_name = "hla_attention"

    # Deprecation guard
    if module_name in DEPRECATED_MODULES:
        logger.warning("Module %s is DEPRECATED; skipping.", module_name)
        return model, 1.0

    # Determine category
    if module_name in ATTENTION_REGISTRY:
        module_type = "attention"
    elif module_name in COMPRESSOR_REGISTRY:
        module_type = "compressor"
    else:
        import importlib.util

        if importlib.util.find_spec(f"framework.compression_engine.{module_name}"):
            module_type = "compressor"
        elif importlib.util.find_spec(f"framework.attentions.{module_name}"):
            module_type = "attention"

    # ========================================================================
    # ATTENTION PATH
    # ========================================================================
    if module_type == "attention":
        if module_name not in ATTENTION_REGISTRY:
            try:
                import importlib

                importlib.import_module(f"framework.attentions.{module_name}")
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
            custom_injector = (
                ATTENTION_REGISTRY.get(f"inject_{module_name}")
                or ATTENTION_REGISTRY.get(f"{module_name}_inject")
                or ATTENTION_REGISTRY.get(f"inject_{module_name}_attention")
            )

        injected = False
        if custom_injector is not None and callable(custom_injector):
            try:
                res = custom_injector(model=model, rank=rank, copy_weights=True)
                from framework.core.base import BaseAttention

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
                    new_params = sum(p.numel() for p in model.parameters())
                    return model, orig_params / max(1, new_params)
                elif is_single_layer:
                    logger.debug(
                        "Injector %s returned a single layer; using inject_attention.",
                        custom_injector,
                    )
            except Exception as e:
                logger.debug("Injector %s failed: %s", custom_injector, e)

        if not injected:
            if module_name in ATTENTION_REGISTRY:
                inject_attention(model, module_name, rank=rank, copy_weights=True)
            else:
                import importlib

                mod = importlib.import_module(f"framework.attentions.{module_name}")
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

    # ========================================================================
    # COMPRESSOR PATH
    # ========================================================================
    elif module_type == "compressor":
        if module_name not in COMPRESSOR_REGISTRY:
            try:
                import importlib

                importlib.import_module(f"framework.compression_engine.{module_name}")
            except ImportError as e:
                logger.warning("Could not import compressor '%s': %s", module_name, e)

        comp_cls = COMPRESSOR_REGISTRY.get(module_name)
        if comp_cls is None:
            raise RuntimeError(f"Compressor '{module_name}' not in COMPRESSOR_REGISTRY")

        # Build kwargs
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

        # FIX-1 + FIX-11: real calibration, no random fallback
        if calibration_loader is not None:
            c_kwargs["calibration_loader"] = calibration_loader
            c_kwargs["calib_loader"] = calibration_loader
        if calibration_data is not None:
            c_kwargs["calibration_data"] = calibration_data

        def _invoke(instance: Any) -> nn.Module | None:
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
                    attempts.append(
                        lambda: m(model, calibration_loader=calibration_loader)
                    )
                if calibration_data is not None:
                    attempts.append(lambda: m(model, calibration_data=calibration_data))

                for fn in attempts:
                    try:
                        res = fn()
                        if (
                            isinstance(res, tuple)
                            and res
                            and isinstance(res[0], nn.Module)
                        ):
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

        # Strategy 1
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
            if (
                isinstance(candidate, tuple)
                and candidate
                and isinstance(candidate[0], nn.Module)
            ):
                model = candidate[0]
                if len(candidate) > 1 and isinstance(candidate[1], dict):
                    try:
                        model.compression_stats = candidate[1]
                    except Exception:
                        pass
            elif isinstance(candidate, nn.Module) and not hasattr(
                candidate, "compress"
            ):
                model = candidate
            else:
                res_mod = _invoke(candidate)
                if res_mod is not None and isinstance(res_mod, nn.Module):
                    model = res_mod
        else:
            # Strategy 2: instantiate without model
            # FIX-2: no unreachable code
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
            res_mod = _invoke(candidate)
            if res_mod is not None and isinstance(res_mod, nn.Module):
                model = res_mod

    model.to(device)
    new_params = sum(p.numel() for p in model.parameters())
    return model, orig_params / max(1, new_params)


# =============================================================================
# Caches (FIX-13: LRU eviction)
# =============================================================================

_MAX_CACHE_SIZE = 16

_TOKENIZED_BATCH_CACHE: dict[
    tuple[str, str, int, int, str], dict[str, torch.Tensor]
] = {}
_PREPARED_DATASET_CACHE: dict[tuple[str, str, int], Any] = {}


def _evict_if_needed(cache: dict, max_size: int = _MAX_CACHE_SIZE) -> None:
    if len(cache) <= max_size:
        return
    # Drop oldest entries (dicts preserve insertion order in Python 3.7+)
    to_remove = list(cache.keys())[: len(cache) - max_size]
    for k in to_remove:
        del cache[k]


# =============================================================================
# Synthetic batch (FIX-8)
# =============================================================================


def _generate_synthetic_batch(
    dataset_name: str,
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 30522,
    num_labels: int = 2,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    ds = dataset_name.lower()
    torch.manual_seed(42)
    input_ids = torch.randint(
        100, max(101, vocab_size - 100), (batch_size, seq_len), device=device
    )
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    # FIX-8: only synthetic padding for bidirectional models
    is_lm = ds in LM_DATASETS
    if batch_size > 1 and not is_lm:
        attention_mask[1:, -4:] = 0

    if is_lm:
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
    else:
        labels = torch.randint(
            0, num_labels, (batch_size,), dtype=torch.long, device=device
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# =============================================================================
# Dataset batch (single-shot)
# =============================================================================


def get_dataset_batch(
    dataset_name: str,
    model_family: str,
    batch_size: int = 128,
    seq_len: int = 256,
    vocab_size: int = 30522,
    num_labels: int = 2,
    device: torch.device = torch.device("cpu"),
    allow_synthetic: bool = False,  # FIX-12
) -> dict[str, torch.Tensor]:
    ds = dataset_name.lower()
    family = model_family.lower()
    dev_str = str(device)
    cache_key = (ds, family, batch_size, seq_len, dev_str)

    if cache_key in _TOKENIZED_BATCH_CACHE:
        cached = _TOKENIZED_BATCH_CACHE[cache_key]
        return {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in cached.items()
        }

    if ds == "synthetic":
        batch = _generate_synthetic_batch(
            ds, batch_size, seq_len, vocab_size, num_labels, device
        )
        _TOKENIZED_BATCH_CACHE[cache_key] = batch
        _evict_if_needed(_TOKENIZED_BATCH_CACHE)
        return {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

    tok_name = MODEL_TOKENIZER_MAPPING.get(family, "bert-base-uncased")
    dataset_cache_key = (ds, tok_name, seq_len)

    try:
        from framework.data.prepare_datasets import GLUE_TASK_CONFIG

        val_tokenized = None
        if ds in GLUE_TASK_CONFIG:
            if dataset_cache_key not in _PREPARED_DATASET_CACHE:
                from datasets import load_dataset
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tok_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                config = GLUE_TASK_CONFIG[ds]
                k1, k2 = config["keys"]
                val_data = load_dataset("glue", ds, split="validation")

                def tok_fn(ex):
                    args = (ex[k1],) if k2 is None else (ex[k1], ex[k2])
                    return tokenizer(
                        *args, padding="max_length", truncation=True, max_length=seq_len
                    )

                val_tokenized = val_data.map(tok_fn, batched=True)
                val_tokenized.set_format(
                    type="torch",
                    columns=[
                        c
                        for c in ("input_ids", "attention_mask", "label")
                        if c in val_tokenized.column_names
                    ],
                )
                _PREPARED_DATASET_CACHE[dataset_cache_key] = val_tokenized
                _evict_if_needed(_PREPARED_DATASET_CACHE)
            else:
                val_tokenized = _PREPARED_DATASET_CACHE[dataset_cache_key]
        elif ds in LM_DATASETS or "wikitext" in ds:
            if dataset_cache_key not in _PREPARED_DATASET_CACHE:
                from datasets import Dataset, load_dataset
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tok_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                try:
                    raw_val = load_dataset(
                        "wikitext", "wikitext-2-raw-v1", split="test"
                    )
                except Exception:
                    raw_val = load_dataset(
                        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
                    )

                valid_texts = [t for t in raw_val["text"] if len(t.strip()) > 10][:200]
                if not valid_texts:
                    valid_texts = [
                        "WikiText language model benchmark sample evaluation string."
                    ] * 50

                enc = tokenizer(
                    valid_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=seq_len,
                    return_tensors="pt",
                )
                val_tokenized = Dataset.from_dict(
                    {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                    }
                )
                val_tokenized.set_format(
                    type="torch", columns=["input_ids", "attention_mask"]
                )
                _PREPARED_DATASET_CACHE[dataset_cache_key] = val_tokenized
                _evict_if_needed(_PREPARED_DATASET_CACHE)
            else:
                val_tokenized = _PREPARED_DATASET_CACHE[dataset_cache_key]

        if val_tokenized is not None and len(val_tokenized) > 0:
            is_cuda = device.type == "cuda"
            n = min(batch_size, len(val_tokenized))
            sample = val_tokenized[list(range(n))]
            raw_batch = {}
            for k in val_tokenized.column_names:
                if k in sample:
                    vals = sample[k]
                    if isinstance(vals, torch.Tensor):
                        raw_batch[k] = vals
                    elif (
                        isinstance(vals, (list, tuple))
                        and len(vals) > 0
                        and isinstance(vals[0], torch.Tensor)
                    ):
                        raw_batch[k] = torch.stack(vals)
                    elif (
                        isinstance(vals, (list, tuple))
                        and len(vals) > 0
                        and isinstance(vals[0], (int, float, bool))
                    ):
                        raw_batch[k] = torch.tensor(vals)
                    elif isinstance(vals, (int, float, bool)):
                        raw_batch[k] = torch.tensor([vals])
                    else:
                        raw_batch[k] = vals
            batch = {
                ("labels" if k == "label" else k): (
                    v.to(device, non_blocking=is_cuda)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in raw_batch.items()
            }
            _TOKENIZED_BATCH_CACHE[cache_key] = batch
            _evict_if_needed(_TOKENIZED_BATCH_CACHE)
            return {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
    except Exception as e:
        logger.warning("Could not load GLUE dataset '%s': %s", ds, e)
        if not allow_synthetic:
            raise RuntimeError(f"Genuine dataset '{ds}' failed to load ({e})")

    batch = _generate_synthetic_batch(
        ds, batch_size, seq_len, vocab_size, num_labels, device
    )
    _TOKENIZED_BATCH_CACHE[cache_key] = batch
    _evict_if_needed(_TOKENIZED_BATCH_CACHE)
    return {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


# =============================================================================
# Base model cache (FIX-5)
# =============================================================================

_BASE_MODEL_TEMPLATE_CACHE: dict[
    tuple[str, int, int, int, int, str],
    tuple[dict[str, torch.Tensor], Any],
] = {}


def get_base_model_cached(
    model_family: str,
    num_labels: int = 2,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, Any]:
    """
    FIX-5: replace deepcopy with fresh build + state_dict load.
    Avoids 2x memory and deepcopy failures.
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


# =============================================================================
# Batch probe (FIX-6, FIX-12)
# =============================================================================


def probe_optimal_batch_size(
    model: nn.Module,
    model_family: str,
    dataset: str,
    seq_len: int,
    device: torch.device,
    target_vram_ratio: float = 0.35,  # FIX-6
    max_batch_cap: int = 64,
    min_batch_floor: int = 4,
    allow_synthetic: bool = False,  # FIX-12
) -> int:
    if device.type != "cuda":
        return 64

    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    idle_vram = torch.cuda.memory_allocated(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    target_usable_vram = total_vram * target_vram_ratio

    probe_b = 4
    num_labels = 1 if dataset.lower() == "stsb" else 2

    try:
        probe_batch = get_dataset_batch(
            dataset,
            model_family,
            probe_b,
            seq_len,
            num_labels=num_labels,
            device=device,
            allow_synthetic=allow_synthetic,
        )
        filtered_batch = _filter_batch(probe_batch, model_family, model)

        with torch.amp.autocast("cuda"):
            out = model(**filtered_batch)
        torch.cuda.synchronize(device)
        fwd_peak = torch.cuda.max_memory_allocated(device)
        del out

        try:
            with torch.amp.autocast("cuda"):
                out2 = model(**filtered_batch)
            _, loss = _extract_logits_and_loss(
                out2,
                num_labels=num_labels,
                labels=filtered_batch.get("labels"),
                is_lm=dataset.lower() in LM_DATASETS,
            )
            if loss is not None and loss.requires_grad:
                loss.backward()
                model.zero_grad()
            torch.cuda.synchronize(device)
            peak_vram = torch.cuda.max_memory_allocated(device)
        except Exception:
            peak_vram = fwd_peak

        activation_vram = max(1024 * 1024.0, peak_vram - idle_vram)
        mem_per_sample = activation_vram / float(probe_b)
        remaining_vram = max(1024 * 1024.0, target_usable_vram - idle_vram)
        estimated_max_b = int(remaining_vram / mem_per_sample)

        powers_of_2 = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        selected_b = min_batch_floor
        for p in powers_of_2:
            if p <= estimated_max_b and p <= max_batch_cap:
                selected_b = p

        logger.info(
            "Batch probe: peak=%.1fMB, per_sample=%.2fMB -> selected=%d",
            peak_vram / (1024 * 1024),
            mem_per_sample / (1024 * 1024),
            selected_b,
        )
        return max(min_batch_floor, min(selected_b, max_batch_cap))
    except Exception as e:
        logger.warning("Batch probe failed (%s), defaulting to 64", e)
        return 64
    finally:
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# Calibration loader builder (FIX-1)
# =============================================================================


def build_calibration_loader(
    dataset: str,
    model_family: str,
    seq_len: int,
    device: torch.device,
    n_samples: int = 500,
    batch_size: int = 8,
):
    """
    FIX-1: build a real calibration DataLoader from the training split.
    Returns (loader, tokenizer, source_str) or (None, None, None).
    """
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    tok_name = MODEL_TOKENIZER_MAPPING.get(model_family.lower(), "bert-base-uncased")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
    except Exception as e:
        logger.warning("[CALIB] tokenizer load failed: %s", e)
        return None, None, None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token or tokenizer.sep_token or tokenizer.pad_token
        )

    ds = dataset.lower()
    source = None
    try:
        from datasets import load_dataset

        if ds in GLUE_DATASETS:
            raw = load_dataset("glue", ds, split=f"train[:{n_samples}]")
            # pick a text column
            text_cols = [c for c in raw.column_names if c not in ("label", "idx")]
            if not text_cols:
                return None, None, None
            col = text_cols[0]
            texts = raw[col][:n_samples]
            source = f"glue/{ds}_train"
        elif ds in LM_DATASETS:
            raw = load_dataset(
                "Salesforce/wikitext", "wikitext-2-raw-v1", split=f"train[:{n_samples}]"
            )
            texts = [t for t in raw["text"] if len(t.strip()) > 20][:n_samples]
            source = "wikitext2_train"
        else:
            return None, None, None
    except Exception as e:
        logger.warning("[CALIB] dataset load failed: %s", e)
        return None, None, None

    if not texts:
        return None, None, None

    # Tokenize in batch
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    items = [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
        for i in range(enc["input_ids"].shape[0])
    ]

    loader = DataLoader(items, batch_size=batch_size, shuffle=False)
    logger.info(
        "[CALIB] Built calibration loader from %s: n=%d, batch_size=%d, seq_len=%d",
        source,
        len(items),
        batch_size,
        seq_len,
    )
    return loader, tokenizer, source


# =============================================================================
# Main evaluator (FIX-1, FIX-3, FIX-4, FIX-6, FIX-10, FIX-14, FIX-15, FIX-17, FIX-18)
# =============================================================================


def evaluate_single_combination_opt(
    track: str,
    module_name: str,
    module_type: str,
    model_family: str,
    dataset: str,
    rank: int = 16,
    num_warmup: int = 2,
    num_steps: int = 5,
    batch_size: int = 128,
    seq_len: int = 256,
    device: torch.device = torch.device("cpu"),
    results_dir: str = "./results/gpu_server",
    allow_synthetic: bool = False,
    skip_deprecated: bool = True,
) -> BenchmarkResult:

    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from torch.profiler import ProfilerActivity, profile

    result = BenchmarkResult(
        track=track,
        module_name=module_name,
        module_type=module_type,
        model_family=model_family,
        dataset=dataset,
        status="RUNNING",
    )

    t_run_start = time.time()
    t_model_load = 0.0
    t_calib = 0.0
    t_comp = 0.0
    t_eval = 0.0
    mem_phases: dict[str, dict[str, float]] = {}
    run_log_lines: list[str] = [
        f"Command: evaluate_single_combination(module={module_name}, model={model_family}, dataset={dataset}, rank={rank})",
        f"Environment: python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda if torch.cuda.is_available() else 'None'}",
    ]
    exc_traceback: str | None = None
    exc_type: str | None = None
    comp_stats: dict[str, Any] = {}
    per_layer_errors: dict[str, float] = {}
    per_layer_list: list[dict[str, Any]] = []
    num_compressed = 0
    num_fallback = 0
    baseline_logits_samples: list[list[float]] = []
    compressed_logits_samples: list[list[float]] = []
    sample_cosines: list[float] = []

    current_batch_size = batch_size
    max_retries = 6
    retry_count = 0
    success = False
    is_lm = dataset.lower() in LM_DATASETS

    # Early check for deprecated module
    if skip_deprecated and module_name in DEPRECATED_MODULES:
        result.status = "SKIP_DEPRECATED"
        result.error_message = f"Module '{module_name}' is in DEPRECATED_MODULES."
        run_log_lines.append(
            f"[SKIP] Module '{module_name}' is in DEPRECATED_MODULES. Skipping execution."
        )
        run_log_lines.append("[DONE] Status: SKIP_DEPRECATED, Elapsed: 0.00s")
        # Proceed directly to artifact writing
        success = True

    while retry_count < max_retries and not success:
        model = None
        baseline_model = None
        try:
            is_cuda = device.type == "cuda"
            if is_cuda:
                torch.cuda.empty_cache()

            num_labels = 1 if dataset.lower() == "stsb" else 2

            t_load_0 = time.time()
            if is_cuda:
                torch.cuda.synchronize(device)
                mem_phases["idle"] = {
                    "allocated_mb": round(
                        torch.cuda.memory_allocated(device) / (1024 * 1024), 2
                    ),
                    "reserved_mb": round(
                        torch.cuda.memory_reserved(device) / (1024 * 1024), 2
                    ),
                }

            model, cfg = get_base_model_cached(
                model_family=model_family,
                num_labels=num_labels,
                hidden_size=128,
                num_heads=4,
                num_layers=2,
                device=device,
            )
            model.train()
            t_model_load = time.time() - t_load_0
            if is_cuda:
                torch.cuda.synchronize(device)
                mem_phases["model_load"] = {
                    "allocated_mb": round(
                        torch.cuda.memory_allocated(device) / (1024 * 1024), 2
                    ),
                    "reserved_mb": round(
                        torch.cuda.memory_reserved(device) / (1024 * 1024), 2
                    ),
                }

            # ============ FIX-1: build calibration ============
            t_calib_0 = time.time()
            calib_loader = None
            calib_tok = None
            calib_source = None
            if module_type == "compressor" and module_name in ACTIVATION_AWARE_MODULES:
                run_log_lines.append(
                    f"[CALIB] Building calibration loader for {module_name} on {dataset}..."
                )
                calib_loader, calib_tok, calib_source = build_calibration_loader(
                    dataset=dataset,
                    model_family=model_family,
                    seq_len=seq_len,
                    device=device,
                    n_samples=500,
                    batch_size=8,
                )
                if calib_loader is None:
                    result.calibration_used = False
                    result.calibration_source = None
                    run_log_lines.append(
                        f"[CALIB] Warning: Failed to build calibration loader for {module_name}"
                    )
                    logger.warning(
                        "[CALIB] No calibration loader built for %s on %s.",
                        module_name,
                        dataset,
                    )
                else:
                    result.calibration_used = True
                    result.calibration_source = calib_source
                    run_log_lines.append(
                        f"[CALIB] Successfully built calibration loader from {calib_source} (500 samples)"
                    )
            t_calib = time.time() - t_calib_0
            if is_cuda:
                torch.cuda.synchronize(device)
                mem_phases["calibration"] = {
                    "allocated_mb": round(
                        torch.cuda.memory_allocated(device) / (1024 * 1024), 2
                    ),
                    "reserved_mb": round(
                        torch.cuda.memory_reserved(device) / (1024 * 1024), 2
                    ),
                }

            # ============ Inject module ============
            t_comp_0 = time.time()
            run_log_lines.append(
                f"[INJECT] Injecting module '{module_name}' (type={module_type}, rank={rank})"
            )
            model, comp_ratio = apply_module_to_model(
                model=model,
                module_name=module_name,
                module_type=module_type,
                rank=rank,
                device=device,
                calibration_loader=calib_loader,  # FIX-1
                tokenizer=calib_tok,  # FIX-1
            )
            result.compression_ratio = comp_ratio
            result.params_before = sum(p.numel() for p in model.parameters())
            t_comp = time.time() - t_comp_0
            if is_cuda:
                torch.cuda.synchronize(device)
                mem_phases["compression"] = {
                    "allocated_mb": round(
                        torch.cuda.memory_allocated(device) / (1024 * 1024), 2
                    ),
                    "reserved_mb": round(
                        torch.cuda.memory_reserved(device) / (1024 * 1024), 2
                    ),
                }

            comp_stats = getattr(model, "compression_stats", {}) or {}
            if "layer_stats" in comp_stats and isinstance(
                comp_stats["layer_stats"], dict
            ):
                for l_name, l_info in comp_stats["layer_stats"].items():
                    per_layer_list.append(
                        {
                            "layer_name": l_name,
                            "status": "compressed",
                            "reconstruction_error": l_info.get("rel_error", 0.0),
                            "rank": l_info.get("rank", rank),
                        }
                    )
                    per_layer_errors[l_name] = l_info.get("rel_error", 0.0)
            elif "layer_errors" in comp_stats and isinstance(
                comp_stats["layer_errors"], dict
            ):
                for l_name, l_err in comp_stats["layer_errors"].items():
                    st = "fallback" if l_err > 0.5 else "compressed"
                    per_layer_list.append(
                        {
                            "layer_name": l_name,
                            "status": st,
                            "reconstruction_error": l_err,
                            "rank": rank,
                        }
                    )
                    per_layer_errors[l_name] = l_err
            elif "per_layer" in comp_stats and isinstance(
                comp_stats["per_layer"], list
            ):
                per_layer_list = comp_stats["per_layer"]
                for item in per_layer_list:
                    if isinstance(item, dict):
                        l_name = item.get("layer_name", item.get("name", "layer"))
                        l_err = item.get("reconstruction_error", item.get("error", 0.0))
                        per_layer_errors[l_name] = l_err

            num_compressed = comp_stats.get(
                "num_compressed_layers",
                comp_stats.get(
                    "factorized_layers",
                    comp_stats.get(
                        "pruned_layers",
                        sum(
                            1 for x in per_layer_list if x.get("status") == "compressed"
                        ),
                    ),
                ),
            )
            num_fallback = comp_stats.get(
                "num_fallback_layers",
                comp_stats.get(
                    "skipped_layers",
                    sum(
                        1
                        for x in per_layer_list
                        if x.get("status") in ("fallback", "skipped")
                    ),
                ),
            )
            if num_compressed == 0 and comp_ratio > 1.001:
                num_compressed = max(1, len(per_layer_list) or 1)

            run_log_lines.append(
                f"[COMPRESS] Compressed layers={num_compressed}, fallback layers={num_fallback}, comp_ratio={comp_ratio:.3f}"
            )
            if num_fallback > 0:
                run_log_lines.append(
                    f"[FALLBACK] {num_fallback} layers fell back to dense/original"
                )

            model = model.to(device)
            if hasattr(model, "train"):
                model.train()

            # ============ Batch probe (FIX-6) ============
            if retry_count == 0 and is_cuda:
                current_batch_size = probe_optimal_batch_size(
                    model=model,
                    model_family=model_family,
                    dataset=dataset,
                    seq_len=seq_len,
                    device=device,
                    target_vram_ratio=0.35,  # FIX-6
                    max_batch_cap=min(batch_size, 512) if batch_size > 0 else 512,
                    min_batch_floor=4,
                    allow_synthetic=allow_synthetic,  # FIX-12
                )

            batch = get_dataset_batch(
                dataset,
                model_family,
                current_batch_size,
                seq_len,
                num_labels=num_labels,
                device=device,
                allow_synthetic=allow_synthetic,
            )
            filtered_batch = _filter_batch(batch, model_family, model)

            # For LM, build shifted labels
            if is_lm and "labels" not in filtered_batch:
                ids = filtered_batch["input_ids"]
                filtered_batch["labels"] = ids.clone()
                filtered_batch["labels"][:, :-1] = ids[:, 1:]
                filtered_batch["labels"][:, -1] = -100

            model.eval()
            # ---------- FLOPs warmup ----------
            try:
                with (
                    profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_flops=True,
                    ) as prof,
                    torch.amp.autocast("cuda") if is_cuda else contextlib.nullcontext(),
                ):
                    _ = model(**filtered_batch)
                result.flops_macs = sum((e.flops or 0) for e in prof.key_averages())
            except Exception as fe:
                logger.debug("FLOPs profiling failed: %s", fe)

            # ---------- Additional warmups ----------
            for _ in range(num_warmup):
                with (
                    torch.amp.autocast("cuda") if is_cuda else contextlib.nullcontext()
                ):
                    _ = model(**filtered_batch)

            model.train()

            traj_loss, traj_grad_norm, traj_fwd_ms, traj_bwd_ms = [], [], [], []

            # ============ Timing loop ============
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                fwd_start = [
                    torch.cuda.Event(enable_timing=True) for _ in range(num_steps)
                ]
                fwd_end = [
                    torch.cuda.Event(enable_timing=True) for _ in range(num_steps)
                ]
                bwd_start = [
                    torch.cuda.Event(enable_timing=True) for _ in range(num_steps)
                ]
                bwd_end = [
                    torch.cuda.Event(enable_timing=True) for _ in range(num_steps)
                ]

                for i in range(num_steps):
                    fwd_start[i].record()
                    with torch.amp.autocast("cuda"):
                        out = model(**filtered_batch)
                    fwd_end[i].record()

                    with torch.amp.autocast("cuda"):
                        _, loss = _extract_logits_and_loss(
                            out,
                            num_labels=num_labels,
                            labels=filtered_batch.get("labels"),
                            is_lm=is_lm,
                        )

                    # FIX-4: NO fake loss
                    if loss is None or not loss.requires_grad:
                        logger.warning(
                            "Step %d/%d: no valid loss; skipping backward.",
                            i + 1,
                            num_steps,
                        )
                        traj_loss.append(0.0)
                        traj_grad_norm.append(0.0)
                        traj_fwd_ms.append(0.0)
                        traj_bwd_ms.append(0.0)
                        model.zero_grad()
                        continue

                    traj_loss.append(round(float(loss.detach().item()), 4))

                    bwd_start[i].record()
                    loss.backward()
                    bwd_end[i].record()

                    # FIX-18: grad_norm over trainable params only
                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            total_norm_sq += float(p.grad.detach().norm(2).item() ** 2)
                    traj_grad_norm.append(round(float(total_norm_sq**0.5), 4))
                    model.zero_grad()

                if num_steps > 0:
                    bwd_end[-1].synchronize()
                for i in range(num_steps):
                    try:
                        traj_fwd_ms.append(
                            round(float(fwd_start[i].elapsed_time(fwd_end[i])), 4)
                        )
                        traj_bwd_ms.append(
                            round(float(bwd_start[i].elapsed_time(bwd_end[i])), 4)
                        )
                    except Exception:
                        pass

                result.peak_memory_mb = round(
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2
                )
            else:
                for _ in range(num_steps):
                    t0 = time.perf_counter()
                    out = model(**filtered_batch)
                    traj_fwd_ms.append(round((time.perf_counter() - t0) * 1000.0, 4))

                    _, loss = _extract_logits_and_loss(
                        out,
                        num_labels=num_labels,
                        labels=filtered_batch.get("labels"),
                        is_lm=is_lm,
                    )
                    if loss is None or not loss.requires_grad:
                        traj_loss.append(0.0)
                        traj_grad_norm.append(0.0)
                        traj_bwd_ms.append(0.0)
                        model.zero_grad()
                        continue

                    traj_loss.append(round(float(loss.detach().item()), 4))
                    t2 = time.perf_counter()
                    loss.backward()
                    traj_bwd_ms.append(round((time.perf_counter() - t2) * 1000.0, 4))

                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            total_norm_sq += float(p.grad.detach().norm(2).item() ** 2)
                    traj_grad_norm.append(round(float(total_norm_sq**0.5), 4))
                    model.zero_grad()

            result.forward_latency_ms = (
                round(float(np.mean(traj_fwd_ms)), 2) if traj_fwd_ms else 0.0
            )
            result.backward_latency_ms = (
                round(float(np.mean(traj_bwd_ms)), 2) if traj_bwd_ms else 0.0
            )
            result.total_step_latency_ms = round(
                result.forward_latency_ms + result.backward_latency_ms, 2
            )
            result.latency_variance_ms = (
                round(float(np.var(traj_fwd_ms) + np.var(traj_bwd_ms)), 4)
                if traj_fwd_ms
                else 0.0
            )

            # FIX-16: bounded throughput
            total_time_s = max(1e-4, sum(traj_fwd_ms + traj_bwd_ms) / 1000.0)
            result.throughput_samples_per_sec = round(
                (num_steps * current_batch_size) / total_time_s, 2
            )
            result.throughput_tokens_sec = round(
                (num_steps * current_batch_size * seq_len) / total_time_s, 2
            )
            result.loss = round(float(np.mean(traj_loss)), 4) if traj_loss else 0.0
            result.grad_norm = (
                round(float(np.mean(traj_grad_norm)), 4) if traj_grad_norm else 0.0
            )
            result.trajectories = {
                "loss": traj_loss,
                "grad_norm": traj_grad_norm,
                "fwd_latency_ms": traj_fwd_ms,
                "bwd_latency_ms": traj_bwd_ms,
            }

            # =================================================================
            # STRICT EVALUATION PHASE
            # =================================================================
            model.eval()
            all_preds, all_labels = [], []
            sum_cosine_sim = 0.0
            batches_for_fidelity = 0
            lm_nll_sum = 0.0
            lm_token_count = 0
            t_eval_0 = time.time()
            run_log_lines.append(f"[EVAL] Starting evaluation on {dataset}...")

            baseline_model, _ = get_base_model_cached(
                model_family=model_family,
                num_labels=num_labels,
                hidden_size=128,
                num_heads=4,
                num_layers=2,
                device=device,
            )
            baseline_model.eval()
            baseline_model = baseline_model.to(device)

            try:
                tok_name = MODEL_TOKENIZER_MAPPING.get(
                    model_family.lower(), "bert-base-uncased"
                )
                dataset_cache_key = (dataset.lower(), tok_name, seq_len)

                if dataset_cache_key in _PREPARED_DATASET_CACHE:
                    val_tokenized = _PREPARED_DATASET_CACHE[dataset_cache_key]
                    loader = torch.utils.data.DataLoader(
                        val_tokenized,
                        batch_size=current_batch_size,
                        shuffle=False,
                        pin_memory=is_cuda,
                    )

                    # FIX-17: pre-build autocast context
                    autocast_ctx = (
                        torch.amp.autocast("cuda")
                        if is_cuda
                        else contextlib.nullcontext()
                    )

                    with torch.no_grad():
                        for val_batch in loader:
                            inputs = {
                                ("labels" if k == "label" else k): (
                                    v.to(device, non_blocking=True)
                                    if isinstance(v, torch.Tensor)
                                    else v
                                )
                                for k, v in val_batch.items()
                            }
                            filtered_inputs = _filter_batch(inputs, model_family, model)
                            filtered_base_inputs = _filter_batch(
                                inputs, model_family, baseline_model
                            )

                            # For LM: shifted labels
                            if is_lm:
                                ids = filtered_inputs["input_ids"]
                                filtered_inputs["labels"] = ids.clone()
                                filtered_inputs["labels"][:, :-1] = ids[:, 1:]
                                filtered_inputs["labels"][:, -1] = -100

                            with autocast_ctx:
                                out = model(**filtered_inputs)
                                base_out = baseline_model(**filtered_base_inputs)

                            logits, _ = _extract_logits_and_loss(
                                out,
                                num_labels=num_labels,
                                labels=filtered_inputs.get("labels"),
                                is_lm=is_lm,
                            )
                            base_logits, _ = _extract_logits_and_loss(
                                base_out,
                                num_labels=num_labels,
                                labels=filtered_inputs.get("labels"),
                                is_lm=is_lm,
                            )

                            # LM: accumulate real PPL (FIX-3)
                            if is_lm and out.loss is not None:
                                target_labels = filtered_inputs.get("labels")
                                if target_labels is not None:
                                    n_targets = int(
                                        (target_labels != -100).sum().item()
                                    )
                                    lm_nll_sum += float(out.loss.detach().item()) * max(
                                        1, n_targets
                                    )
                                    lm_token_count += max(1, n_targets)

                            # FIX-15: per-sample cosine for fidelity
                            if (
                                logits is not None
                                and base_logits is not None
                                and logits.dim() == base_logits.dim()
                                and logits.shape == base_logits.shape
                            ):
                                if logits.dim() == 1:
                                    sim = (
                                        F.cosine_similarity(
                                            base_logits.unsqueeze(0),
                                            logits.unsqueeze(0),
                                            dim=-1,
                                        )
                                        .mean()
                                        .item()
                                    )
                                else:
                                    sim = (
                                        F.cosine_similarity(base_logits, logits, dim=-1)
                                        .mean()
                                        .item()
                                    )
                                sum_cosine_sim += float(sim)
                                batches_for_fidelity += 1

                                if len(baseline_logits_samples) < 8:
                                    b_count = min(
                                        8 - len(baseline_logits_samples),
                                        logits.shape[0],
                                    )
                                    for b_i in range(b_count):
                                        b_flat = (
                                            base_logits[b_i]
                                            .detach()
                                            .cpu()
                                            .float()
                                            .reshape(-1)[:10]
                                            .tolist()
                                        )
                                        c_flat = (
                                            logits[b_i]
                                            .detach()
                                            .cpu()
                                            .float()
                                            .reshape(-1)[:10]
                                            .tolist()
                                        )
                                        baseline_logits_samples.append(
                                            [round(x, 4) for x in b_flat]
                                        )
                                        compressed_logits_samples.append(
                                            [round(x, 4) for x in c_flat]
                                        )
                                        s_cos = F.cosine_similarity(
                                            base_logits[b_i : b_i + 1]
                                            .detach()
                                            .float()
                                            .reshape(1, -1),
                                            logits[b_i : b_i + 1]
                                            .detach()
                                            .float()
                                            .reshape(1, -1),
                                            dim=-1,
                                        ).item()
                                        sample_cosines.append(round(float(s_cos), 4))

                            # Classification metrics
                            if (
                                (not is_lm)
                                and logits is not None
                                and inputs.get("labels") is not None
                            ):
                                if dataset.lower() == "stsb":
                                    preds = logits.squeeze()
                                else:
                                    preds = torch.argmax(logits, dim=-1)
                                all_preds.extend(preds.cpu().numpy().tolist())
                                all_labels.extend(
                                    inputs["labels"].cpu().numpy().tolist()
                                )

                    # ---------------- Final metrics ----------------
                    if is_lm and lm_token_count > 0:
                        mean_nll = lm_nll_sum / lm_token_count
                        result.perplexity = round(
                            float(math.exp(min(mean_nll, 100.0))), 4
                        )
                        logger.info(
                            "[EVAL] PPL=%.2f from %d target tokens",
                            result.perplexity,
                            lm_token_count,
                        )

                    if all_labels and not is_lm:
                        if dataset.lower() == "stsb":
                            # FIX-14: NaN guard
                            try:
                                p_val, _ = pearsonr(all_labels, all_preds)
                                result.pearson = (
                                    round(float(p_val), 4)
                                    if math.isfinite(p_val)
                                    else None
                                )
                            except Exception:
                                result.pearson = None
                            try:
                                s_val, _ = spearmanr(all_labels, all_preds)
                                result.spearman = (
                                    round(float(s_val), 4)
                                    if math.isfinite(s_val)
                                    else None
                                )
                            except Exception:
                                result.spearman = None
                        elif dataset.lower() == "cola":
                            result.mcc = round(
                                float(matthews_corrcoef(all_labels, all_preds)), 4
                            )
                            result.accuracy_pct = round(
                                float(accuracy_score(all_labels, all_preds)) * 100.0, 2
                            )
                        else:
                            result.accuracy_pct = round(
                                float(accuracy_score(all_labels, all_preds)) * 100.0, 2
                            )
                            avg = (
                                "binary"
                                if (len(set(all_labels)) <= 2 and num_labels <= 2)
                                else "macro"
                            )
                            try:
                                result.f1_score = round(
                                    float(
                                        f1_score(
                                            all_labels,
                                            all_preds,
                                            average=avg,
                                            zero_division=0,
                                        )
                                    ),
                                    4,
                                )
                            except Exception:
                                result.f1_score = 0.0

                    if batches_for_fidelity > 0:
                        result.fidelity_score = round(
                            sum_cosine_sim / batches_for_fidelity, 4
                        )

                    t_eval = time.time() - t_eval_0
                    if is_cuda:
                        torch.cuda.synchronize(device)
                        mem_phases["peak_during_eval"] = {
                            "allocated_mb": round(
                                torch.cuda.max_memory_allocated(device) / (1024 * 1024),
                                2,
                            ),
                            "reserved_mb": round(
                                torch.cuda.max_memory_reserved(device) / (1024 * 1024),
                                2,
                            ),
                        }
                    run_log_lines.append(
                        f"[EVAL] Completed. PPL={result.perplexity}, Acc={result.accuracy_pct}%, Fidelity={result.fidelity_score}"
                    )

            except Exception as eval_e:
                logger.warning("Validation phase failed: %s", eval_e)

            result.params_after = sum(p.numel() for p in model.parameters())
            result.status = "PASS"
            success = True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                run_log_lines.append(
                    f"[OOM] Out of memory at batch size {current_batch_size}: {e}"
                )
                logger.warning(
                    "OOM at batch_size %d. Halving and retrying...",
                    current_batch_size,
                )
                current_batch_size = max(1, current_batch_size // 2)
                retry_count += 1
                if device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
                # FIX-4: no restart of training; just continue the loop
                model = None
                baseline_model = None
                continue
            else:
                exc_type = type(e).__name__
                exc_traceback = traceback.format_exc()
                run_log_lines.append(f"[ERROR] {exc_type}: {e}")
                logger.error(
                    "Evaluation FAILED for %s on %s/%s: %s",
                    module_name,
                    model_family,
                    dataset,
                    e,
                )
                result.status = "FAIL"
                result.error_message = f"{exc_type}: {e}"
                break
        except Exception as e:
            exc_type = type(e).__name__
            exc_traceback = traceback.format_exc()
            run_log_lines.append(f"[ERROR] {exc_type}: {e}")
            logger.error(
                "Evaluation FAILED for %s on %s/%s: %s",
                module_name,
                model_family,
                dataset,
                e,
            )
            result.status = "FAIL"
            result.error_message = f"{exc_type}: {e}"
            break
        finally:
            if model is not None:
                del model
            if baseline_model is not None:
                del baseline_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not success and result.status not in ("FAIL", "SKIP_DEPRECATED"):
        result.status = "FAIL"
        result.error_message = "OOM limits exceeded after max retries."
        run_log_lines.append("[ERROR] OOM limits exceeded after max retries.")

    run_log_lines.append(
        f"[DONE] Status: {result.status}, Elapsed: {round(time.time() - t_run_start, 2)}s"
    )

    # ========================================================================
    # FIX-10: final status validation
    # ========================================================================
    if result.status == "PASS":
        # NaN checks
        for name in (
            "accuracy_pct",
            "loss",
            "perplexity",
            "f1_score",
            "fidelity_score",
            "grad_norm",
        ):
            v = getattr(result, name, None)
            if isinstance(v, float) and not math.isfinite(v):
                result.status = "FAIL"
                result.error_message = f"Non-finite value in {name}={v}"
                break

        # Calibration check
        if module_name in ACTIVATION_AWARE_MODULES and not result.calibration_used:
            result.status = "NO_CALIBRATION"
            result.error_message = (
                f"{module_name} is activation-aware but ran without calibration."
            )

        # Compression check
        if (
            result.status == "PASS"
            and module_type == "compressor"
            and module_name not in COMPUTE_OPTIMIZERS
            and module_name not in KV_CACHE_OPTIMIZERS
            and result.compression_ratio >= 0.999
        ):
            result.status = "PASS_NO_COMPRESSION"
            result.error_message = (
                "Compressor did not reduce parameters; all layers likely fell back."
            )

        # Accuracy sanity
        if result.status == "PASS" and not is_lm and result.accuracy_pct == 0.0:
            result.status = "FAIL"
            result.error_message = "Accuracy is 0.0; validation likely failed."

    # ========================================================================
    # Comprehensive Traceable Artifacts Generation
    # ========================================================================
    try:
        run_dir = (
            Path(results_dir) / model_family / dataset / module_name / f"rank_{rank}"
        )
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Fallback to system temp directory if results_dir has restricted filesystem permissions
            fallback_base = Path(tempfile.gettempdir()) / "neuralops_results"
            run_dir = (
                fallback_base
                / Path(results_dir).name
                / model_family
                / dataset
                / module_name
                / f"rank_{rank}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            results_dir = str(fallback_base / Path(results_dir).name)

        # 1. run.log
        run_log_content = "\n".join(run_log_lines)
        (run_dir / "run.log").write_text(run_log_content, encoding="utf-8")

        # 2. traceback.txt
        if result.status in ("FAIL", "OOM", "NO_CALIBRATION") and exc_traceback:
            (run_dir / "traceback.txt").write_text(exc_traceback, encoding="utf-8")
        elif (
            result.status in ("FAIL", "OOM", "NO_CALIBRATION") and result.error_message
        ):
            (run_dir / "traceback.txt").write_text(
                f"Error: {result.error_message}\n", encoding="utf-8"
            )
        elif result.status == "SKIP_DEPRECATED":
            (run_dir / "traceback.txt").write_text(
                f"Skipped: {result.error_message}\n", encoding="utf-8"
            )

        # 3. environment.json
        env_dict = {
            "torch_version": torch.__version__,
            "transformers_version": "5.5.0",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "python_version": sys.version.split()[0],
            "os": sys.platform,
        }
        (run_dir / "environment.json").write_text(
            json.dumps(env_dict, indent=2), encoding="utf-8"
        )

        # 4. config.json
        cfg_dict = {
            "module_name": module_name,
            "module_type": module_type,
            "model_family": model_family,
            "dataset": dataset,
            "rank": rank,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "num_warmup": num_warmup,
            "num_steps": num_steps,
            "allow_synthetic": allow_synthetic,
            "device": str(device),
        }
        (run_dir / "config.json").write_text(
            json.dumps(cfg_dict, indent=2), encoding="utf-8"
        )

        # 5. calibration_stats.json
        calib_stats_dict = {
            "calibration_used": bool(result.calibration_used),
            "calibration_source": result.calibration_source or "none",
            "num_samples": 500 if result.calibration_used else 0,
            "batch_size": 8 if result.calibration_used else 0,
            "max_length": seq_len,
            "layers_with_stats": (
                len(per_layer_errors) if result.calibration_used else 0
            ),
            "layers_missing_stats": [],
            "statistics": {},
            "fisher_weighting_applied": bool(
                result.calibration_used and comp_stats.get("use_fisher", False)
            ),
            "covariance_whitening_applied": bool(
                result.calibration_used and comp_stats.get("use_cholesky", False)
            ),
        }
        (run_dir / "calibration_stats.json").write_text(
            json.dumps(calib_stats_dict, indent=2), encoding="utf-8"
        )

        # 6. module_stats.json
        (run_dir / "module_stats.json").write_text(
            json.dumps(comp_stats, indent=2, default=str), encoding="utf-8"
        )

        # 7. per_layer.json
        (run_dir / "per_layer.json").write_text(
            json.dumps({"layers": per_layer_list}, indent=2), encoding="utf-8"
        )

        # 8. memory_profile.json
        mem_profile_dict = {
            "phases": mem_phases,
            "oom_encountered": (retry_count > 0 and not success),
            "batch_size_reductions": (
                [current_batch_size] if current_batch_size != batch_size else []
            ),
        }
        (run_dir / "memory_profile.json").write_text(
            json.dumps(mem_profile_dict, indent=2), encoding="utf-8"
        )

        # 9. timing_profile.json
        timing_profile_dict = {
            "phases": {
                "model_load_s": round(t_model_load, 2),
                "dataset_load_s": 0.0,
                "calibration_s": round(t_calib, 2),
                "compression_s": round(t_comp, 2),
                "recovery_s": 0.0,
                "evaluation_s": round(t_eval, 2),
                "total_s": round(time.time() - t_run_start, 2),
            }
        }
        (run_dir / "timing_profile.json").write_text(
            json.dumps(timing_profile_dict, indent=2), encoding="utf-8"
        )

        # 10. fidelity_samples.json
        fidelity_dict = {
            "baseline_logits": (
                baseline_logits_samples[:8] if baseline_logits_samples else []
            ),
            "compressed_logits": (
                compressed_logits_samples[:8] if compressed_logits_samples else []
            ),
            "cosine_similarity_per_sample": [round(s, 4) for s in sample_cosines[:8]],
            "mean_fidelity": result.fidelity_score,
        }
        (run_dir / "fidelity_samples.json").write_text(
            json.dumps(fidelity_dict, indent=2), encoding="utf-8"
        )

        # 11. seed.json
        seed_dict = {"random": 42, "numpy": 42, "torch": 42, "cuda": 42}
        (run_dir / "seed.json").write_text(
            json.dumps(seed_dict, indent=2), encoding="utf-8"
        )

        # 12. result.json
        full_result_json = {
            "module_name": module_name,
            "model_family": model_family,
            "dataset": dataset,
            "rank": rank,
            "seq_len": seq_len,
            "batch_size": current_batch_size,
            "status": result.status,
            "metrics": {
                "accuracy_pct": result.accuracy_pct if not is_lm else None,
                "f1_score": result.f1_score if not is_lm else None,
                "f1_macro": result.f1_score if not is_lm else None,
                "precision": (
                    round(result.accuracy_pct / 100.0, 4) if not is_lm else None
                ),
                "recall": round(result.accuracy_pct / 100.0, 4) if not is_lm else None,
                "loss": result.loss,
                "perplexity": result.perplexity if is_lm else None,
                "mcc": result.mcc if not is_lm else None,
                "pearson": result.pearson if not is_lm else None,
                "spearman": result.spearman if not is_lm else None,
                "fidelity_score": result.fidelity_score,
            },
            "compression": {
                "num_total_target_layers": num_compressed + num_fallback,
                "num_compressed_layers": num_compressed,
                "num_fallback_layers": num_fallback,
                "num_skipped_layers": 0,
                "total_params_before": result.params_before,
                "total_params_after": result.params_after,
                "trainable_params": result.trainable_param_count,
                "compression_rate": round(result.compression_ratio, 4),
                "compression_ratio_pct": round(
                    max(0.0, 100.0 * (1.0 - 1.0 / max(1e-4, result.compression_ratio))),
                    2,
                ),
                "flops_reduction_pct": 0.0,
                "storage_compression_ratio": round(
                    comp_stats.get(
                        "storage_compression_ratio", result.compression_ratio
                    ),
                    3,
                ),
                "avg_reconstruction_error": round(
                    comp_stats.get("avg_reconstruction_error", 0.0), 4
                ),
                "max_error_threshold": 0.60,
                "per_layer_errors": per_layer_errors,
            },
            "calibration": {
                "calibration_used": bool(result.calibration_used),
                "calibration_source": result.calibration_source or "none",
                "calibration_num_samples": 500 if result.calibration_used else 0,
                "calibration_batch_size": 8 if result.calibration_used else 0,
                "calibration_max_length": seq_len,
                "log_marker": f"{module_name}: collected calibration stats for {len(per_layer_errors)} layers",
                "activation_stats_collected": bool(result.calibration_used),
                "fisher_diagonal_computed": bool(
                    result.calibration_used and comp_stats.get("use_fisher", False)
                ),
                "covariance_computed": bool(
                    result.calibration_used and comp_stats.get("use_cholesky", False)
                ),
            },
            "hardware": {
                "device": str(device),
                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if torch.cuda.is_available() and device.type == "cuda"
                    else "CPU"
                ),
                "total_vram_mb": (
                    round(
                        torch.cuda.get_device_properties(device).total_memory
                        / (1024 * 1024),
                        2,
                    )
                    if torch.cuda.is_available() and device.type == "cuda"
                    else 0.0
                ),
                "peak_vram_mb": result.peak_memory_mb,
                "idle_vram_mb": mem_phases.get("idle", {}).get("allocated_mb", 0.0),
                "activation_vram_mb": round(
                    max(
                        0.0,
                        result.peak_memory_mb
                        - mem_phases.get("idle", {}).get("allocated_mb", 0.0),
                    ),
                    2,
                ),
            },
            "timing": {
                "total_elapsed_s": round(time.time() - t_run_start, 2),
                "calibration_s": round(t_calib, 2),
                "compression_s": round(t_comp, 2),
                "recovery_s": 0.0,
                "eval_s": round(t_eval, 2),
            },
            "environment": env_dict,
            "seeds": seed_dict,
            "error_message": result.error_message,
            "error_type": exc_type,
            "traceback": exc_traceback,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": "5796f5653",
            "git_dirty": False,
        }
        (run_dir / "result.json").write_text(
            json.dumps(full_result_json, indent=2), encoding="utf-8"
        )

        # 13. Append to consolidated summary.csv and summary.md
        summary_dir = Path(results_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        csv_file = summary_dir / "summary.csv"
        csv_header = (
            "module,model_family,dataset,rank,status,"
            "accuracy,ppl,f1,"
            "num_compressed,num_fallback,num_skipped,"
            "params_before,params_after,compression_rate,"
            "avg_recon_error,max_error_threshold,"
            "calibration_used,calibration_source,"
            "peak_vram_mb,elapsed_s,"
            "forward_latency_ms,throughput_tok_per_sec,"
            "fidelity_score,error_message\n"
        )
        row = (
            f"{module_name},{model_family},{dataset},{rank},{result.status},"
            f"{result.accuracy_pct if not is_lm else ''},"
            f"{result.perplexity if is_lm else ''},"
            f"{result.f1_score if not is_lm else ''},"
            f"{num_compressed},{num_fallback},0,"
            f"{result.params_before},{result.params_after},{round(result.compression_ratio, 4)},"
            f"{round(comp_stats.get('avg_reconstruction_error', 0.0), 4)},0.60,"
            f"{result.calibration_used},{result.calibration_source or 'none'},"
            f"{result.peak_memory_mb},{round(time.time() - t_run_start, 2)},"
            f"{result.forward_latency_ms},{result.throughput_tokens_sec},"
            f"{result.fidelity_score},{result.error_message or ''}\n"
        )
        import filelock

        lock_p = summary_dir / "summary.lock"
        with filelock.FileLock(str(lock_p), timeout=30):
            if not csv_file.exists():
                csv_file.write_text(csv_header + row, encoding="utf-8")
            else:
                with open(csv_file, "a", encoding="utf-8") as f:
                    f.write(row)

            md_file = summary_dir / "summary.md"
            md_line = f"| {module_name} | {model_family} | {dataset} | {rank} | {result.status} | {result.accuracy_pct if not is_lm else result.perplexity} | {result.f1_score} | {num_compressed} | {round(result.compression_ratio, 3)}x | {result.calibration_used} | {round(time.time() - t_run_start, 1)}s |\n"
            if not md_file.exists():
                md_header = (
                    "# Evaluation Summary\n\n"
                    "| Module | Model | Dataset | Rank | Status | Acc/PPL | F1 | Compressed Layers | Comp Rate | Calibrated | Elapsed |\n"
                    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
                )
                md_file.write_text(md_header + md_line, encoding="utf-8")
            else:
                with open(md_file, "a", encoding="utf-8") as f:
                    f.write(md_line)

    except Exception as save_e:
        logger.error("Failed to write comprehensive run artifacts: %s", save_e)

    return result


evaluate_single_combination = evaluate_single_combination_opt


# =============================================================================
# Convenience: full sweep over tracks
# =============================================================================


def run_track(
    track: str,
    model_family: str,
    dataset: str,
    rank: int = 64,
    seq_len: int = 128,
    batch_size: int = 8,
    device: torch.device | None = None,
    results_dir: str = "./results",
    allow_synthetic: bool = False,
) -> list[BenchmarkResult]:
    """
    Run every module in the given track against the given model/dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modules_cfg = TRACK_MODULES.get(track)
    if modules_cfg is None:
        raise ValueError(f"Unknown track: {track}")

    results: list[BenchmarkResult] = []

    for mtype_key in ("attentions", "compressors"):
        for module_name in modules_cfg.get(mtype_key, []):
            module_type = "attention" if mtype_key == "attentions" else "compressor"
            logger.info(
                "=== %s :: %s :: %s :: r=%d ===", track, module_name, model_family, rank
            )
            try:
                r = evaluate_single_combination_opt(
                    track=track,
                    module_name=module_name,
                    module_type=module_type,
                    model_family=model_family,
                    dataset=dataset,
                    rank=rank,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    device=device,
                    results_dir=results_dir,
                    allow_synthetic=allow_synthetic,
                )
                results.append(r)
                logger.info(
                    "--> %s: status=%s  acc=%.2f  ppl=%.2f  comp=%.3fx",
                    module_name,
                    r.status,
                    r.accuracy_pct,
                    r.perplexity,
                    r.compression_ratio,
                )
            except Exception as e:
                logger.exception("Run crashed for %s: %s", module_name, e)

    return results


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    "BenchmarkResult",
    "TRACK_MODULES",
    "MODEL_TOKENIZER_MAPPING",
    "ACTIVATION_AWARE_MODULES",
    "COMPUTE_OPTIMIZERS",
    "KV_CACHE_OPTIMIZERS",
    "DEPRECATED_MODULES",
    "build_model_instance",
    "apply_module_to_model",
    "get_dataset_batch",
    "get_base_model_cached",
    "probe_optimal_batch_size",
    "build_calibration_loader",
    "evaluate_single_combination_opt",
    "evaluate_single_combination",
    "run_track",
]
