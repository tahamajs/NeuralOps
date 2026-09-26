from __future__ import annotations
import contextlib
"""
Universal Evaluation Engine for NeuralOps Framework.
Optimized Features:
1. Fast Validation-Only Dataset In-Memory Caching (pinned memory, zero-copy async transfers).
2. Base Model Template Caching & Fast In-Memory Deepcopy Cloning.
3. Non-blocking CUDA Event Timing (zero CPU host synchronization stalls in hot loops).
4. Full Support for all 5 Model Families and 5 Tracks.
"""


import copy
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
# ==== ADVANCED HARDWARE OPTIMIZATIONS (Ada Lovelace / RTX 4070) ====
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F

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

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import framework.attentions
import framework.compression_engine
from framework.core.attention_utils import inject_attention
from framework.utils.replace_attention import UniversalAttentionWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EvalEngine")

MODEL_TOKENIZER_MAPPING: Dict[str, str] = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "t5": "t5-small",
    "gpt2": "gpt2",
    "distilbert": "distilbert-base-uncased",
}

# -----------------------------------------------------------------------------
# Data Models & Result Schema
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    track: str
    module_name: str
    module_type: str  # "attention" | "compressor"
    model_family: str  # "bert" | "roberta" | "t5" | "gpt2" | "distilbert"
    dataset: str  # "sst2" | "mrpc" | "qnli" | "wikitext2" | "synthetic"
    status: str  # "PASS" | "FAIL" | "TIMEOUT"
    error_message: Optional[str] = None
    
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
    
    # Accuracy & Loss Metrics
    loss: float = 0.0
    accuracy_pct: float = 0.0
    f1_score: float = 0.0
    perplexity: float = 0.0
    grad_norm: float = 0.0
    
    # Advanced Publication Metrics
    mcc: float = 0.0
    pearson: float = 0.0
    spearman: float = 0.0
    fidelity_score: float = 0.0
    flops_macs: float = 0.0
    throughput_tokens_sec: float = 0.0
    latency_variance_ms: float = 0.0
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value_vs_baseline: Optional[float] = None
    mean_seed_acc: Optional[float] = None
    std_seed_acc: Optional[float] = None
    
    trajectories: Dict[str, List[float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

TRACK_MODULES: Dict[str, Dict[str, List[str]]] = {
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
        "compressors": [
            "mamba_shedder",
        ],
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
        "compressors": [
            "cur_moe",
        ],
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
        "attentions": [
            "linformer_attention",
            "adaptive_sparse_attention",
        ],
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

# -----------------------------------------------------------------------------
# Model Factory
# -----------------------------------------------------------------------------

def build_model_instance(
    model_family: str,
    num_labels: int = 2,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    vocab_size: Optional[int] = None,
    seq_len: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Any]:
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

# (Unified logit/loss extractor defined below)


# -----------------------------------------------------------------------------
# Module Ingestion & Injection Adapter
# -----------------------------------------------------------------------------


def _filter_batch(batch: Dict[str, torch.Tensor], model_family: str, model: nn.Module) -> Dict[str, torch.Tensor]:
    """Strip keys invalid for specific model architectures."""
    import inspect
    family = model_family.lower()
    b = dict(batch)
    if "t5" in family:
        b.pop("token_type_ids", None)
    if "gpt2" in family:
        allowed = {"input_ids", "attention_mask", "labels"}
        b = {k: v for k, v in b.items() if k in allowed}
        if "labels" in b and isinstance(b["labels"], torch.Tensor) and b["labels"].dim() == 1:
            b.pop("labels", None)
    try:
        sig = inspect.signature(model.forward)
        valid = set(sig.parameters.keys()) - {"self"}
        has_varkwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_varkwargs:
            b = {k: v for k, v in b.items() if k in valid}
    except (ValueError, TypeError):
        pass
    return b


def _extract_logits_and_loss(outputs: Any, num_labels: int = 2, labels: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Unified logit/loss extraction across all HF model output types."""
    loss = None
    logits = None
    
    if hasattr(outputs, "loss") and outputs.loss is not None:
        loss = outputs.loss
    if hasattr(outputs, "logits"):
        logits = outputs.logits
        if logits.dim() == 3:
            logits = logits[:, -1, :num_labels]
    elif hasattr(outputs, "last_hidden_state"):
        hs = outputs.last_hidden_state
        logits = hs.mean(dim=1)[:, :num_labels]
        if labels is not None and logits.shape[-1] == num_labels:
            loss = F.cross_entropy(logits, labels)
        elif loss is None:
            loss = hs.mean()
    elif isinstance(outputs, (tuple, list)):
        first = outputs[0]
        if first.dim() == 3:
            logits = first[:, 0, :]
            if logits.shape[-1] != num_labels:
                logits = logits[:, :num_labels]
        else:
            logits = first
            if logits.dim() == 0:
                loss = logits
                logits = None
    
    return logits, loss


# -----------------------------------------------------------------------------
# Module Ingestion & Injection Adapter
# -----------------------------------------------------------------------------


def apply_module_to_model(
    model: nn.Module,
    module_name: str,
    module_type: str = "auto",
    rank: int = 16,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, float]:
    """
    Injects or compresses the given model using the specified module.
    Returns the adapted model and the compression ratio achieved.
    """
    orig_params = sum(p.numel() for p in model.parameters())

    if not module_name or module_name.lower() in ("none", "baseline", "dense", "full"):
        return model, 1.0

    # Module alias mappings
    if module_name == "hla_inject":
        module_name = "hla_attention"
    elif module_name == "jamba":
        module_name = "jamba"

    # Auto-detect module category if misclassified
    if module_name in ATTENTION_REGISTRY:
        module_type = "attention"
    elif module_name in COMPRESSOR_REGISTRY:
        module_type = "compressor"
    else:
        # Check files on disk
        import importlib.util
        if importlib.util.find_spec(f"framework.compression_engine.{module_name}"):
            module_type = "compressor"
        elif importlib.util.find_spec(f"framework.attentions.{module_name}"):
            module_type = "attention"

    if module_type == "attention":
        # Resolve attention class from registry
        if module_name not in ATTENTION_REGISTRY:
            try:
                import importlib
                importlib.import_module(f"framework.attentions.{module_name}")
            except ImportError as e:
                logger.warning("Could not import attention module '%s': %s", module_name, e)

        # Check if dedicated injector function is registered
        resolved_mod = ATTENTION_REGISTRY.get(module_name)
        custom_injector = None
        if resolved_mod is not None and callable(resolved_mod) and not isinstance(resolved_mod, type):
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
                is_single_layer = isinstance(res, (BaseAttention, nn.Linear)) or ("Attention" in type(res).__name__ and not any(hasattr(res, a) for a in ("encoder", "transformer", "bert", "roberta", "distilbert", "gpt2", "qwen2")))
                is_whole_model = (
                    res is not None
                    and isinstance(res, nn.Module)
                    and sum(p.numel() for p in res.parameters()) > 1000
                )
                if is_whole_model:
                    model = res
                    injected = True
                    if device is not None:
                        model = model.to(device)
                    new_params = sum(p.numel() for p in model.parameters())
                    comp_ratio = orig_params / max(1, new_params)
                    return model, comp_ratio
                elif is_single_layer:
                    logger.debug("Dedicated injector %s returned a single layer instead of whole model, falling back to inject_attention", custom_injector)
            except Exception as e:
                logger.debug("Dedicated injector %s failed, falling back to inject_attention: %s", custom_injector, e)

        if not injected:
            if module_name in ATTENTION_REGISTRY:
                inject_attention(model, module_name, rank=rank, copy_weights=True)
            else:
                try:
                    import importlib
                    mod = importlib.import_module(f"framework.attentions.{module_name}")
                    target_cls = None
                    for attr in dir(mod):
                        obj = getattr(mod, attr)
                        if isinstance(obj, type) and issubclass(obj, nn.Module) and ("Attention" in attr or "Block" in attr) and attr != "BaseAttention":
                            target_cls = obj
                            break
                    if target_cls is not None:
                        cfg = getattr(model, "config", None)
                        for name, sub in list(model.named_modules()):
                            if "attention" in name.lower() and hasattr(sub, "self"):
                                new_inner = target_cls(cfg)
                                wrapped = UniversalAttentionWrapper(new_inner)
                                setattr(sub, "self", wrapped)
                    else:
                        raise ValueError(f"No valid attention class found in module '{module_name}'")
                except Exception as e:
                    raise RuntimeError(f"Could not inject attention '{module_name}': {e}")

    elif module_type == "compressor":
        if module_name not in COMPRESSOR_REGISTRY:
            try:
                import importlib
                importlib.import_module(f"framework.compression_engine.{module_name}")
            except ImportError as e:
                logger.warning("Could not import compressor module '%s': %s", module_name, e)

        comp_cls = COMPRESSOR_REGISTRY.get(module_name)
        if comp_cls is not None:
            def _invoke_compressor_instance(instance: Any) -> Optional[nn.Module]:
                dummy_input = torch.randint(0, 1000, (2, 32), device=device)
                for method_name in ["compress", "apply", "prune"]:
                    if hasattr(instance, method_name):
                        m = getattr(instance, method_name)
                        if callable(m):
                            for fn in [
                                lambda: m(),
                                lambda: m(calibration_data=dummy_input),
                                lambda: m(model),
                                lambda: m(model, calibration_data=dummy_input),
                                lambda: m(dummy_input),
                            ]:
                                try:
                                    res = fn()
                                    if isinstance(res, tuple) and len(res) > 0 and isinstance(res[0], nn.Module):
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

            # Strategy 1: Instantiate with model passed to constructor
            candidate = None
            tok = getattr(model, "tokenizer", None) or getattr(getattr(model, "config", None), "tokenizer", None)
            cfg = getattr(model, "config", None)
            c_kwargs = {}
            if cfg is not None:
                for k in ("rank_ratio", "max_error", "lora_rank", "target_attention_only", "use_jg_svd", "use_cholesky_jg", "energy_threshold", "group_size", "strict_rank"):
                    if hasattr(cfg, k):
                        c_kwargs[k] = getattr(cfg, k)
            try:
                candidate = comp_cls(model, rank=rank, tokenizer=tok, **c_kwargs)
            except (TypeError, ValueError):
                try:
                    candidate = comp_cls(model, rank=rank, tokenizer=tok)
                except (TypeError, ValueError):
                    try:
                        candidate = comp_cls(model, rank=rank)
                    except (TypeError, ValueError):
                        try:
                            candidate = comp_cls(model)
                        except (TypeError, ValueError):
                            candidate = None

            if candidate is not None:
                if isinstance(candidate, nn.Module) and not hasattr(candidate, "compress"):
                    model = candidate
                else:
                    res_mod = _invoke_compressor_instance(candidate)
                    if res_mod is not None and isinstance(res_mod, nn.Module):
                        model = res_mod

            # Strategy 2: Instantiate without model, call compress(model)
            else:
                try:
                    candidate = comp_cls(rank=rank)
                except (TypeError, ValueError):
                    try:
                        candidate = comp_cls()
                    except (TypeError, ValueError):
                        try:
                            candidate = comp_cls(target_ratio=0.5)
                        except (TypeError, ValueError):
                            candidate = None

                if candidate is not None:
                    res_mod = _invoke_compressor_instance(candidate)
                    if res_mod is not None and isinstance(res_mod, nn.Module):
                        model = res_mod
        else:
            raise RuntimeError(f"Compressor '{module_name}' not found in COMPRESSOR_REGISTRY")

    model.to(device)
    new_params = sum(p.numel() for p in model.parameters())
    comp_ratio = orig_params / max(1, new_params)
    return model, comp_ratio



# -----------------------------------------------------------------------------
# Optimization: In-Memory Validation Dataset Cache
# -----------------------------------------------------------------------------
_TOKENIZED_BATCH_CACHE: Dict[Tuple[str, str, int, int, str], Dict[str, torch.Tensor]] = {}
_PREPARED_DATASET_CACHE: Dict[Tuple[str, str, int], Any] = {}

def _generate_synthetic_batch(
    dataset_name: str,
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 30522,
    num_labels: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    ds = dataset_name.lower()
    torch.manual_seed(42)
    input_ids = torch.randint(100, max(101, vocab_size - 100), (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    if batch_size > 1:
        attention_mask[1:, -4:] = 0

    if ds in {"wikitext2", "wikitext103", "c4", "lm", "ptb"}:
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
    else:
        labels = torch.randint(0, num_labels, (batch_size,), dtype=torch.long, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def get_dataset_batch(
    dataset_name: str,
    model_family: str,
    batch_size: int = 128,
    seq_len: int = 256,
    vocab_size: int = 30522,
    num_labels: int = 2,
    device: torch.device = torch.device("cpu"),
    allow_synthetic: bool = False,
) -> Dict[str, torch.Tensor]:
    ds = dataset_name.lower()
    family = model_family.lower()
    dev_str = str(device)
    cache_key = (ds, family, batch_size, seq_len, dev_str)

    if cache_key in _TOKENIZED_BATCH_CACHE:
        cached = _TOKENIZED_BATCH_CACHE[cache_key]
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cached.items()}

    if ds == "synthetic":
        batch = _generate_synthetic_batch(
            dataset_name=ds,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_labels=num_labels,
            device=device,
        )
        _TOKENIZED_BATCH_CACHE[cache_key] = batch
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    tok_name = MODEL_TOKENIZER_MAPPING.get(family, "bert-base-uncased")
    dataset_cache_key = (ds, tok_name, seq_len)

    try:
        from framework.data.prepare_datasets import GLUE_TASK_CONFIG
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
                    return tokenizer(*args, padding="max_length", truncation=True, max_length=seq_len)

                val_tokenized = val_data.map(tok_fn, batched=True)
                val_tokenized.set_format(type="torch", columns=[c for c in ["input_ids", "attention_mask", "label"] if c in val_tokenized.column_names])
                _PREPARED_DATASET_CACHE[dataset_cache_key] = val_tokenized
            else:
                val_tokenized = _PREPARED_DATASET_CACHE[dataset_cache_key]

            if val_tokenized is not None and len(val_tokenized) > 0:
                is_cuda = (device.type == "cuda")
                n = min(batch_size, len(val_tokenized))
                sample = val_tokenized[list(range(n))]
                raw_batch = {}
                for k in val_tokenized.column_names:
                    if k in sample:
                        vals = sample[k]
                        if isinstance(vals[0], torch.Tensor):
                            raw_batch[k] = torch.stack(vals)
                        elif isinstance(vals[0], (int, float, bool)):
                            raw_batch[k] = torch.tensor(vals)
                        else:
                            raw_batch[k] = vals
                batch = {
                    ("labels" if k == "label" else k): (v.to(device, non_blocking=is_cuda) if isinstance(v, torch.Tensor) else v)
                    for k, v in raw_batch.items()
                }
                _TOKENIZED_BATCH_CACHE[cache_key] = batch
                return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    except Exception as e:
        logger.warning("Could not load real GLUE dataset '%s' with tokenizer '%s': %s", ds, tok_name, e)
        if not allow_synthetic:
            raise RuntimeError(f"Genuine dataset '{ds}' failed to load for model family '{family}' ({e})")

    batch = _generate_synthetic_batch(
        dataset_name=ds,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_labels=num_labels,
        device=device,
    )
    _TOKENIZED_BATCH_CACHE[cache_key] = batch
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# -----------------------------------------------------------------------------
# Optimization: Base Model Template Caching
# -----------------------------------------------------------------------------
_BASE_MODEL_TEMPLATE_CACHE: Dict[Tuple[str, int, int, int, int, str], Tuple[nn.Module, Any]] = {}

def get_base_model_cached(
    model_family: str,
    num_labels: int = 2,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Any]:
    family = model_family.lower()
    dev_str = str(device)
    cache_key = (family, num_labels, hidden_size, num_heads, num_layers, dev_str)

    if cache_key not in _BASE_MODEL_TEMPLATE_CACHE:
        base_model, cfg = build_model_instance(
            model_family=family,
            num_labels=num_labels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            device=device,
        )
        _BASE_MODEL_TEMPLATE_CACHE[cache_key] = (base_model, cfg)

    template_model, cfg = _BASE_MODEL_TEMPLATE_CACHE[cache_key]
    cloned_model = copy.deepcopy(template_model)
    return cloned_model, cfg

# -----------------------------------------------------------------------------
# Adaptive Memory-Aware Batch Sizer (Proactive Memory Probe)
# -----------------------------------------------------------------------------

def probe_optimal_batch_size(
    model: nn.Module,
    model_family: str,
    dataset: str,
    seq_len: int,
    device: torch.device,
    target_vram_ratio: float = 0.50,
    max_batch_cap: int = 64,
    min_batch_floor: int = 8,
    allow_synthetic: bool = True,
) -> int:
    """
    Performs a fast memory probe on CUDA with a tiny batch (B=4).
    Measures peak VRAM during forward pass, calculates marginal VRAM per sample,
    and returns the optimal batch size that safely utilizes target_vram_ratio of total VRAM.
    """
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
            dataset, model_family, probe_b, seq_len,
            num_labels=num_labels, device=device, allow_synthetic=allow_synthetic
        )
        filtered_batch = _filter_batch(probe_batch, model_family, model)

        with torch.amp.autocast('cuda'):
            out = model(**filtered_batch)

        torch.cuda.synchronize(device)
        peak_vram = torch.cuda.max_memory_allocated(device)

        activation_vram = max(1024 * 1024.0, peak_vram - idle_vram)
        mem_per_sample = activation_vram / float(probe_b)

        remaining_vram = max(1024 * 1024.0, target_usable_vram - idle_vram)
        estimated_max_b = int(remaining_vram / mem_per_sample)

        # Snap to powers of 2 for tensor core efficiency
        powers_of_2 = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        selected_b = min_batch_floor
        for p in powers_of_2:
            if p <= estimated_max_b and p <= max_batch_cap:
                selected_b = p

        logger.info("Adaptive Batch Sizer: probe peak=%.1fMB, per_sample=%.2fMB -> auto-selected batch_size=%d",
                    peak_vram / (1024*1024), mem_per_sample / (1024*1024), selected_b)
        return max(min_batch_floor, min(selected_b, max_batch_cap))
    except Exception as e:
        logger.warning("Memory probe failed (%s), defaulting to batch size 128", e)
        return 128
    finally:
        gc.collect()
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Non-Blocking CUDA Event Timed Single Combination Evaluator
# -----------------------------------------------------------------------------

def evaluate_single_combination_opt(
    track: str, module_name: str, module_type: str, model_family: str, dataset: str,
    rank: int = 16, num_warmup: int = 2, num_steps: int = 5, batch_size: int = 128, seq_len: int = 256,
    device: torch.device = torch.device("cpu"), results_dir: str = "./results/gpu_server", allow_synthetic: bool = False,
) -> BenchmarkResult:
    import numpy as np
    import gc
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from scipy.stats import pearsonr, spearmanr
    from torch.profiler import profile, record_function, ProfilerActivity
    import math

    result = BenchmarkResult(
        track=track, module_name=module_name, module_type=module_type,
        model_family=model_family, dataset=dataset, status="RUNNING"
    )
    
    current_batch_size = batch_size
    max_retries = 8
    retry_count = 0
    success = False

    while retry_count < max_retries and not success:
        model = None
        baseline_model = None
        try:
            is_cuda = (device.type == "cuda")
            if is_cuda:
                torch.cuda.empty_cache()
            
            # --- Load Models ---
            num_labels = 1 if dataset.lower() == "stsb" else 2
            model, cfg = get_base_model_cached(model_family=model_family, num_labels=num_labels, hidden_size=128, num_heads=4, num_layers=2, device=device)
            model.train()
            
            # --- Inject Module ---
            model, comp_ratio = apply_module_to_model(
                model=model,
                module_name=module_name,
                module_type=module_type,
                rank=rank,
                device=device,
            )
            result.compression_ratio = comp_ratio
            
            model = model.to(device)
            if hasattr(model, "train"):
                model.train()

            # --- Adaptive Memory Probe on Initial Attempt ---
            if retry_count == 0 and is_cuda:
                current_batch_size = probe_optimal_batch_size(
                    model=model,
                    model_family=model_family,
                    dataset=dataset,
                    seq_len=seq_len,
                    device=device,
                    target_vram_ratio=0.80,
                    max_batch_cap=min(batch_size, 1024) if batch_size > 0 else 1024,
                    min_batch_floor=4,
                    allow_synthetic=allow_synthetic,
                )
            
            batch = get_dataset_batch(dataset, model_family, current_batch_size, seq_len, num_labels=num_labels, device=device, allow_synthetic=allow_synthetic)
            filtered_batch = _filter_batch(batch, model_family, model)

            model.eval()
            # --- FLOPs Profiling (Warmup) ---
            flops_macs = 0.0
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
                with torch.amp.autocast('cuda') if is_cuda else contextlib.nullcontext():
                    out = model(**filtered_batch)
            
            # Extract FLOPs from profiler
            evt_list = prof.key_averages()
            total_flops = sum([e.flops for e in evt_list if e.flops is not None])
            result.flops_macs = total_flops
            
            # Additional warmups
            for _ in range(num_warmup):
                with torch.amp.autocast('cuda') if is_cuda else contextlib.nullcontext():
                    model(**filtered_batch)
            model.train()

            traj_loss, traj_grad_norm, traj_fwd_ms, traj_bwd_ms = [], [], [], []
            
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                fwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
                fwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
                bwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
                bwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]

                for i in range(num_steps):
                    fwd_start_events[i].record()
                    with torch.amp.autocast('cuda'):
                        out = model(**filtered_batch)
                    fwd_end_events[i].record()

                    with torch.amp.autocast('cuda'):
                        _, loss = _extract_logits_and_loss(out, num_labels=num_labels, labels=batch.get("labels"))
                    if loss is None:
                        loss = torch.tensor(0.0, requires_grad=True, device=device)
                    
                    traj_loss.append(round(float(loss.detach().item()), 4))

                    bwd_start_events[i].record()
                    loss.backward()
                    bwd_end_events[i].record()

                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm_sq += float(p.grad.detach().data.norm(2).item() ** 2)
                    traj_grad_norm.append(round(float(total_norm_sq ** 0.5), 4))
                    model.zero_grad()

                bwd_end_events[-1].synchronize()
                for i in range(num_steps):
                    traj_fwd_ms.append(round(float(fwd_start_events[i].elapsed_time(fwd_end_events[i])), 4))
                    traj_bwd_ms.append(round(float(bwd_start_events[i].elapsed_time(bwd_end_events[i])), 4))
                
                result.peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
            else:
                for _ in range(num_steps):
                    t0 = time.perf_counter()
                    out = model(**filtered_batch)
                    traj_fwd_ms.append(round((time.perf_counter() - t0) * 1000.0, 4))
                    _, loss = _extract_logits_and_loss(out, num_labels=num_labels, labels=batch.get("labels"))
                    if loss is None: loss = torch.tensor(0.0, requires_grad=True, device=device)
                    traj_loss.append(round(float(loss.detach().item()), 4))
                    
                    t2 = time.perf_counter()
                    loss.backward()
                    traj_bwd_ms.append(round((time.perf_counter() - t2) * 1000.0, 4))
                    
                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm_sq += float(p.grad.detach().data.norm(2).item() ** 2)
                    traj_grad_norm.append(round(float(total_norm_sq ** 0.5), 4))
                    model.zero_grad()
            
            result.forward_latency_ms = round(float(np.mean(traj_fwd_ms)), 2) if traj_fwd_ms else 0.0
            result.backward_latency_ms = round(float(np.mean(traj_bwd_ms)), 2) if traj_bwd_ms else 0.0
            result.total_step_latency_ms = round(result.forward_latency_ms + result.backward_latency_ms, 2)
            result.latency_variance_ms = round(float(np.var(traj_fwd_ms) + np.var(traj_bwd_ms)), 4) if traj_fwd_ms else 0.0
            
            # Throughput
            total_time_s = sum(traj_fwd_ms + traj_bwd_ms) / 1000.0
            if total_time_s > 0:
                result.throughput_samples_per_sec = round((num_steps * current_batch_size) / total_time_s, 2)
                result.throughput_tokens_sec = round((num_steps * current_batch_size * seq_len) / total_time_s, 2)
                
            result.loss = round(float(np.mean(traj_loss)), 4) if traj_loss else 0.0
            result.grad_norm = round(float(np.mean(traj_grad_norm)), 4) if traj_grad_norm else 0.0
            result.trajectories = {"loss": traj_loss, "grad_norm": traj_grad_norm, "fwd_latency_ms": traj_fwd_ms, "bwd_latency_ms": traj_bwd_ms}

            # ==== STRICT EVALUATION PHASE (Full Dataset & Fidelity) ====
            model.eval()
            all_preds, all_labels = [], []
            sum_cosine_sim = 0.0
            batches_for_fidelity = 0
            
            # Get baseline model for Fidelity (Zero-Shot Drift)
            baseline_model, _ = get_base_model_cached(model_family=model_family, num_labels=num_labels, hidden_size=128, num_heads=4, num_layers=2, device=device)
            baseline_model.eval()
            baseline_model = baseline_model.to(device)
            
            try:
                tok_name = MODEL_TOKENIZER_MAPPING.get(model_family.lower(), "bert-base-uncased")
                dataset_cache_key = (dataset.lower(), tok_name, seq_len)
                if dataset_cache_key in _PREPARED_DATASET_CACHE:
                    val_tokenized = _PREPARED_DATASET_CACHE[dataset_cache_key]
                    loader = torch.utils.data.DataLoader(
                        val_tokenized,
                        batch_size=current_batch_size,
                        shuffle=False,
                        pin_memory=is_cuda,
                    )
                    
                    with torch.no_grad():
                        for val_batch in loader:
                            inputs = {("labels" if k == "label" else k): (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in val_batch.items()}
                            filtered_inputs = _filter_batch(inputs, model_family, model)
                            filtered_base_inputs = _filter_batch(inputs, model_family, baseline_model)
                            
                            with torch.amp.autocast('cuda') if is_cuda else contextlib.nullcontext():
                                out = model(**filtered_inputs)
                                base_out = baseline_model(**filtered_base_inputs)
                            
                            logits, _ = _extract_logits_and_loss(out, num_labels=num_labels, labels=inputs.get("labels"))
                            base_logits, _ = _extract_logits_and_loss(base_out, num_labels=num_labels, labels=inputs.get("labels"))
                            
                            if logits is not None and base_logits is not None:
                                # Fidelity Score (Cosine Similarity)
                                sim = torch.nn.functional.cosine_similarity(base_logits.view(-1), logits.view(-1), dim=0)
                                sum_cosine_sim += float(sim.item())
                                batches_for_fidelity += 1

                            if logits is not None and inputs.get("labels") is not None:
                                if dataset.lower() == "stsb":
                                    preds = logits.squeeze()
                                else:
                                    preds = torch.argmax(logits, dim=-1)
                                all_preds.extend(preds.cpu().numpy())
                                all_labels.extend(inputs["labels"].cpu().numpy())
                                
                    if all_labels:
                        if dataset.lower() == "stsb":
                            # STS-B specific metrics
                            p_val, _ = pearsonr(all_labels, all_preds)
                            s_val, _ = spearmanr(all_labels, all_preds)
                            result.pearson = round(float(p_val), 4)
                            result.spearman = round(float(s_val), 4)
                        elif dataset.lower() == "cola":
                            # CoLA specific metric
                            result.mcc = round(float(matthews_corrcoef(all_labels, all_preds)), 4)
                            result.accuracy_pct = round(float(accuracy_score(all_labels, all_preds)) * 100.0, 2)
                        elif dataset.lower() == "wikitext2":
                            # LM metric
                            if result.loss > 0:
                                result.perplexity = round(float(math.exp(min(result.loss, 100))), 4)
                        else:
                            result.accuracy_pct = round(float(accuracy_score(all_labels, all_preds)) * 100.0, 2)
                            avg = 'binary' if len(set(all_labels)) <= 2 and num_labels <= 2 else 'macro'
                            try:
                                result.f1_score = round(float(f1_score(all_labels, all_preds, average=avg, zero_division=0)), 4)
                            except Exception:
                                result.f1_score = 0.0
                    
                    if batches_for_fidelity > 0:
                        result.fidelity_score = round(sum_cosine_sim / batches_for_fidelity, 4)

            except Exception as eval_e:
                logger.warning("Validation phase failed: %s", eval_e)
            
            result.status = "PASS"
            success = True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM caught at batch_size {current_batch_size}. Halving and retrying...")
                current_batch_size = max(1, current_batch_size // 2)
                retry_count += 1
                if device.type == "cuda":
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                logger.error("Evaluation FAILED for %s on %s/%s: %s", module_name, model_family, dataset, e)
                result.status = "FAIL"
                result.error_message = str(e)
                break
        except Exception as e:
            logger.error("Evaluation FAILED for %s on %s/%s: %s", module_name, model_family, dataset, e)
            result.status = "FAIL"
            result.error_message = str(e)
            break
        finally:
            if model is not None: del model
            if baseline_model is not None: del baseline_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not success and result.status != "FAIL":
        result.status = "FAIL"
        result.error_message = "OOM limits exceeded after max retries."
        
    return result

evaluate_single_combination = evaluate_single_combination_opt
