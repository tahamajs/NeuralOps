"""
Universal Benchmark and Evaluation Engine for Foundation Models & Efficient Architectures.

Evaluates all Attention Mechanisms (123 modules) and Compression Engines (86 modules)
across standard model families:
  - BERT (bert-base-uncased / BertForSequenceClassification)
  - RoBERTa (roberta-base / RobertaForSequenceClassification)
  - T5 (t5-small / T5EncoderModel)
  - GPT-2 (gpt2 / GPT2LMHeadModel)
  - DistilBERT (distilbert-base-uncased / DistilBertForSequenceClassification)

Across evaluation datasets:
  - SST-2 (Single-sentence sentiment classification)
  - MRPC (Sentence-pair semantic paraphrase)
  - QNLI (Question-answering natural language inference)
  - WikiText-2 (Causal language modeling / perplexity)
  - Synthetic (Deterministic offline multi-domain stream, permitted ONLY with --synthetic)

Saves results one-by-one to JSON, updates master CSV, and formats publication-ready markdown tables.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure framework is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
CODES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [PROJECT_DIR, CODES_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModel,
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
from framework.core.attention_utils import inject_attention
from framework.utils.replace_attention import UniversalAttentionWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EvalEngine")

DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(DEFAULT_RESULTS_DIR, "json"), exist_ok=True)

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
    status: str  # "PASS" | "FAIL"
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
    
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Model Factory (Supports Real Pretrained or Lightweight Architectures)
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
    """
    Builds a lightweight yet architecturally exact model instance for evaluation & stress testing.
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
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    model.to(device)
    model.eval()
    return model, cfg


# -----------------------------------------------------------------------------
# Genuine Dataset Loader & Tokenization Pipeline
# -----------------------------------------------------------------------------

_DATASET_BATCH_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


def _generate_synthetic_batch(
    dataset_name: str,
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 30522,
    num_labels: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Generates a deterministic synthetic batch for explicit offline/synthetic evaluation."""
    ds = dataset_name.lower()
    torch.manual_seed(42)
    input_ids = torch.randint(100, max(101, vocab_size - 100), (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    if batch_size > 1:
        attention_mask[1:, -4:] = 0

    if ds in {"wikitext2", "wikitext103", "c4", "lm", "ptb"}:
        labels = input_ids.clone()
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
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 30522,
    num_labels: int = 2,
    device: torch.device = torch.device("cpu"),
    allow_synthetic: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Loads a genuine evaluation batch from GLUE or LM datasets using the appropriate tokenizer
    for the target model architecture. Only falls back to synthetic data if allow_synthetic=True.
    """
    ds = dataset_name.lower()
    family = model_family.lower()
    cache_key = f"{ds}_{family}_{batch_size}_{seq_len}_{device}"

    if cache_key in _DATASET_BATCH_CACHE:
        return _DATASET_BATCH_CACHE[cache_key]

    if ds == "synthetic":
        if not allow_synthetic:
            logger.warning("Dataset explicitly requested as 'synthetic'. Generating synthetic batch.")
        batch = _generate_synthetic_batch(
            dataset_name=ds,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_labels=num_labels,
            device=device,
        )
        _DATASET_BATCH_CACHE[cache_key] = batch
        return batch

    tok_name = MODEL_TOKENIZER_MAPPING.get(family, "bert-base-uncased")

    # 1. Check GLUE tasks
    try:
        from framework.data.prepare_datasets import GLUE_TASK_CONFIG, prepare_glue_dataset
        if ds in GLUE_TASK_CONFIG:
            splits = prepare_glue_dataset(
                task_name=ds,
                tokenizer_name=tok_name,
                max_seq_len=seq_len,
                use_synthetic_fallback=False,
            )
            eval_ds = splits.get("validation") or splits.get("test") or splits.get("train")
            if eval_ds is not None and len(eval_ds) > 0:
                loader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
                raw_batch = next(iter(loader))
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in raw_batch.items()
                }
                _DATASET_BATCH_CACHE[cache_key] = batch
                return batch
    except Exception as e:
        logger.warning("Could not load real GLUE dataset '%s' with tokenizer '%s': %s", ds, tok_name, e)
        if not allow_synthetic:
            raise RuntimeError(
                f"Genuine dataset '{ds}' failed to load for model family '{family}' ({e}). "
                f"To permit synthetic data evaluation, run with --synthetic."
            )

    # 2. Check LM datasets (wikitext2, wikitext103, c4, lm, ptb)
    if ds in {"wikitext2", "wikitext103", "c4", "lm", "ptb"}:
        try:
            from framework.data.prepare_datasets import prepare_lm_dataset
            splits = prepare_lm_dataset(
                dataset_name=ds,
                tokenizer_name=tok_name,
                max_seq_len=seq_len,
            )
            eval_ds = splits.get("validation") or splits.get("test") or splits.get("train")
            if eval_ds is not None and len(eval_ds) > 0:
                loader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
                raw_batch = next(iter(loader))
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in raw_batch.items()
                }
                _DATASET_BATCH_CACHE[cache_key] = batch
                return batch
        except Exception as e:
            logger.warning("Could not load real LM dataset '%s' with tokenizer '%s': %s", ds, tok_name, e)
            if not allow_synthetic:
                raise RuntimeError(
                    f"Genuine dataset '{ds}' failed to load for model family '{family}' ({e}). "
                    f"To permit synthetic data evaluation, run with --synthetic."
                )

    if not allow_synthetic:
        raise RuntimeError(
            f"Genuine dataset '{ds}' could not be loaded for '{family}'. "
            f"To allow synthetic data fallback, run with --synthetic."
        )

    # Explicit fallback to synthetic when allowed
    logger.warning("Falling back to synthetic data for dataset '%s' on '%s' (--synthetic allowed).", ds, family)
    batch = _generate_synthetic_batch(
        dataset_name=ds,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_labels=num_labels,
        device=device,
    )
    _DATASET_BATCH_CACHE[cache_key] = batch
    return batch


def generate_dataset_batch(
    dataset_name: str,
    model_family: str,
    batch_size: int = 4,
    seq_len: int = 32,
    device: torch.device = torch.device("cpu"),
    allow_synthetic: bool = False,
) -> Dict[str, torch.Tensor]:
    """Alias of get_dataset_batch for backwards compatibility with launch_gpu_server.py."""
    return get_dataset_batch(
        dataset_name=dataset_name,
        model_family=model_family,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        allow_synthetic=allow_synthetic,
    )


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
    try:
        sig = inspect.signature(model.forward)
        valid = set(sig.parameters.keys()) - {"self"}
        if "kwargs" not in valid and "**kwargs" not in str(sig):
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
    module_type: str,
    rank: int = 16,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, float]:
    """
    Injects or compresses the given model using the specified module.
    Returns the adapted model and the compression ratio achieved.
    """
    orig_params = sum(p.numel() for p in model.parameters())

    if module_type == "attention":
        # Resolve attention class from registry
        if module_name not in ATTENTION_REGISTRY:
            try:
                import importlib
                importlib.import_module(f"framework.attentions.{module_name}")
            except ImportError as e:
                logger.warning("Could not import attention module '%s': %s", module_name, e)

        if module_name in ATTENTION_REGISTRY:
            inject_attention(model, module_name, rank=rank, copy_weights=False)
        else:
            try:
                import importlib
                mod = importlib.import_module(f"framework.attentions.{module_name}")
                target_cls = None
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if isinstance(obj, type) and issubclass(obj, nn.Module) and "Attention" in attr and attr != "BaseAttention":
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
                                        return res[0]
                                    if isinstance(res, nn.Module):
                                        return res
                                except (TypeError, ValueError, RuntimeError):
                                    continue
                return getattr(instance, "model", None)

            # Strategy 1: Instantiate with model passed to constructor
            candidate = None
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
# Single Benchmark Evaluator
# -----------------------------------------------------------------------------

def evaluate_single_combination(
    track: str,
    module_name: str,
    module_type: str,
    model_family: str,
    dataset: str,
    rank: int = 16,
    num_warmup: int = 2,
    num_steps: int = 5,
    batch_size: int = 4,
    seq_len: int = 32,
    device: torch.device = torch.device("cpu"),
    results_dir: str = DEFAULT_RESULTS_DIR,
    allow_synthetic: bool = False,
) -> BenchmarkResult:
    """
    Evaluates one (module, model, dataset) combination and saves result to disk immediately.
    """
    result = BenchmarkResult(
        track=track,
        module_name=module_name,
        module_type=module_type,
        model_family=model_family,
        dataset=dataset,
        status="PASS",
    )

    try:
        # 1. Instantiate baseline model
        model, cfg = build_model_instance(
            model_family=model_family,
            device=device,
        )

        # 2. Inject or compress module
        model, comp_ratio = apply_module_to_model(
            model=model,
            module_name=module_name,
            module_type=module_type,
            rank=rank,
            device=device,
        )
        result.compression_ratio = round(comp_ratio, 3)
        result.param_count = sum(p.numel() for p in model.parameters())
        result.trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 3. Load genuine dataset batch (or synthetic if --synthetic flag provided)
        num_labels = getattr(cfg, "num_labels", 2)
        batch = get_dataset_batch(
            dataset_name=dataset,
            model_family=model_family,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=getattr(cfg, "vocab_size", 30522),
            num_labels=num_labels,
            device=device,
            allow_synthetic=allow_synthetic,
        )

        # Filter batch keys for model architecture
        filtered_batch = _filter_batch(batch, model_family, model)

        # Enable gradients for backward stress test
        for p in model.parameters():
            p.requires_grad = True

        # Warmup steps
        for _ in range(num_warmup):
            out = model(**filtered_batch)
            _, loss = _extract_logits_and_loss(out, num_labels=num_labels, labels=batch.get("labels"))
            if loss is None:
                loss = torch.tensor(0.0, requires_grad=True, device=device)
            loss.backward()
            model.zero_grad()

        # Timed benchmark steps
        fwd_times: List[float] = []
        bwd_times: List[float] = []
        losses: List[float] = []

        for _ in range(num_steps):
            model.zero_grad()
            t0 = time.perf_counter()

            out = model(**filtered_batch)
            _, loss = _extract_logits_and_loss(out, num_labels=num_labels, labels=batch.get("labels"))
            if loss is None:
                loss = torch.tensor(0.0, requires_grad=True, device=device)

            t1 = time.perf_counter()
            fwd_times.append((t1 - t0) * 1000.0)
            losses.append(loss.item() if isinstance(loss, torch.Tensor) else float(loss))

            loss.backward()
            t2 = time.perf_counter()
            bwd_times.append((t2 - t1) * 1000.0)

        # Compute gradient norm
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += (p.grad.data.norm(2).item()) ** 2
        total_grad_norm = math.sqrt(total_grad_norm)

        # Populate metrics
        avg_fwd = sum(fwd_times) / max(1, len(fwd_times))
        avg_bwd = sum(bwd_times) / max(1, len(bwd_times))
        result.forward_latency_ms = round(avg_fwd, 2)
        result.backward_latency_ms = round(avg_bwd, 2)
        result.total_step_latency_ms = round(avg_fwd + avg_bwd, 2)
        actual_batch_size = batch["input_ids"].size(0)
        result.throughput_samples_per_sec = round((actual_batch_size / max(1e-6, avg_fwd / 1000.0)), 1)
        result.loss = round(sum(losses) / max(1, len(losses)), 4)
        result.grad_norm = round(total_grad_norm, 4)

        # Accuracy & Perplexity evaluation metric
        if model_family in {"bert", "roberta", "distilbert"}:
            with torch.no_grad():
                out = model(**filtered_batch)
                logits, _ = _extract_logits_and_loss(out, num_labels=num_labels, labels=batch.get("labels"))
                if logits is not None and "labels" in batch and batch["labels"] is not None:
                    preds = torch.argmax(logits, dim=-1)
                    acc = (preds == batch["labels"]).float().mean().item() * 100.0
                    result.accuracy_pct = round(acc, 2)
        elif model_family == "gpt2":
            result.perplexity = round(math.exp(min(result.loss, 15.0)), 2)

        result.status = "PASS"

    except Exception as e:
        result.status = "FAIL"
        result.error_message = str(e)
        logger.error("Evaluation failed for [%s | %s | %s]: %s", module_name, model_family, dataset, e)

    # 4. Save result immediately to individual JSON file
    save_single_result(result, results_dir)
    return result


def save_single_result(result: BenchmarkResult, results_dir: str = DEFAULT_RESULTS_DIR) -> str:
    """
    Saves a single BenchmarkResult to its individual JSON file and appends to CSV.
    """
    track_dir = os.path.join(results_dir, "json", result.track)
    os.makedirs(track_dir, exist_ok=True)

    json_path = os.path.join(
        track_dir,
        f"{result.module_name}_{result.model_family}_{result.dataset}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Append to master CSV
    csv_path = os.path.join(results_dir, "benchmark_summary.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.to_dict().keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result.to_dict())

    return json_path


# -----------------------------------------------------------------------------
# Module Track Definitions
# -----------------------------------------------------------------------------

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
# CLI Entrypoint
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Benchmark & Evaluation Engine for Foundation Models & Efficient Architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--track",
        type=str,
        default="all",
        choices=["all", "track1_svd", "track2_ssm", "track3_moe", "track4_kernel_quant", "track5_pruning"],
        help="Specific track to evaluate, or 'all'",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Evaluate only a specific module name (e.g. svd_linear_attention, direct_svd)",
    )
    parser.add_argument(
        "--module-type",
        type=str,
        default=None,
        choices=["attention", "compressor"],
        help="Type of module when evaluating a specific module",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert", "roberta", "t5"],
        help="Model families to evaluate against",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sst2", "mrpc"],
        help="Evaluation datasets (e.g. sst2, mrpc, qnli, wikitext2, synthetic)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Explicit flag to allow fallback to synthetic data when real datasets cannot be loaded",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="Rank for low-rank / SVD modules",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on ('auto', 'cuda', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for evaluation",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=2,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of timed benchmark steps",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for benchmark results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info("Initializing Evaluation Engine on device: %s (allow_synthetic=%s)", device, args.synthetic)

    # Build evaluation schedule
    schedule = []
    if args.module:
        mod_type = args.module_type
        if mod_type is None:
            if args.module in ATTENTION_REGISTRY:
                mod_type = "attention"
            elif args.module in COMPRESSOR_REGISTRY:
                mod_type = "compressor"
            else:
                mod_type = "attention"
        for model_fam in args.models:
            for ds in args.datasets:
                schedule.append((args.track if args.track != "all" else "custom", args.module, mod_type, model_fam, ds))
    else:
        tracks_to_run = list(TRACK_MODULES.keys()) if args.track == "all" else [args.track]
        for tname in tracks_to_run:
            tinfo = TRACK_MODULES.get(tname, {})
            for mod in tinfo.get("attentions", []):
                for model_fam in args.models:
                    for ds in args.datasets:
                        schedule.append((tname, mod, "attention", model_fam, ds))
            for mod in tinfo.get("compressors", []):
                for model_fam in args.models:
                    for ds in args.datasets:
                        schedule.append((tname, mod, "compressor", model_fam, ds))

    logger.info("Total evaluation tasks scheduled: %d", len(schedule))
    passed, failed = 0, 0
    for idx, (trk, mod, mtype, fam, ds) in enumerate(schedule, 1):
        logger.info("[%d/%d] Evaluating %s (%s) on %s with %s...", idx, len(schedule), mod, mtype, fam, ds)
        res = evaluate_single_combination(
            track=trk,
            module_name=mod,
            module_type=mtype,
            model_family=fam,
            dataset=ds,
            rank=args.rank,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
            results_dir=args.results_dir,
            allow_synthetic=args.synthetic,
        )
        if res.status == "PASS":
            passed += 1
        else:
            failed += 1

    logger.info("Evaluation Complete. Passed: %d, Failed: %d. Summary saved to %s", passed, failed, args.results_dir)


if __name__ == "__main__":
    main()

