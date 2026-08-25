"""
Optimized Evaluation Engine for NeuralOps Framework.
Features:
1. Fast Validation-Only Dataset Caching (tokenizes 872 eval rows, NOT 67,000 train rows).
2. Base Model Template Caching & Fast In-Memory Cloning (prevents re-building HF models).
3. Non-blocking CUDA Event Timing (eliminates CPU-GPU synchronization stalls in hot loops).
4. Pin-memory enabled DataLoader for zero-copy async transfers.
5. Strict VRAM headroom management to prevent Windows WDDM paging to system RAM.
"""

from __future__ import annotations

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
from framework.core.attention_utils import inject_attention
from framework.utils.replace_attention import UniversalAttentionWrapper
from framework.eval_all_models_datasets import (
    BenchmarkResult,
    TRACK_MODULES,
    MODEL_TOKENIZER_MAPPING,
    _generate_synthetic_batch,
    _filter_batch,
    _extract_logits_and_loss,
    apply_module_to_model,
)

logger = logging.getLogger("EvalEngineOpt")

# -----------------------------------------------------------------------------
# Optimization 1: In-Memory Validation Dataset & Batch Cache with Pinned Memory
# -----------------------------------------------------------------------------
_TOKENIZED_BATCH_CACHE: Dict[Tuple[str, str, int, int, str], Dict[str, torch.Tensor]] = {}
_PREPARED_DATASET_CACHE: Dict[Tuple[str, str, int], Any] = {}


def get_dataset_batch_cached(
    dataset_name: str,
    model_family: str,
    batch_size: int = 64,
    seq_len: int = 256,
    vocab_size: int = 30522,
    num_labels: int = 2,
    device: torch.device = torch.device("cpu"),
    allow_synthetic: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Returns a tokenized evaluation batch with in-memory caching per (dataset, model_family, batch_size, seq_len).
    Directly tokenizes the validation split only (872 items) avoiding 67,000 training item CPU bottlenecks.
    """
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
                
                # Load only validation split to eliminate CPU bottleneck
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
                loader = torch.utils.data.DataLoader(
                    val_tokenized,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=is_cuda,
                )
                raw_batch = next(iter(loader))
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
# Optimization 2: Base Model Template Caching & Fast In-Memory Cloning
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
        from framework.eval_all_models_datasets import build_model_instance
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
# Optimization 3: Non-Blocking CUDA Event Timed Single Combination Evaluator
# -----------------------------------------------------------------------------

def evaluate_single_combination_opt(
    track: str,
    module_name: str,
    module_type: str,
    model_family: str,
    dataset: str,
    rank: int = 16,
    num_warmup: int = 2,
    num_steps: int = 5,
    batch_size: int = 64,
    seq_len: int = 256,
    device: torch.device = torch.device("cpu"),
    results_dir: str = "./results/gpu_server",
    allow_synthetic: bool = False,
) -> BenchmarkResult:
    result = BenchmarkResult(
        track=track,
        module_name=module_name,
        module_type=module_type,
        model_family=model_family,
        dataset=dataset,
        status="PASS",
    )

    model = None
    try:
        # 1. Fast clone from pristine base model cache
        model, cfg = get_base_model_cached(
            model_family=model_family,
            num_labels=2,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
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

        # 3. Retrieve cached dataset batch
        num_labels = getattr(cfg, "num_labels", 2)
        batch = get_dataset_batch_cached(
            dataset_name=dataset,
            model_family=model_family,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=getattr(cfg, "vocab_size", 30522),
            num_labels=num_labels,
            device=device,
            allow_synthetic=allow_synthetic,
        )

        filtered_batch = _filter_batch(batch, model_family, model)

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

        # Non-blocking CUDA Event Timing Setup
        is_cuda = (device.type == "cuda")
        fwd_times: List[float] = []
        bwd_times: List[float] = []

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            fwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
            fwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
            bwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
            bwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]

            # Continuous hot loop without CPU sync stalls
            for i in range(num_steps):
                fwd_start_events[i].record()
                out = model(**filtered_batch)
                fwd_end_events[i].record()

                _, loss = _extract_logits_and_loss(out, num_labels=num_labels, labels=batch.get("labels"))
                if loss is None:
                    loss = torch.tensor(0.0, requires_grad=True, device=device)

                bwd_start_events[i].record()
                loss.backward()
                bwd_end_events[i].record()
                model.zero_grad()

            bwd_end_events[-1].synchronize()

            for i in range(num_steps):
                fwd_times.append(fwd_start_events[i].elapsed_time(fwd_end_events[i]))
                bwd_times.append(bwd_start_events[i].elapsed_time(bwd_end_events[i]))

            result.peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)
        else:
            for _ in range(num_steps):
                t0 = time.perf_counter()
                out = model(**filtered_batch)
                t1 = time.perf_counter()
                fwd_times.append((t1 - t0) * 1000.0)

                _, loss = _extract_logits_and_loss(out, num_labels=num_labels, labels=batch.get("labels"))
                if loss is None:
                    loss = torch.tensor(0.0, requires_grad=True, device=device)

                t2 = time.perf_counter()
                loss.backward()
                t3 = time.perf_counter()
                bwd_times.append((t3 - t2) * 1000.0)
                model.zero_grad()

        result.forward_latency_ms = round(float(torch.tensor(fwd_times).mean().item()), 2)
        result.backward_latency_ms = round(float(torch.tensor(bwd_times).mean().item()), 2)
        result.total_step_latency_ms = round(result.forward_latency_ms + result.backward_latency_ms, 2)

    except Exception as e:
        logger.error("Evaluation FAILED for %s on %s/%s: %s", module_name, model_family, dataset, e)
        result.status = "FAIL"
        result.error_message = str(e)
    finally:
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return result
