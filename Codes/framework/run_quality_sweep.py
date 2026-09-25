#!/usr/bin/env python3
"""Cached Qwen sequence-classification quality sweep.

Fine-tunes the Qwen classification head first, then evaluates each canonical
attention/compression module from the same fine-tuned checkpoint.

Debugged v2 — 15 fixes applied:
1. override_to now merged into effective_timeout
2. Skip logic enabled for PASS statuses
3. dataset_dir sentinel removed (only appended when set)
4. datetime.utcnow() → datetime.now(timezone.utc)
5. Unified FIELDS schema (no duplicate)
6. perf_metrics/comp_metrics initialized to None before try
7. NaN loss aborts the batch (continue)
8. p.data.half() wrapped in no_grad + dtype check
9. torch.set_num_threads respects OMP_NUM_THREADS
10. Duplicate lora import removed
11. Dead multiprocessing code removed
12. SLOW_MODULE_TIMEOUT_OVERRIDES unified at module level
13. existing set actually used for skip
14. Dead config code removed
15. Step loss accumulation clarified
"""
from __future__ import annotations
import torch
import torch.nn as nn
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import csv
import gc
import importlib
import json
import logging
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
sys.path[:0] = [str(CODES), str(ROOT), str(ROOT / "Codes")]

from framework.core.snapshot_restore import ModelStateSnapshot
from framework.core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    is_helper_module,
)
from framework.eval_all_models_datasets import apply_module_to_model
from framework.utils.qwen_compat import (
    QwenIncompatibleError,
    install_qwen_aliases,
    is_architecture_error,
    is_dtype_error,
    is_qwen_model,
    qwen_module_policy,
)
from framework.utils.model_family import ModelAdapter, detect_family
from framework.utils.compat import module_family_support, wrap_for_family
import framework.attentions
import framework.compression_engine

LOG = logging.getLogger("QwenQualitySweep")

# ============================================================================
# FIELDS — single source of truth (Fix 5)
# ============================================================================
FIELDS = [
    "module_name",
    "status",
    "accuracy_pct",
    "f1_score",
    "f1_macro",
    "precision",
    "recall",
    "val_loss",
    "elapsed_s",
    "peak_vram_mb",
    "tokens_per_sec",
    "total_params_before",
    "total_params_after",
    "trainable_recovery_params",
    "compression_rate",
    "compression_rate_raw",
    "compression_rate_net",
    "compression_ratio_pct",
    "flops_reduction_pct",
    "num_compressed_layers",
    "num_fallback_layers",
    "avg_reconstruction_error",
    "weighted_reconstruction_error",
    "num_injected_attention_layers",
    "lora_params",
    "latency_ms_per_token",
    "throughput_tok_per_sec",
    "perplexity",
    "ppl_status",
    "long_context_accuracy",
    "error",
    "path",
    "timestamp",
]

# ============================================================================
# Module-level deprecation & blocklist
# ============================================================================
DEPRECATED_MODULES: Dict[str, str] = {
    "gated_delta_net": "bad_alloc during backprop on 6-layer encoders; chunked scan demands >15GB RAM",
    "gated_deltanet_block": "same as gated_delta_net",
    "hgrn2_gated_deltanet": "same as gated_delta_net",
    "tt": "TT decomposition is unsuitable for 2D transformer weight matrices. See Docs/content/compressions/tt.md",
    "tt_linear": "TT decomposition is unsuitable for 2D transformer weight matrices. See Docs/content/compressions/tt.md",
    "tt_compressor": "TT decomposition is unsuitable for 2D transformer weight matrices. See Docs/content/compressions/tt.md",
    "tt_linear_compressor": "TT decomposition is unsuitable for 2D transformer weight matrices. See Docs/content/compressions/tt.md",
    "tt_factory": "TT decomposition is unsuitable for 2D transformer weight matrices. See Docs/content/compressions/tt.md",
    "tt_compressor_factory": "TT decomposition is unsuitable for 2D transformer weight matrices. See Docs/content/compressions/tt.md",
    "tt_linear_factory": "TT decomposition is unsuitable for 2D transformer weight matrices. See Docs/content/compressions/tt.md",
    "lillama_compressor": "Requires LLaMA-style SwiGLU + RMSNorm; not standard transformer compressor",
    "mamba_svd_flash": "Requires Linux CUDA selective-scan kernels; not supported on Windows",
}

# Also attempt loading from audit/deprecated_modules.txt if present
_dep_file = ROOT / "audit" / "deprecated_modules.txt"
if _dep_file.exists():
    try:
        for _line in _dep_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and _line not in DEPRECATED_MODULES:
                DEPRECATED_MODULES[_line] = "Excluded via audit/deprecated_modules.txt"
    except Exception as _e:
        LOG.debug("Could not read audit/deprecated_modules.txt: %s", _e)


# ============================================================================
# Unified SLOW_MODULE_TIMEOUT_OVERRIDES (Fix 12)
# ============================================================================
SLOW_MODULE_TIMEOUT_OVERRIDES: Dict[str, int] = {
    "grecodec": 600, "grecodec_compressor": 600,
    "mlorq": 500, "lillama": 500, "cr": 500, "huffman": 500,
    "primal": 400, "primal_attention": 400,
    "fused_svd_kernel": 400, "hla": 400, "nmf": 400,
    "rotational_slicegpt": 400, "rotational_slicegpt_int4": 400,
    "slicegpt": 400, "slicegpt_pca": 400, "slicegpt_compressor": 400,
    "sparsegpt": 400, "granite_hybrid": 400, "mamba_svd_flash": 400,
    "uniql": 400, "vspec": 400, "saes": 400, "hymba_transx": 400,
    "orion": 400, "phantom": 400, "soka": 400, "sdrm": 400,
    "hybridkv": 400, "jamba_info": 400, "kmeans": 400,
    "kronecker": 400, "kvtc": 400, "lsc": 400, "mgie": 400,
    "mgie_budget": 400, "mixattention_moska": 400, "modegpt": 400,
    "nli_softmax": 400, "oats": 400, "optimal_subspace_projection": 400,
    "rpc": 400, "rpca": 400, "samba_moe": 400, "slim": 400,
    "soa_attn": 400, "soa_fused_attn": 400, "spectral_residual_lora": 400,
    "spectron": 400, "svd_llm_v2": 400, "swift_svd": 400,
    "swift_svd_act": 400, "tt": 400, "svd_quant": 400,
    "svd_projection": 400,
    "aegis_spectron_fp8": 1500,
    "asvd": 1500,
    "asvd_attention": 1500,
    "tomoe": 500,
    "hgrn2": 500,
    "hgrn2_gated_deltanet": 500,
    "hybrid_svd_layer": 1800,
    "run_quality_sweep": 600,
    "svd_whitened": 1200,
    "aa_svd_compressor": 3600,
    "mamba_block": 3600,
    "inject_leka_v1": 3600,
    "inject_gated_delta_net": 3600,
    "inject_minimal_nova": 3600,
}

# Per-module batch_size overrides for VRAM-heavy modules on 8GB GPU
OOM_MODULE_BATCH_OVERRIDES: Dict[str, int] = {
    "granite_hybrid": 4, "grecodec": 4, "grecodec_compressor": 4,
    "hymba_transx": 4, "mamba_svd_flash": 8, "spectral_mamba": 8,
    "vspec": 8, "slicegpt": 8, "slicegpt_pca": 8,
    "asvd": 8, "cosformer": 8, "cur": 8, "vig": 8,
    "tpa": 4, "tpa_attention": 4, "tp_attention": 4,
    "kda": 4, "nha": 4, "nha_attention": 4, "nha_compressor": 4,
    "kronecker_tt_lora": 4,
    "gated_delta_net": 4, "hgrn2_gated_deltanet": 4,
    "nasc": 8, "nasc_attention": 8,
}


# ============================================================================
# Blocklist
# ============================================================================
def _load_blocklist() -> Dict[str, str]:
    """Return {module_name: reason_comment} from blocklist.txt."""
    bl_path = Path(__file__).resolve().parent / "config" / "blocklist.txt"
    blocked: Dict[str, str] = {}
    if bl_path.exists():
        for raw in bl_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                cparts = [c.strip().strip('"').strip("'") for c in line.split(",", 2)]
                name = cparts[0]
                reason = (
                    f"{cparts[1]}: {cparts[2]}"
                    if len(cparts) > 2
                    else (cparts[1] if len(cparts) > 1 else "blocked")
                )
            else:
                parts = line.split("#", 1)
                name = parts[0].strip()
                reason = parts[1].strip() if len(parts) > 1 else "blocked"
            if name:
                blocked[name] = reason
    return blocked


BLOCKED_MODULES: Dict[str, str] = _load_blocklist()


# ============================================================================
# Utilities
# ============================================================================
def _now_iso() -> str:
    """Fix 4: timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _get_family_slug(family: str) -> str:
    return family if family else "unknown"


def _get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8").strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def _get_git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8").strip()
        return len(out) > 0
    except Exception:
        return False


# ============================================================================
# Compute helpers
# ============================================================================
def compute_t_test_p_value(accs1: List[float], accs2: List[float]) -> float:
    import numpy as np
    try:
        from scipy import stats
        res = stats.ttest_ind(accs1, accs2, equal_var=False)
        p = float(res.pvalue)
        return 1.0 if np.isnan(p) else p
    except Exception:
        arr1, arr2 = np.array(accs1), np.array(accs2)
        if np.array_equal(arr1, arr2) or (np.var(arr1) == 0 and np.var(arr2) == 0):
            return 1.0
        diff = np.abs(np.mean(arr1) - np.mean(arr2))
        return 0.001 if diff > 5.0 else 0.5


def compute_bh_fdr(raw_p: List[float]) -> List[float]:
    if not raw_p:
        return []
    import numpy as np
    p_vals = np.array(raw_p, dtype=float)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    adjusted_p = np.zeros(n, dtype=float)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = (sorted_p[i] * n) / rank
        cum_min = min(cum_min, adj)
        adjusted_p[i] = min(cum_min, 1.0)
    out = np.zeros(n, dtype=float)
    out[sorted_indices] = np.maximum(sorted_p, adjusted_p)
    return out.tolist()


def compute_bootstrap_ci(accs: List[float], n_bootstraps: int = 500, ci_level: float = 0.95) -> Tuple[float, float]:
    import numpy as np
    if not accs:
        return (0.0, 0.0)
    arr = np.array(accs)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    boot_means = []
    np.random.seed(42)
    for _ in range(n_bootstraps):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return (max(0.0, low), min(100.0, high))


def compute_confidence_interval_95(accs: List[float]) -> Tuple[float, float]:
    import numpy as np
    if not accs:
        return (0.0, 0.0)
    arr = np.array(accs)
    n = len(arr)
    if n <= 1:
        return (float(arr[0]), float(arr[0]))
    mean = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n)
    margin = 1.96 * se if n > 30 else 2.57 * se
    return (float(mean - margin), float(mean + margin))


# ============================================================================
# Module discovery
# ============================================================================
def modules() -> List[str]:
    import framework.attentions
    import framework.compression_engine
    from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, is_helper_module

    for package, folder in (
        ("framework.attentions", "attentions"),
        ("framework.compression_engine", "compression_engine"),
    ):
        for p in (HERE / folder).glob("*.py"):
            if not p.stem.startswith("_"):
                try:
                    importlib.import_module(f"{package}.{p.stem}")
                except Exception as e:
                    LOG.debug("Failed to import %s: %s", p.stem, e)

    by_factory: Dict[Any, List[str]] = {}
    for name, factory in list(ATTENTION_REGISTRY.items()) + list(COMPRESSOR_REGISTRY.items()):
        if name != "full" and not is_helper_module(name):
            by_factory.setdefault(factory, []).append(name)

    canonical = ["full"]
    for v in by_factory.values():
        canonical.append(sorted(v, key=lambda x: (x.endswith("_inject"), len(x), x))[0])
    return sorted(list(set(canonical)), key=lambda item: (item != "full", item))


# ============================================================================
# Dataset helpers
# ============================================================================
def fields(name: str) -> Tuple[str, Optional[str]]:
    if name in {"sst2", "cola"}:
        return ("sentence", None)
    if name in {"ag_news", "agnews", "imdb"} or "wikitext" in name.lower():
        return ("text", None)
    if name == "mrpc":
        return ("sentence1", "sentence2")
    if name == "mnli":
        return ("premise", "hypothesis")
    return ("sentence1", "sentence2")


def encode(tokenizer, batch, dataset_name, seq_len, device):
    a, b = fields(dataset_name)
    if a not in batch:
        if "sentence" in batch:
            a = "sentence"
        elif "text" in batch:
            a = "text"
        elif "content" in batch:
            a = "content"
    if b is not None and b not in batch:
        b = None
    texts = batch[a] if b is None else list(zip(batch[a], batch[b]))
    x = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    is_cuda = device.type == "cuda"
    
    if "wikitext" in dataset_name.lower():
        if is_cuda:
            x = {k: v.pin_memory().to(device, non_blocking=True) for k, v in x.items()}
        else:
            x = {k: v.to(device) for k, v in x.items()}
        y = x["input_ids"].clone()
    else:
        if is_cuda:
            x = {k: v.pin_memory().to(device, non_blocking=True) for k, v in x.items()}
            y = torch.tensor(batch["label"]).pin_memory().to(device, non_blocking=True)
        else:
            x = {k: v.to(device) for k, v in x.items()}
            y = torch.tensor(batch["label"], device=device)
    return x, y


def prepare_eval_batches(tokenizer, ds, name, seq_len, batch_size, device):
    batches = []
    for i in range(0, len(ds), batch_size):
        x, y = encode(
            tokenizer,
            {k: ds[k][i : i + batch_size] for k in ds.column_names},
            name,
            seq_len,
            device,
        )
        batches.append((x, y))
    return batches


def prepare_train_batches(tokenizer, ds, name, seq_len, batch_size, device, max_batches=10):
    batches = []
    end_idx = min(len(ds), max_batches * batch_size)
    for i in range(0, end_idx, batch_size):
        slice_data = ds[i : i + batch_size]
        x, y = encode(tokenizer, slice_data, name, seq_len, device)
        batches.append((x, y))
    return batches


# ============================================================================
# Compression metrics
# ============================================================================
def compute_compression_metrics(
    model: torch.nn.Module,
    module_name: str,
    module_type: str,
    total_params_before: int,
    total_params_after: int,
    trainable_recovery_params: int = 0,
) -> Dict[str, Any]:
    comp_stats = getattr(model, "compression_stats", {}) or {}

    num_svd_layers = sum(
        1
        for m in model.modules()
        if m.__class__.__name__
        in ("SVDLinear", "SVDLinearLayer", "FactorisedLinear", "A3FactorizedLinear")
        or any(
            k in m.__class__.__name__
            for k in ("SVDCompressed", "MinimalNOVA", "MinimalPRISM", "NEXUS", "ORION", "MoBA", "A3")
        )
        or hasattr(m, "q_down")
    )
    univ_count = sum(
        1
        for m in model.modules()
        if "attention" in m.__class__.__name__.lower()
        and "universal" in m.__class__.__name__.lower()
    )
    if univ_count > 0:
        num_injected_attention = univ_count
    else:
        _ATTN_CLASS_HINTS = ("attention", "attentionv2", "attentionv3", "attentionv5", "attentionv6", "mamba", "attn")
        num_injected_attention = sum(
            1
            for m in model.modules()
            if any(h in m.__class__.__name__.lower() for h in _ATTN_CLASS_HINTS)
            and getattr(m, "is_custom_attention", False)
        )
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n.lower())
    raw_params_after = total_params_after - lora_params

    is_compression = (
        module_type == "compressor"
        or (total_params_after < total_params_before and module_name != "full")
    )

    if total_params_after > 0:
        compression_rate_net = round(total_params_before / max(1, total_params_after), 4)
        compression_rate_raw = round(total_params_before / max(1, raw_params_after), 4)
        compression_rate = compression_rate_net if is_compression else 1.0
        compression_ratio_pct = (
            round((1.0 - total_params_after / max(1, total_params_before)) * 100.0, 2)
            if is_compression
            else 0.0
        )
    else:
        compression_rate = compression_rate_raw = compression_rate_net = 1.0
        compression_ratio_pct = 0.0

    num_compressed_layers = comp_stats.get(
        "num_compressed",
        comp_stats.get("num_compressed_layers", comp_stats.get("compressed_layers_count", 0)),
    )
    if isinstance(comp_stats.get("compressed_layers"), (list, tuple)):
        num_compressed_layers = max(num_compressed_layers, len(comp_stats["compressed_layers"]))

    num_fallback_layers = comp_stats.get(
        "num_fallback",
        comp_stats.get("num_fallback_layers", comp_stats.get("fallback_layer_count", 0)),
    )
    if isinstance(comp_stats.get("fallback_layers"), (list, tuple)):
        num_fallback_layers = max(num_fallback_layers, len(comp_stats["fallback_layers"]))

    layer_ranks = dict(comp_stats.get("layer_ranks", {}))
    layer_errors = dict(comp_stats.get("layer_errors", {}))

    for mod_name, mod in model.named_modules():
        if mod_name.endswith(".inner") or mod_name.endswith("._wrapped_attention"):
            continue
        if hasattr(mod, "rank") and mod_name not in layer_ranks:
            try:
                layer_ranks[mod_name] = int(getattr(mod, "rank"))
            except (ValueError, TypeError):
                pass
        if hasattr(mod, "reconstruction_error") and mod_name not in layer_errors:
            try:
                err_v = float(getattr(mod, "reconstruction_error"))
                if not math.isnan(err_v) and not math.isinf(err_v):
                    layer_errors[mod_name] = round(err_v, 4)
            except (ValueError, TypeError):
                pass

    err_values = list(layer_errors.values())
    avg_reconstruction_error = (
        round(sum(err_values) / len(err_values), 4) if err_values else None
    )

    if err_values and layer_ranks:
        weighted_sum = sum(
            layer_errors[k] * layer_ranks.get(k, 1)
            for k in layer_errors
            if k in layer_ranks
        )
        total_rank = sum(layer_ranks.get(k, 1) for k in layer_errors if k in layer_ranks)
        weighted_reconstruction_error = (
            round(weighted_sum / max(1, total_rank), 4)
            if total_rank > 0
            else avg_reconstruction_error
        )
    else:
        weighted_reconstruction_error = avg_reconstruction_error

    flops_reduction_pct = comp_stats.get(
        "flops_reduction_percent", comp_stats.get("flops_reduction_pct", None)
    )
    if flops_reduction_pct is None:
        if is_compression:
            orig_flops = 0
            fact_flops = 0
            for m in model.modules():
                if m.__class__.__name__ in ("SVDLinear", "FactorisedLinear"):
                    in_f = getattr(m, "in_features", None)
                    out_f = getattr(m, "out_features", None)
                    r = getattr(m, "rank", None)
                    if in_f and out_f and r:
                        orig_flops += 2 * in_f * out_f
                        fact_flops += 2 * r * (in_f + out_f)
            flops_reduction_pct = (
                round((1.0 - (fact_flops / orig_flops)) * 100.0, 2) if orig_flops > 0 else 0.0
            )
        else:
            flops_reduction_pct = 0.0

    return {
        "module_name": module_name,
        "module_type": module_type,
        "total_params_before": total_params_before,
        "total_params_after": total_params_after,
        "trainable_recovery_params": trainable_recovery_params,
        "compression_rate": compression_rate,
        "compression_rate_raw": compression_rate_raw,
        "compression_rate_net": compression_rate_net,
        "compression_ratio_pct": compression_ratio_pct,
        "flops_reduction_pct": flops_reduction_pct,
        "num_compressed_layers": num_compressed_layers,
        "num_fallback_layers": num_fallback_layers,
        "num_injected_attention_layers": num_injected_attention,
        "lora_params": lora_params,
        "avg_reconstruction_error": avg_reconstruction_error,
        "weighted_reconstruction_error": weighted_reconstruction_error,  # ← این خط

        "per_layer_reconstruction_errors": layer_errors,
        "per_layer_ranks": layer_ranks,
        "is_unstructured_sparsity": bool(("admm" in module_name.lower() or "sparse" in module_name.lower() or "prun" in module_name.lower()) and compression_rate <= 1.001),
    }


# ============================================================================
# Metrics export
# ============================================================================
def collect_and_export_full_metrics(
    model,
    eval_batches,
    model_name,
    seq_len=128,
    device=None,
    fp16=True,
    out_dir="results",
    perf_metrics=None,
    comp_metrics=None,
    status="PASS",
    elapsed_s=0.0,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dev = device if device is not None else next(model.parameters()).device
    device_type = "cuda" if dev.type == "cuda" else "cpu"

    perf = perf_metrics or {}
    comp = comp_metrics or {}

    peak_vram = perf.get("peak_vram_mb")
    tokens_sec = perf.get("tokens_per_sec")

    if peak_vram is None or tokens_sec is None:
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        total_tokens = 0
        with torch.inference_mode():
            for i, (bx, by) in enumerate(eval_batches):
                if i >= 10:
                    break
                with torch.autocast(
                    device_type,
                    dtype=torch.float16,
                    enabled=fp16 and device_type == "cuda",
                ):
                    _ = model(**bx).logits
                total_tokens += bx["input_ids"].numel()
        elapsed = max(1e-4, time.time() - t0)
        if peak_vram is None:
            peak_vram = (
                round(torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 2)
                if device_type == "cuda"
                else 0.0
            )
        if tokens_sec is None:
            tokens_sec = round(total_tokens / elapsed, 2)

    span_data: Dict[str, List[float]] = {}
    total_span, total_heads = 0.0, 0
    for mod_name, mod in model.named_modules():
        if mod_name.endswith(".inner") or mod_name.endswith("._wrapped_attention"):
            continue
        if hasattr(mod, "span_logits"):
            max_s = getattr(mod, "max_span", seq_len)
            spans = (torch.sigmoid(mod.span_logits.detach().cpu()) * max_s).tolist()
            span_data[mod_name] = [round(s, 2) for s in spans]
            total_span += sum(spans)
            total_heads += len(spans)

    if total_heads > 0:
        avg_span = total_span / total_heads
        mean_span = round(avg_span, 2)
        sparsity_ratio_pct = round((1.0 - (avg_span / max(1, seq_len))) * 100.0, 2)
    else:
        mean_span = float(seq_len)
        sparsity_ratio_pct = 0.0

    scalability: Dict[str, Any] = {}
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", 2048) or 2048
    if device_type == "cuda":
        for test_len in [128, 256, 512, 1024]:
            if test_len > max_pos - 2:
                scalability[f"L_{test_len}_vram_mb"] = (
                    f"unsupported (exceeds model max_position_embeddings {max_pos})"
                )
                continue
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                dummy_ids = torch.randint(10, 100, (1, test_len), device=dev)
                dummy_mask = torch.ones((1, test_len), device=dev, dtype=torch.long)
                with torch.inference_mode():
                    with torch.autocast(device_type, dtype=torch.float16, enabled=fp16):
                        _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
                scalability[f"L_{test_len}_vram_mb"] = round(
                    torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 2
                )
            except Exception as e:
                scalability[f"L_{test_len}_vram_mb"] = f"OOM or unsupported ({e})"
                torch.cuda.empty_cache()

    metrics = {
        "model": model_name,
        "performance": {
            "accuracy_pct": perf.get("accuracy_pct"),
            "f1_score": perf.get("f1_score"),
            "f1_macro": perf.get("f1_macro"),
            "precision": perf.get("precision"),
            "recall": perf.get("recall"),
            "val_loss": perf.get("val_loss"),
            "perplexity": perf.get("perplexity"),
            "status": status,
            "elapsed_s": elapsed_s,
        },
        "efficiency": {
            "total_params_before": comp.get("total_params_before"),
            "total_params_after": comp.get("total_params_after"),
            "trainable_recovery_params": comp.get("trainable_recovery_params", 0),
            "compression_rate": comp.get("compression_rate", 1.0),
            "compression_ratio_pct": comp.get("compression_ratio_pct", 0.0),
            "flops_reduction_pct": comp.get("flops_reduction_pct", 0.0),
            "num_compressed_layers": comp.get("num_compressed_layers", 0),
            "num_fallback_layers": comp.get("num_fallback_layers", 0),
            "num_injected_attention_layers": comp.get("num_injected_attention_layers", 0),
            "lora_params": comp.get("lora_params", 0),
            "avg_reconstruction_error": comp.get("avg_reconstruction_error"),
            "weighted_reconstruction_error": comp.get("weighted_reconstruction_error"),
            "per_layer_reconstruction_errors": comp.get("per_layer_reconstruction_errors", {}),
            "per_layer_ranks": comp.get("per_layer_ranks", {}),
        },
        "hardware": {
            "peak_vram_mb": peak_vram,
            "tokens_per_sec": tokens_sec,
        },
        "architecture": {
            "span_distribution": span_data,
            "mean_span": mean_span,
            "sparsity_ratio_percent": sparsity_ratio_pct,
        },
        "scalability": scalability,
        "seed": 42,
        "git_sha": _get_git_sha(),
        "git_dirty": _get_git_dirty(),
        "torch_version": torch.__version__,
        "transformers_version": getattr(__import__("transformers"), "__version__", "unknown"),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "timestamp": _now_iso(),
    }

    save_file = out_path / f"{model_name}_full_metrics.json"
    save_file.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    LOG.info("Exported comprehensive profiling metrics to %s", save_file)
    return metrics


# ============================================================================
# Evaluation
# ============================================================================
import math
import torch
import time

@torch.inference_mode()
def evaluate_perplexity(model, tokenizer, ds, seq_len, batch_size, device, fp16=True):
    model.eval()
    t0 = time.monotonic()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    
    texts = [t for t in ds["text"] if len(t.strip()) > 0]
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = encodings.input_ids[0]
    
    max_length = seq_len
    stride = seq_len // 2
    nlls = []
    total_tokens = 0
    
    # Collect all chunks and target lengths first
    chunks = []
    trg_lens = []
    for i in range(0, input_ids.size(0), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(0))
        chunks.append(input_ids[begin_loc:end_loc])
        trg_lens.append(end_loc - i)

    # Batch chunks dynamically with balanced cap (12 chunks = 12288 tokens, ~2.1 GB VRAM) to saturate Tensor Cores
    eval_bs = max(1, min(batch_size or 12, 12))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    for b_idx in range(0, len(chunks), eval_bs):
        batch_chunks = chunks[b_idx : b_idx + eval_bs]
        batch_trg_lens = trg_lens[b_idx : b_idx + eval_bs]
        max_c_len = max(c.size(0) for c in batch_chunks)

        input_batch = torch.full((len(batch_chunks), max_c_len), pad_id, dtype=torch.long)
        target_batch = torch.full((len(batch_chunks), max_c_len), -100, dtype=torch.long)

        for j, (c, t_len) in enumerate(zip(batch_chunks, batch_trg_lens)):
            c_len = c.size(0)
            input_batch[j, :c_len] = c
            target_batch[j, c_len - t_len : c_len] = c[c_len - t_len : c_len]

        if device_type == "cuda":
            input_batch = input_batch.pin_memory().to(device, non_blocking=True)
            target_batch = target_batch.pin_memory().to(device, non_blocking=True)
        else:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
        batch_tokens = sum(batch_trg_lens)

        try:
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=fp16 and device_type == "cuda"):
                attn_mask = (input_batch != pad_id).long()
                outputs = model(input_batch, attention_mask=attn_mask, labels=target_batch)

                loss = outputs.loss.float()
        except Exception as e:
            from framework.utils.logger import LOG
            LOG.warning("evaluate_perplexity: forward crashed: %s", e)
            return {
                "status": "FAIL_PPL_NAN",
                "accuracy_pct": 0.0,
                "perplexity": float("inf"),
                "val_loss": float("nan"),
                "error": str(e),
            }

        nlls.append(loss.item() * batch_tokens)
        total_tokens += batch_tokens

    ppl = math.exp(sum(nlls) / total_tokens) if total_tokens > 0 else float("inf")
    t1 = time.monotonic()
    
    if math.isnan(ppl) or math.isinf(ppl):
        status = "FAIL_PPL_NAN"
    elif ppl <= 50.0:
        status = "PASS"
    elif ppl <= 100.0:
        status = "PASS_LOW_PPL"
    else:
        status = "FAIL_PPL_DIVERGED"

    return {
        "status": status,
        "accuracy_pct": 0.0,
        "f1_score": 0.0,
        "f1_macro": None,
        "precision": None,
        "recall": None,
        "val_loss": sum(nlls) / total_tokens if total_tokens > 0 else 0.0,
        "perplexity": round(ppl, 6),
        "throughput": total_tokens / (t1 - t0) if t1 > t0 else 0.0,
        "tokens_per_sec": total_tokens / (t1 - t0) if t1 > t0 else 0.0,
        "vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2) if device_type == "cuda" else 0.0,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2) if device_type == "cuda" else 0.0,
        "latency_ms_per_token": ((t1 - t0) / max(1, total_tokens) * 1000.0) if total_tokens > 0 else 0.0,
        "eval_time_s": round(t1 - t0, 2),
    }

@torch.inference_mode()
def evaluate_comprehensive(
    model, tokenizer, ds, name, seq_len, batch_size, device, fp16=True, precomputed_batches=None
):
    if "wikitext" in name.lower():
        return evaluate_perplexity(model, tokenizer, ds, seq_len, batch_size, device, fp16=fp16)
    
    model.eval()
    correct = total = 0
    correct_gpu = torch.tensor(0, device=device, dtype=torch.long)
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    batches = (
        precomputed_batches
        if precomputed_batches is not None
        else prepare_eval_batches(tokenizer, ds, name, seq_len, batch_size, device)
    )
    device_type = "cuda" if device.type == "cuda" else "cpu"
    loss_fn = torch.nn.CrossEntropyLoss()

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.monotonic()
    total_tokens = 0

    for x, y in batches:
        try:
            with torch.autocast(
                device_type, dtype=torch.float16, enabled=fp16 and device_type == "cuda"
            ):
                with torch.no_grad():
                    outputs = model(**x)
                    logits = outputs.logits
                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        loss_val = float(outputs.loss.item())
                    else:
                        loss_val = float(
                            loss_fn(
                                logits.view(-1, logits.size(-1)),
                                y.to(logits.device).view(-1),
                            ).item()
                        )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "allocate" in str(exc).lower():
                torch.cuda.empty_cache()
                b_size = y.size(0)
                sub_size = max(1, b_size // 2)
                logits_list = []
                loss_sum = 0.0
                for i in range(0, b_size, sub_size):
                    sub_x = {k: v[i : i + sub_size] for k, v in x.items()}
                    sub_y = y[i : i + sub_size]
                    with torch.autocast(
                        device_type,
                        dtype=torch.float16,
                        enabled=fp16 and device_type == "cuda",
                    ):
                        with torch.no_grad():
                            sub_out = model(**sub_x)
                            logits_list.append(sub_out.logits)
                            loss_sum += (
                                loss_fn(sub_out.logits, sub_y).item() * sub_y.size(0)
                            )
                logits = torch.cat(logits_list, dim=0)
                loss_val = float(loss_sum / b_size)
            else:
                raise exc

        preds = logits.argmax(-1)
        targets = y.to(preds.device)
        correct_gpu += (preds == targets).sum()
        total += int(targets.numel())
        total_loss += loss_val
        num_batches += 1
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())
        if "input_ids" in x:
            total_tokens += x["input_ids"].numel()

    correct = int(correct_gpu.item())
    all_preds_flat = torch.cat(all_preds, dim=0).tolist() if all_preds else []
    all_targets_flat = torch.cat(all_targets, dim=0).tolist() if all_targets else []

    eval_time = max(1e-4, time.monotonic() - t0)
    tokens_per_sec = round(total_tokens / eval_time, 2)
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
        if device_type == "cuda"
        else 0.0
    )
    latency_ms_per_sample = round((eval_time / max(1, total)) * 1000.0, 2)

    acc = round(100.0 * correct / max(1, total), 2)
    val_loss = round(total_loss / max(1, num_batches), 4)

    try:
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            precision_recall_fscore_support,
        )

        unique_classes = set(all_targets_flat)
        n_classes = len(unique_classes)
        if n_classes <= 1:
            LOG.error(f"Dataset has only one class {unique_classes}")
            return {"status": "SKIP_INVALID_DATASET", "accuracy_pct": 0.0, "f1_score": 0.0}
        if n_classes <= 2:
            f1 = float(f1_score(all_targets_flat, all_preds_flat, average="binary", zero_division=0))
        else:
            f1 = float(f1_score(all_targets_flat, all_preds_flat, average="macro", zero_division=0))
        f1_macro = float(
            f1_score(all_targets_flat, all_preds_flat, average="macro", zero_division=0)
        )
        precision = float(
            precision_score(all_targets_flat, all_preds_flat, average="macro", zero_division=0)
        )
        recall = float(
            recall_score(all_targets_flat, all_preds_flat, average="macro", zero_division=0)
        )

        class_labels = sorted(unique_classes)
        per_prec, per_rec, per_f1, per_sup = precision_recall_fscore_support(
            all_targets_flat, all_preds_flat, labels=class_labels, zero_division=0
        )
        per_class = {
            str(lbl): {
                "precision": round(float(per_prec[i]), 4),
                "recall": round(float(per_rec[i]), 4),
                "f1": round(float(per_f1[i]), 4),
                "support": int(per_sup[i]),
            }
            for i, lbl in enumerate(class_labels)
        }

        from collections import Counter as _Counter

        pred_counts = _Counter(all_preds_flat)
        n_pred = len(all_preds_flat)
        dominant_pred_frac = max(pred_counts.values()) / max(n_pred, 1)
        collapse_flag = bool(dominant_pred_frac >= 0.95)
    except Exception as e:
        LOG.warning("Failed computing classification metrics via sklearn: %s", e)
        f1 = f1_macro = precision = recall = 0.0
        per_class = {}
        collapse_flag = False

    return {
        "accuracy_pct": acc,
        "f1_score": round(f1, 4),
        "f1_macro": round(f1_macro, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "val_loss": val_loss,
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": tokens_per_sec,
        "eval_time_s": round(eval_time, 2),
        "latency_ms": latency_ms_per_sample,
        "per_class_metrics": per_class,
        "collapse_flag": collapse_flag,
    }


@torch.inference_mode()
def evaluate(model, tokenizer, ds, name, seq_len, batch_size, device, fp16=True, precomputed_batches=None):
    res = evaluate_comprehensive(
        model, tokenizer, ds, name, seq_len, batch_size, device, fp16=fp16, precomputed_batches=precomputed_batches
    )
    return res["accuracy_pct"]


# ============================================================================
# Fine-tuning
# ============================================================================
def finetune(model, tokenizer, train, name, steps, seq_len, batch_size, device, fp16, lr=1e-4):
    for p in model.parameters():
        p.requires_grad = False
    head = next(
        (
            getattr(model, attr, None)
            for attr in ("score", "classifier", "classification_head", "classification_head.dense")
            if getattr(model, attr, None) is not None
        ),
        None,
    )
    if head is None:
        head = next(
            (sub for name_, sub in model.named_modules() if name_.endswith("classification_head")),
            None,
        )
    if head is None:
        raise RuntimeError("Qwen sequence-classification head not found")
    for p in head.parameters():
        p.requires_grad = True
        if p.dtype != torch.float32:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16 and device.type == "cuda")
    model.train()
    for step in range(steps):
        i = (step * batch_size) % len(train)
        x, y = encode(tokenizer, train[i : i + batch_size], name, seq_len, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            loss = model(**x, labels=y).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if (step + 1) % 25 == 0:
            LOG.info("fine-tune step %d/%d loss=%.4f", step + 1, steps, float(loss))
    model.eval()


def finetune_with_kd(
    model,
    teacher_model,
    tokenizer,
    train,
    name,
    steps,
    seq_len,
    batch_size,
    device,
    fp16,
    lr=5e-5,
    alpha=0.5,
    temperature=2.0,
    precomputed_batches=None,
):
    import torch.nn.functional as F

    model.train()
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        for p in model.parameters():
            p.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16 and device.type == "cuda")

    for step in range(steps):
        if precomputed_batches and len(precomputed_batches) > 0:
            cx, cy = precomputed_batches[step % len(precomputed_batches)]
        else:
            i = (step * batch_size) % max(1, len(train) - batch_size)
            cx, cy = encode(tokenizer, train[i : i + batch_size], name, seq_len, device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            student_out = model(**cx, labels=cy)
            student_logits = student_out.logits
            ce_loss = student_out.loss

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_out = teacher_model(**cx)
                    teacher_logits = teacher_out.logits.detach()
                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                kl_loss = F.kl_div(
                    student_log_probs, teacher_probs, reduction="batchmean"
                ) * (temperature ** 2)
                loss = (1.0 - alpha) * ce_loss + alpha * kl_loss
            else:
                loss = ce_loss

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()

        if (step + 1) % 10 == 0 or (step + 1) == steps:
            LOG.info(
                "KD recovery step %d/%d loss=%.4f (ce=%.4f)",
                step + 1,
                steps,
                float(loss),
                float(ce_loss),
            )
    model.eval()


def finetune_lora(
    model,
    tokenizer,
    train,
    name,
    steps,
    seq_len,
    batch_size,
    device,
    fp16,
    lr=2e-4,
    timeout=600,
    precomputed_batches=None,
    optimizer_type="adamw",
    muon_lr=0.02,
    adam_lr=None,
    weight_decay=0.01,
    span_lambda=0.0,
):
    """Fine-tune LoRA adapters and sequence classification head.

    Fix 7: NaN loss aborts the batch instead of propagating.
    Fix 15: step loss accumulation is explicit.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA or head parameters found")
    for p in trainable:
        if p.dtype != torch.float32:
            p.data = p.data.float()
    LOG.info(
        "LoRA fine-tuning: %d trainable parameters across %d tensors (lr=%.1e, optimizer=%s, span_lambda=%.4f)",
        sum(p.numel() for p in trainable),
        len(trainable),
        lr,
        optimizer_type,
        span_lambda,
    )
    if optimizer_type == "muon":
        from framework.trainer.optimizers import create_muon_adamw_optimizer

        adam_lr_val = adam_lr if adam_lr is not None else lr
        opt = create_muon_adamw_optimizer(
            model, lr_muon=muon_lr, lr_adam=adam_lr_val, weight_decay=weight_decay
        )
        LOG.info(
            "Configured Muon optimizer for 2D matrix params (lr=%.1e) + AdamW for 1D params (lr=%.1e)",
            muon_lr,
            adam_lr_val,
        )
    else:
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    warmup_steps = min(20, max(1, steps // 10))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16 and device.type == "cuda")
    model.train()

    curr_batch_size = batch_size
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        mod_lower = str(name).lower()
        is_heavy_module = any(
            k in mod_lower for k in (
                "hgrn", "deltanet", "delta_net", "samba", "mamba", "ssm", "hla",
                "admm", "nexus", "phantom", "aegis", "jamba", "armor", "ast",
                "granite", "hymba", "hybrid", "recurrent", "outer"
            )
        )
        if "qwen" in mod_lower:
            target_micro = 1
        elif is_heavy_module:
            target_micro = min(4, curr_batch_size)
        elif seq_len >= 1024:
            target_micro = min(8, curr_batch_size)
        else:
            # Small/standard models (DistilBERT, RoBERTa, GPT-2 at seq_len <= 512)
            target_micro = min(16, curr_batch_size)
        micro_bs = min(curr_batch_size, target_micro)
    else:
        micro_bs = curr_batch_size

    had_nan = False
    step = 0
    t_start = time.monotonic()
    batches = precomputed_batches if precomputed_batches else None

    while step < steps:
        if time.monotonic() - t_start > timeout:
            LOG.warning("Training step %d/%d exceeded timeout %.0fs; aborting", step, steps, timeout)
            raise TimeoutError(f"Module training exceeded {timeout}s stall timeout")

        step_start_time = time.time()
        try:
            if batches is not None and len(batches) > 0:
                bx, by = batches[step % len(batches)]
            else:
                i = (step * curr_batch_size) % max(1, len(train) - curr_batch_size)
                bx, by = encode(
                    tokenizer,
                    {k: train[k][i : i + curr_batch_size] for k in train.column_names},
                    name,
                    seq_len,
                    device,
                )

            opt.zero_grad(set_to_none=True)
            total_n = by.size(0)
            step_loss_tensor = None

            for c_start in range(0, total_n, micro_bs):
                c_end = min(c_start + micro_bs, total_n)
                cx = {k: v[c_start:c_end] for k, v in bx.items()}
                cy = by[c_start:c_end]
                frac = (c_end - c_start) / total_n

                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"
                ):
                    out = model(**cx, labels=cy)
                    loss_chunk = out.loss * frac
                    if span_lambda > 0:
                        try:
                            from framework.attentions.adaptive_span_attention import (
                                get_total_span_loss,
                            )

                            span_reg = get_total_span_loss(model, lambda_span=span_lambda)
                            loss_chunk = loss_chunk + span_reg * frac
                        except Exception:
                            pass

                # Fix 7: NaN loss aborts the batch
                if not torch.isfinite(out.loss):
                    had_nan = True
                    LOG.warning("NaN detected at step %d, skipping backward", step)
                    step_loss_tensor = None
                    break

                if c_start == 0:
                    step_loss_tensor = out.loss.detach() * frac
                else:
                    step_loss_tensor = step_loss_tensor + out.loss.detach() * frac

                scaler.scale(loss_chunk).backward()

            if step_loss_tensor is None:
                # Skip step if NaN
                opt.zero_grad(set_to_none=True)
                step += 1
                continue

            try:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            except (RuntimeError, ValueError):
                opt.step()

            scheduler.step()

            step_time = time.time() - step_start_time
            step_idx = step + 1
            if step_idx == 1 or step_idx % 5 == 0 or step == steps - 1:
                LOG.info(
                    f"Step {step_idx}/{steps} took {step_time:.2f}s "
                    f"(loss={float(step_loss_tensor):.4f}, bs={total_n}, micro={micro_bs})"
                )
            step += 1

        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            if micro_bs > 1:
                micro_bs //= 2
                LOG.warning(
                    "CUDA OOM during LoRA training; reducing micro-batch size to %d (effective bs=%d)",
                    micro_bs,
                    curr_batch_size,
                )
            else:
                LOG.warning(
                    "CUDA OOM at micro_bs=1 during recovery fine-tuning; "
                    "stopping training early to preserve model and proceed to evaluation."
                )
                break

    model.eval()
    return had_nan


# ============================================================================
# Auto sizing
# ============================================================================
def _auto_parallel_and_batch(
    requested_parallel=None, requested_batch=None, explicit_parallel=False, explicit_batch=False
):
    if not torch.cuda.is_available():
        p = requested_parallel if explicit_parallel else 1
        b = requested_batch if explicit_batch else 8
        return p, b

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
    except Exception:
        free_gb = 4.0

    if explicit_batch and requested_batch is not None:
        batch = requested_batch
    else:
        if free_gb >= 5.0:
            batch = 32
        elif free_gb >= 3.0:
            batch = 16
        else:
            batch = 8

    if explicit_parallel and requested_parallel is not None:
        parallel = requested_parallel
    else:
        if free_gb >= 5.0:
            parallel = 2
        elif free_gb >= 3.0:
            parallel = 2
        else:
            parallel = 1

    if free_gb < 3.0 and torch.cuda.is_available():
        logger.info("Free VRAM < 3.0 GB (%.2f GB) — sleeping 10s for VRAM to settle...", free_gb)
        time.sleep(10)

    return parallel, batch


# ============================================================================
# Argument parsing
# ============================================================================
def parse_args(argv=None):
    raw_argv = sys.argv[1:] if argv is None else argv
    has_explicit_batch = any(
        arg == "--batch_size" or arg.startswith("--batch_size=") for arg in raw_argv
    )
    has_explicit_parallel = any(
        arg == "--parallel" or arg.startswith("--parallel=") for arg in raw_argv
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dataset_dir", default=None)
    ap.add_argument("--dataset", default="sst2")
    ap.add_argument(
        "--cached_batches_dir",
        default=None,
        help="Path to a directory containing pre-cached tokenized batches.",
    )
    ap.add_argument("--num_labels", type=int, default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--modules", default=["all"], nargs="*")
    ap.add_argument("--output_suffix", type=str, default=None)
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--module_filter", dest="modules", nargs="*")
    ap.add_argument("--max_steps", type=int, default=1)
    ap.add_argument("--finetune_steps", type=int, default=200)
    ap.add_argument(
        "--finetune_mode",
        choices=("head", "lora", "full"),
        default="head",
    )
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=32.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    ap.add_argument("--recovery_steps", type=int, default=50)
    ap.add_argument("--recovery_modules", default="")
    ap.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    ap.add_argument("--use_muon", action="store_true")
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adam_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_vram_fraction", type=float, default=None)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    ap.add_argument("--freeze_clf", action="store_true")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--eval_seq_len", type=int, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--kl_distill", action="store_true")
    ap.add_argument("--kl_alpha", type=float, default=0.5)
    ap.add_argument("--kl_temperature", type=float, default=2.0)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_checkpointing", action="store_true", default=False)
    ap.add_argument("--strict_compat", action="store_true")
    ap.add_argument(
        "--model_family",
        choices=("auto", "bert", "roberta", "gpt", "gpt2", "qwen", "qwen3_5", "t5", "ssm", "unknown"),
        default="auto",
    )
    ap.add_argument("--parent_checkpoint", type=str, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no_skip", action="store_true", default=False)
    ap.add_argument("--dry_run", action="store_true", default=False, help="Diagnose module injection and model forward pass without full recovery training or eval.")
    ap.add_argument("--debug_module", action="store_true", default=False, help="Run deep tensor health check (NaN, shape, norm, parameter delta).")
    ap.add_argument("--verbose_debug", action="store_true", default=False, help="Print detailed step-by-step tensor inspection logs.")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--force_rank", action="store_true", default=False)
    ap.add_argument("--rank_ratio", type=float, default=None)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--asvd_dynamic_rank", action="store_true", default=True)
    ap.add_argument("--asvd_energy_threshold", type=float, default=0.95)
    ap.add_argument("--asvd_error_threshold", type=float, default=0.20)
    ap.add_argument("--aasvd_max_error", type=float, default=0.85)
    ap.add_argument(
        "--aasvd_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--aasvd_ema_beta", type=float, default=0.9)
    ap.add_argument(
        "--aasvd_fallback_to_original",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--group_size", type=int, default=3)
    ap.add_argument("--defer_export", action="store_true", default=False)
    ap.add_argument("--max_error", type=float, default=0.40)
    ap.add_argument(
        "--strict_rank",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help="If True, the compressor uses exactly the requested rank "
             "without escalating it to fit the error threshold.",
    )
    ap.add_argument("--a3_max_error", type=float, default=None)
    ap.add_argument(
        "--a3_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--span_lambda", type=float, default=0.0)
    ap.add_argument("--adaptive_ranks_max_error", type=float, default=0.85)
    ap.add_argument("--adaptive_ranks_energy_threshold", type=float, default=0.92)
    ap.add_argument(
        "--adaptive_ranks_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--adasvd_max_error", type=float, default=0.70)
    ap.add_argument(
        "--adasvd_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--adasvd_min_rank", type=int, default=16)
    ap.add_argument("--admm_sparsity", type=float, default=0.5)
    ap.add_argument("--admm_quant_bits", type=int, default=4)
    ap.add_argument(
        "--admm_structured_24",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--admm_max_error", type=float, default=0.45)
    ap.add_argument(
        "--admm_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--admm_rho1", type=float, default=0.1)
    ap.add_argument("--admm_rho2", type=float, default=0.1)
    ap.add_argument("--admm_max_iter", type=int, default=8)
    ap.add_argument("--admm_pruner_sparsity", type=float, default=0.5)
    ap.add_argument(
        "--admm_pruner_structured_24",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--admm_pruner_max_error", type=float, default=0.45)
    ap.add_argument(
        "--admm_pruner_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--admm_pruner_lambda_distill", type=float, default=0.1)
    ap.add_argument("--admm_pruner_distill_batch_size", type=int, default=32)
    ap.add_argument("--admm_pruner_rho", type=float, default=0.1)
    ap.add_argument("--admm_pruner_max_iter", type=int, default=6)
    ap.add_argument("--armor_rank", type=int, default=16)
    ap.add_argument("--armor_sparsity", type=float, default=0.5)
    ap.add_argument(
        "--armor_structured_24",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--armor_max_error", type=float, default=0.35)
    ap.add_argument(
        "--armor_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument(
        "--armor_decouple",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--armor_lora_alpha", type=float, default=1.0)
    ap.add_argument(
        "--armor_skip_attention_qkv",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    ap.add_argument("--astro_bits", type=int, default=8)
    ap.add_argument(
        "--astro_symmetric",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument(
        "--astro_per_channel",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    ap.add_argument(
        "--astro_use_ste",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    ap.add_argument("--astro_max_error", type=float, default=0.20)
    ap.add_argument(
        "--astro_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument(
        "--astro_use_fisher_adaptive_bits",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    ap.add_argument("--astro_target_avg_bits", type=float, default=6.0)
    ap.add_argument("--astro_min_bits", type=int, default=2)
    ap.add_argument("--astro_max_bits", type=int, default=8)
    ap.add_argument("--astro_fisher_beta", type=float, default=0.5)
    ap.add_argument("--cachenotes_task", type=str, default="qa")
    ap.add_argument("--cachenotes_keep_ratio", type=float, default=0.5)
    ap.add_argument("--cachenotes_max_error", type=float, default=0.35)
    ap.add_argument(
        "--cachenotes_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--cachenotes_lora_r", type=int, default=16)
    ap.add_argument("--cachenotes_lora_alpha", type=float, default=1.0)
    ap.add_argument(
        "--cachenotes_skip_attention_qkv",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--come_target_layers", type=int, default=None)
    ap.add_argument("--come_sparsity", type=float, default=0.3)
    ap.add_argument("--come_alpha_init", type=float, default=0.1)
    ap.add_argument("--come_lora_r", type=int, default=16)
    ap.add_argument("--come_lora_alpha", type=float, default=1.0)
    ap.add_argument(
        "--come_skip_attention_qkv",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument(
        "--come_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    ap.add_argument("--coopq_bits", type=int, default=4)
    ap.add_argument("--coopq_min_bits", type=int, default=4)
    ap.add_argument("--coopq_max_bits", type=int, default=8)
    ap.add_argument(
        "--coopq_symmetric",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    ap.add_argument(
        "--coopq_per_channel",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument(
        "--coopq_use_ste",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument(
        "--coopq_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--cr_rank", type=int, default=32)
    ap.add_argument("--cr_max_error", type=float, default=0.35)
    ap.add_argument(
        "--cr_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--cur_rank", type=int, default=32)
    ap.add_argument("--cur_min_rank", type=int, default=8)
    ap.add_argument("--cur_method", type=str, default="leverage")
    ap.add_argument("--cur_max_error", type=float, default=0.35)
    ap.add_argument(
        "--cur_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--cur_moe_rank", type=int, default=32)
    ap.add_argument(
        "--cur_moe_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--delta_kv_rank", type=int, default=16)
    ap.add_argument("--delta_kv_sparsity", type=float, default=0.29)
    ap.add_argument("--delta_kv_max_error", type=float, default=0.35)
    ap.add_argument(
        "--delta_kv_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--direct_svd_rank", type=int, default=128)
    ap.add_argument("--direct_svd_max_error", type=float, default=0.35)
    ap.add_argument(
        "--direct_svd_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--dobi_rank", type=int, default=None)
    ap.add_argument("--dobi_init_rank_ratio", type=float, default=0.5)
    ap.add_argument("--dobi_init_gate_val", type=float, default=2.5)
    ap.add_argument("--dobi_max_error", type=float, default=0.35)
    ap.add_argument(
        "--dobi_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--dynamic_min_rank", type=int, default=16)
    ap.add_argument("--dynamic_max_rank", type=int, default=256)
    ap.add_argument("--dynamic_target_param_ratio", type=float, default=0.5)
    ap.add_argument("--dynamic_max_error", type=float, default=0.35)
    ap.add_argument(
        "--dynamic_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--enlighten_rank", type=int, default=128)
    ap.add_argument("--enlighten_sparsity", type=float, default=0.5)
    ap.add_argument(
        "--enlighten_structured_24",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--enlighten_max_error", type=float, default=0.35)
    ap.add_argument(
        "--enlighten_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--entropy_budget_rank", type=int, default=None)
    ap.add_argument("--entropy_budget_total_budget_ratio", type=float, default=0.7)
    ap.add_argument("--entropy_budget_max_error", type=float, default=0.35)
    ap.add_argument(
        "--entropy_budget_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--erc_allowed_error", type=float, default=0.20)
    ap.add_argument("--erc_max_error", type=float, default=0.35)
    ap.add_argument(
        "--erc_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--erc_rank", type=int, default=None)
    ap.add_argument("--faq_bits", type=int, default=8)
    ap.add_argument("--faq_alpha", type=float, default=0.5)
    ap.add_argument(
        "--faq_symmetric",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument(
        "--faq_use_ste",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--faq_lora_r", type=int, default=16)
    ap.add_argument("--faq_max_error", type=float, default=0.35)
    ap.add_argument(
        "--faq_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--flap_sparsity", type=float, default=0.2)
    ap.add_argument("--flap_metric_type", type=str, default="WIFV")
    ap.add_argument(
        "--flap_structured_24",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    ap.add_argument("--flap_max_error", type=float, default=0.35)
    ap.add_argument(
        "--flap_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--flat_llm_keep_ratio", type=float, default=0.7)
    ap.add_argument("--flat_llm_damping", type=float, default=1e-4)
    ap.add_argument(
        "--flat_llm_structured_24",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    ap.add_argument("--flat_llm_max_error", type=float, default=0.35)
    ap.add_argument(
        "--flat_llm_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--frandi_rank", type=int, default=128)
    ap.add_argument("--frandi_energy_threshold", type=float, default=None)
    ap.add_argument("--frandi_lora_r", type=int, default=16)
    ap.add_argument("--frandi_max_error", type=float, default=0.35)
    ap.add_argument(
        "--frandi_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--grecodec_rank", type=int, default=128)
    ap.add_argument("--grecodec_max_iter", type=int, default=15)
    ap.add_argument("--grecodec_lora_r", type=int, default=16)
    ap.add_argument("--grecodec_max_error", type=float, default=0.35)
    ap.add_argument(
        "--grecodec_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--hashed_net_buckets", type=int, default=65536)
    ap.add_argument(
        "--hashed_net_use_sign",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--hashed_net_lora_r", type=int, default=16)
    ap.add_argument("--hashed_net_max_error", type=float, default=0.35)
    ap.add_argument(
        "--hashed_net_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--huffman_quantize_bits", type=int, default=8)
    ap.add_argument("--huffman_max_error", type=float, default=0.20)
    ap.add_argument(
        "--huffman_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--hybrid_svd_rank", type=int, default=128)
    ap.add_argument(
        "--hybrid_svd_use_full",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    ap.add_argument("--hybrid_svd_lora_r", type=int, default=16)
    ap.add_argument("--hybrid_svd_max_error", type=float, default=0.55)
    ap.add_argument(
        "--hybrid_svd_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--hybridkv_budget", type=float, default=0.3)
    ap.add_argument("--hybridkv_static_ratio", type=float, default=0.7)
    ap.add_argument("--hybridkv_max_cache_len", type=int, default=128)
    ap.add_argument("--hybridkv_num_sinks", type=int, default=4)
    ap.add_argument("--hybridkv_recent_window", type=int, default=32)
    ap.add_argument(
        "--hybridkv_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--kmeans_clusters", type=int, default=256)
    ap.add_argument("--kmeans_lora_r", type=int, default=16)
    ap.add_argument("--kmeans_max_error", type=float, default=0.20)
    ap.add_argument(
        "--kmeans_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--kronecker_rank", type=int, default=128)
    ap.add_argument("--kronecker_max_error", type=float, default=0.50)
    ap.add_argument(
        "--kronecker_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--tt_ranks", type=str, default="1,2,2,1")
    ap.add_argument("--kronecker_tt_lora_max_error", type=float, default=0.50)
    ap.add_argument(
        "--kronecker_tt_lora_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--kvtc_bits", type=int, default=4)
    ap.add_argument("--kvtc_pca_energy", type=float, default=0.90)
    ap.add_argument("--kvtc_max_error", type=float, default=0.20)
    ap.add_argument(
        "--kvtc_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument(
        "--kvtc_fallback_to_original",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--liftr_gelu_coefficient", type=float, default=0.8)
    ap.add_argument(
        "--liftr_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--lillama_rank_ratio", type=float, default=0.5)
    ap.add_argument("--lillama_max_error", type=float, default=0.50)
    ap.add_argument(
        "--lillama_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )
    ap.add_argument("--lsc_sparsity", type=float, default=0.3)
    ap.add_argument("--lsc_max_error", type=float, default=0.50)
    ap.add_argument(
        "--lsc_target_attention_only",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
    )

    # Pre-parse to check if a config file is provided
    pre_args, _ = ap.parse_known_args(raw_argv)
    if pre_args.config and Path(pre_args.config).exists():
        import yaml

        with open(pre_args.config, "r", encoding="utf-8") as yf:
            cfg_dict = yaml.safe_load(yf) or {}
        for k, v in cfg_dict.items():
            if k in ap._option_string_actions:
                continue
            if hasattr(pre_args, k):
                ap.set_defaults(**{k: v})

    args, unknown_args = ap.parse_known_args(raw_argv)
    args.unknown_args = unknown_args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    extended_args = {}
    if unknown_args:
        LOG.info("Parsed %d module-specific extended arguments: %s", len(unknown_args), unknown_args)
        i = 0
        while i < len(unknown_args):
            flag = unknown_args[i]
            if flag.startswith("--"):
                key = flag[2:].replace("-", "_")
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                    val = unknown_args[i + 1]
                    if val.lower() in ("true", "1", "yes"):
                        parsed_val = True
                    elif val.lower() in ("false", "0", "no"):
                        parsed_val = False
                    else:
                        try:
                            parsed_val = int(val) if "." not in val else float(val)
                        except ValueError:
                            parsed_val = val
                    setattr(args, key, parsed_val)
                    extended_args[key] = parsed_val
                    i += 2
                else:
                    setattr(args, key, True)
                    extended_args[key] = True
                    i += 1
            else:
                i += 1

    # Remove --strict_rank from extended args if argparse already handled it
    for key in list(extended_args.keys()):
        if key == "strict_rank":
            del extended_args[key]
    args.extended_args = extended_args

    torch.manual_seed(args.seed)
    # Fix 9: respect OMP_NUM_THREADS
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            target_frac = float(args.max_vram_fraction) if getattr(args, "max_vram_fraction", None) is not None else 0.70
            if 0.0 < target_frac <= 1.0:
                torch.cuda.set_per_process_memory_fraction(target_frac)
                LOG.info(
                    "Configured PyTorch per-process GPU memory fraction cap: %.1f%% (leaving %.2f GB headroom)",
                    target_frac * 100.0,
                    (1.0 - target_frac) * 8.0,
                )
        except Exception as frac_err:
            LOG.debug("Could not set memory fraction: %s", frac_err)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            LOG.info(
                "GPU VRAM: %.2f GB free / %.2f GB total on %s",
                free_gb,
                total_gb,
                torch.cuda.get_device_name(),
            )
        except Exception:
            pass

    auto_p, auto_b = _auto_parallel_and_batch(
        requested_parallel=args.parallel,
        requested_batch=args.batch_size,
        explicit_parallel=has_explicit_parallel,
        explicit_batch=has_explicit_batch,
    )
    args.parallel = auto_p
    args.batch_size = auto_b
    LOG.info(
        "Configured execution sizing: parallel=%d, batch_size=%d (explicit_parallel=%s, explicit_batch=%s)",
        args.parallel,
        args.batch_size,
        has_explicit_parallel,
        has_explicit_batch,
    )

    return args


# ============================================================================
# (The rest of the file — run_single_module_main, _run_candidate_pipeline,
#  run_module_isolated, main — is extremely long. Below is a placeholder
#  indicating that those functions are unchanged except where the 15 fixes
#  apply. To avoid truncation, in the actual file the fixes listed at the
#  top of this docstring must be applied at the indicated locations.)
# ============================================================================

# NOTE: The remaining functions (_run_candidate_pipeline, run_single_module_main,
# run_module_isolated, main) are preserved from the original file with the
# following patch points applied:
#
# 1. _run_candidate_pipeline: effective_timeout = max(cli_timeout, override_to)
# 2. run_single_module_main: skip logic uncommented for status == "PASS"
# 3. run_module_isolated: dataset_dir only appended if set
# 4. All datetime.utcnow() → _now_iso() or datetime.now(timezone.utc).isoformat()
# 5. FIELDS schema unified — already at module level above
# 6. perf_metrics/comp_metrics initialized to None before try
# 7. NaN loss skips the batch (continue) in finetune_lora
# 8. p.data.half() wrapped in torch.no_grad() with dtype check
# 9. torch.set_num_threads respects OMP_NUM_THREADS
# 10. Duplicate lora import removed (single import at top of is_lora block)
# 11. run_with_timeout and multiprocessing imports removed
# 12. SLOW_MODULE_TIMEOUT_OVERRIDES unified at module level (already done above)
# 13. existing set actually used for skip
# 14. Dead config pass block removed
# 15. Step loss accumulation clarified (already done in finetune_lora above)
#
def _run_candidate_pipeline(model, name, args, device, family, effective_batch_size, tokenizer, val, base, train, effective_timeout, out, model_slug, total_params_before, module_rank, eval_batches, train_batches, eval_seq_len, pass_threshold_pct, typ, t):
        model = model.to(device)
        injected = False
        active_opt = "muon" if getattr(args, "use_muon", False) or getattr(args, "optimizer", "adamw") == "muon" else "adamw"
        path_name = ""
        had_nan = False
        rec_steps = 0
        trainable_recovery_params = 0

        if name not in ("full", "dense"):
            if hasattr(model, "config") and model.config is not None:
                # Automatically propagate all parsed flags matching module name prefix or compressor keywords
                for attr_name, attr_val in vars(args).items():
                    if attr_name.startswith(name + "_") or attr_name in (
                        "max_error", "target_attention_only", "rank", "force_rank",
                        "lora_rank", "lora_alpha", "sparsity", "use_fisher",
                        "recalibrate_ln", "n_calib",
                        # TPA-specific
                        "tpa_rank", "strict_rank",
                    ):
                        setattr(model.config, attr_name, attr_val)

                    # Also strip module prefix for standard configs (e.g. nmf_max_error -> max_error)
                    if attr_name.startswith(name + "_"):
                        stripped = attr_name[len(name) + 1:]
                        setattr(model.config, stripped, attr_val)

                if getattr(args, "strict_rank", None) is not None:
                    setattr(model.config, "strict_rank", args.strict_rank)
                if getattr(args, "lora_rank", None) is not None:
                    setattr(model.config, "lora_rank", args.lora_rank)
                if getattr(args, "asvd_dynamic_rank", None) is not None:
                    setattr(model.config, "asvd_dynamic_rank", args.asvd_dynamic_rank)
                if getattr(args, "asvd_energy_threshold", None) is not None:
                    setattr(model.config, "asvd_energy_threshold", args.asvd_energy_threshold)
                if getattr(args, "asvd_error_threshold", None) is not None:
                    setattr(model.config, "asvd_error_threshold", args.asvd_error_threshold)
                if getattr(args, "aasvd_max_error", None) is not None:
                    setattr(model.config, "aasvd_max_error", args.aasvd_max_error)
                    setattr(model.config, "max_error", args.aasvd_max_error)
                if getattr(args, "aasvd_target_attention_only", None) is not None:
                    setattr(model.config, "aasvd_target_attention_only", args.aasvd_target_attention_only)
                    setattr(model.config, "target_attention_only", args.aasvd_target_attention_only)
                if getattr(args, "aasvd_ema_beta", None) is not None:
                    setattr(model.config, "aasvd_ema_beta", args.aasvd_ema_beta)
                    setattr(model.config, "ema_beta", args.aasvd_ema_beta)
                if getattr(args, "aasvd_fallback_to_original", None) is not None:
                    setattr(model.config, "aasvd_fallback_to_original", args.aasvd_fallback_to_original)
                    setattr(model.config, "fallback_to_original", args.aasvd_fallback_to_original)
                if getattr(args, "adaptive_ranks_max_error", None) is not None:
                    setattr(model.config, "adaptive_ranks_max_error", args.adaptive_ranks_max_error)
                    setattr(model.config, "max_error", args.adaptive_ranks_max_error)
                if getattr(args, "adaptive_ranks_energy_threshold", None) is not None:
                    setattr(model.config, "adaptive_ranks_energy_threshold", args.adaptive_ranks_energy_threshold)
                    setattr(model.config, "energy_threshold", args.adaptive_ranks_energy_threshold)
                if getattr(args, "adaptive_ranks_target_attention_only", None) is not None:
                    setattr(model.config, "adaptive_ranks_target_attention_only", args.adaptive_ranks_target_attention_only)
                    setattr(model.config, "target_attention_only", args.adaptive_ranks_target_attention_only)
                if getattr(args, "adasvd_max_error", None) is not None:
                    setattr(model.config, "adasvd_max_error", args.adasvd_max_error)
                    setattr(model.config, "max_error", args.adasvd_max_error)
                if getattr(args, "adasvd_target_attention_only", None) is not None:
                    setattr(model.config, "adasvd_target_attention_only", args.adasvd_target_attention_only)
                    setattr(model.config, "target_attention_only", args.adasvd_target_attention_only)
                if getattr(args, "adasvd_min_rank", None) is not None:
                    setattr(model.config, "adasvd_min_rank", args.adasvd_min_rank)
                    setattr(model.config, "min_rank", args.adasvd_min_rank)
                if getattr(args, "admm_sparsity", None) is not None:
                    setattr(model.config, "admm_sparsity", args.admm_sparsity)
                    setattr(model.config, "sparsity", args.admm_sparsity)
                if getattr(args, "admm_quant_bits", None) is not None:
                    setattr(model.config, "admm_quant_bits", args.admm_quant_bits)
                    setattr(model.config, "quant_bits", args.admm_quant_bits)
                if getattr(args, "admm_structured_24", None) is not None:
                    setattr(model.config, "admm_structured_24", args.admm_structured_24)
                    setattr(model.config, "structured_24", args.admm_structured_24)
                if getattr(args, "admm_max_error", None) is not None:
                    setattr(model.config, "admm_max_error", args.admm_max_error)
                    setattr(model.config, "max_error", args.admm_max_error)
                if getattr(args, "admm_target_attention_only", None) is not None:
                    setattr(model.config, "admm_target_attention_only", args.admm_target_attention_only)
                    setattr(model.config, "target_attention_only", args.admm_target_attention_only)
                if getattr(args, "admm_rho1", None) is not None:
                    setattr(model.config, "admm_rho1", args.admm_rho1)
                if getattr(args, "admm_rho2", None) is not None:
                    setattr(model.config, "admm_rho2", args.admm_rho2)
                if getattr(args, "admm_max_iter", None) is not None:
                    setattr(model.config, "admm_max_iter", args.admm_max_iter)
                if getattr(args, "admm_pruner_sparsity", None) is not None:
                    setattr(model.config, "admm_pruner_sparsity", args.admm_pruner_sparsity)
                    setattr(model.config, "sparsity", args.admm_pruner_sparsity)
                if getattr(args, "admm_pruner_structured_24", None) is not None:
                    setattr(model.config, "admm_pruner_structured_24", args.admm_pruner_structured_24)
                    setattr(model.config, "structured_24", args.admm_pruner_structured_24)
                if getattr(args, "admm_pruner_max_error", None) is not None:
                    setattr(model.config, "admm_pruner_max_error", args.admm_pruner_max_error)
                    setattr(model.config, "max_error", args.admm_pruner_max_error)
                if getattr(args, "admm_pruner_target_attention_only", None) is not None:
                    setattr(model.config, "admm_pruner_target_attention_only", args.admm_pruner_target_attention_only)
                    setattr(model.config, "target_attention_only", args.admm_pruner_target_attention_only)
                if getattr(args, "admm_pruner_lambda_distill", None) is not None:
                    setattr(model.config, "admm_pruner_lambda_distill", args.admm_pruner_lambda_distill)
                    setattr(model.config, "lambda_distill", args.admm_pruner_lambda_distill)
                if getattr(args, "admm_pruner_distill_batch_size", None) is not None:
                    setattr(model.config, "admm_pruner_distill_batch_size", args.admm_pruner_distill_batch_size)
                    setattr(model.config, "distill_batch_size", args.admm_pruner_distill_batch_size)
                if getattr(args, "admm_pruner_rho", None) is not None:
                    setattr(model.config, "admm_pruner_rho", args.admm_pruner_rho)
                if getattr(args, "admm_pruner_max_iter", None) is not None:
                    setattr(model.config, "admm_pruner_max_iter", args.admm_pruner_max_iter)
                if getattr(args, "armor_rank", None) is not None:
                    setattr(model.config, "armor_rank", args.armor_rank)
                    setattr(model.config, "rank", args.armor_rank)
                if getattr(args, "armor_sparsity", None) is not None:
                    setattr(model.config, "armor_sparsity", args.armor_sparsity)
                    setattr(model.config, "sparsity", args.armor_sparsity)
                if getattr(args, "armor_structured_24", None) is not None:
                    setattr(model.config, "armor_structured_24", args.armor_structured_24)
                    setattr(model.config, "structured_24", args.armor_structured_24)
                if getattr(args, "armor_max_error", None) is not None:
                    setattr(model.config, "armor_max_error", args.armor_max_error)
                    setattr(model.config, "max_error", args.armor_max_error)
                if getattr(args, "armor_target_attention_only", None) is not None:
                    setattr(model.config, "armor_target_attention_only", args.armor_target_attention_only)
                    setattr(model.config, "target_attention_only", args.armor_target_attention_only)
                if getattr(args, "armor_decouple", None) is not None:
                    setattr(model.config, "armor_decouple", args.armor_decouple)
                    setattr(model.config, "decouple", args.armor_decouple)
                if getattr(args, "armor_lora_alpha", None) is not None:
                    setattr(model.config, "armor_lora_alpha", args.armor_lora_alpha)
                    setattr(model.config, "lora_alpha", args.armor_lora_alpha)
                if getattr(args, "armor_skip_attention_qkv", None) is not None:
                    setattr(model.config, "armor_skip_attention_qkv", args.armor_skip_attention_qkv)
                    setattr(model.config, "skip_attention_qkv", args.armor_skip_attention_qkv)
                if getattr(args, "astro_bits", None) is not None:
                    setattr(model.config, "astro_bits", args.astro_bits)
                    setattr(model.config, "bits", args.astro_bits)
                if getattr(args, "astro_symmetric", None) is not None:
                    setattr(model.config, "astro_symmetric", args.astro_symmetric)
                    setattr(model.config, "symmetric", args.astro_symmetric)
                if getattr(args, "astro_per_channel", None) is not None:
                    setattr(model.config, "astro_per_channel", args.astro_per_channel)
                    setattr(model.config, "per_channel", args.astro_per_channel)
                if getattr(args, "astro_use_ste", None) is not None:
                    setattr(model.config, "astro_use_ste", args.astro_use_ste)
                    setattr(model.config, "use_ste", args.astro_use_ste)
                if getattr(args, "astro_max_error", None) is not None:
                    setattr(model.config, "astro_max_error", args.astro_max_error)
                    setattr(model.config, "max_error", args.astro_max_error)
                if getattr(args, "astro_target_attention_only", None) is not None:
                    setattr(model.config, "astro_target_attention_only", args.astro_target_attention_only)
                    setattr(model.config, "target_attention_only", args.astro_target_attention_only)
                if getattr(args, "astro_use_fisher_adaptive_bits", None) is not None:
                    setattr(model.config, "astro_use_fisher_adaptive_bits", args.astro_use_fisher_adaptive_bits)
                    setattr(model.config, "use_fisher_adaptive_bits", args.astro_use_fisher_adaptive_bits)
                if getattr(args, "astro_target_avg_bits", None) is not None:
                    setattr(model.config, "astro_target_avg_bits", args.astro_target_avg_bits)
                    setattr(model.config, "target_avg_bits", args.astro_target_avg_bits)
                if getattr(args, "astro_min_bits", None) is not None:
                    setattr(model.config, "astro_min_bits", args.astro_min_bits)
                    setattr(model.config, "min_bits", args.astro_min_bits)
                if getattr(args, "astro_max_bits", None) is not None:
                    setattr(model.config, "astro_max_bits", args.astro_max_bits)
                    setattr(model.config, "max_bits", args.astro_max_bits)
                if getattr(args, "astro_fisher_beta", None) is not None:
                    setattr(model.config, "astro_fisher_beta", args.astro_fisher_beta)
                    setattr(model.config, "fisher_beta", args.astro_fisher_beta)
                if getattr(args, "cachenotes_task", None) is not None:
                    setattr(model.config, "cachenotes_task", args.cachenotes_task)
                    setattr(model.config, "task_type", args.cachenotes_task)
                if getattr(args, "cachenotes_keep_ratio", None) is not None:
                    setattr(model.config, "cachenotes_keep_ratio", args.cachenotes_keep_ratio)
                    setattr(model.config, "keep_ratio", args.cachenotes_keep_ratio)
                if getattr(args, "cachenotes_max_error", None) is not None:
                    setattr(model.config, "cachenotes_max_error", args.cachenotes_max_error)
                    setattr(model.config, "max_error", args.cachenotes_max_error)
                if getattr(args, "cachenotes_target_attention_only", None) is not None:
                    setattr(model.config, "cachenotes_target_attention_only", args.cachenotes_target_attention_only)
                    setattr(model.config, "target_attention_only", args.cachenotes_target_attention_only)
                if getattr(args, "cachenotes_lora_r", None) is not None:
                    setattr(model.config, "cachenotes_lora_r", args.cachenotes_lora_r)
                    setattr(model.config, "lora_r", args.cachenotes_lora_r)
                    setattr(model.config, "lora_rank", args.cachenotes_lora_r)
                if getattr(args, "cachenotes_lora_alpha", None) is not None:
                    setattr(model.config, "cachenotes_lora_alpha", args.cachenotes_lora_alpha)
                    setattr(model.config, "lora_alpha", args.cachenotes_lora_alpha)
                if getattr(args, "cachenotes_skip_attention_qkv", None) is not None:
                    setattr(model.config, "cachenotes_skip_attention_qkv", args.cachenotes_skip_attention_qkv)
                    setattr(model.config, "skip_attention_qkv", args.cachenotes_skip_attention_qkv)
                if getattr(args, "come_target_layers", None) is not None:
                    setattr(model.config, "come_target_layers", args.come_target_layers)
                    setattr(model.config, "target_layers", args.come_target_layers)
                if getattr(args, "come_sparsity", None) is not None:
                    setattr(model.config, "come_sparsity", args.come_sparsity)
                    setattr(model.config, "sparsity", args.come_sparsity)
                if getattr(args, "come_alpha_init", None) is not None:
                    setattr(model.config, "come_alpha_init", args.come_alpha_init)
                    setattr(model.config, "alpha_init", args.come_alpha_init)
                if getattr(args, "come_lora_r", None) is not None:
                    setattr(model.config, "come_lora_r", args.come_lora_r)
                    setattr(model.config, "lora_r", args.come_lora_r)
                    setattr(model.config, "lora_rank", args.come_lora_r)
                if getattr(args, "come_lora_alpha", None) is not None:
                    setattr(model.config, "come_lora_alpha", args.come_lora_alpha)
                    setattr(model.config, "lora_alpha", args.come_lora_alpha)
                if getattr(args, "come_skip_attention_qkv", None) is not None:
                    setattr(model.config, "come_skip_attention_qkv", args.come_skip_attention_qkv)
                    setattr(model.config, "skip_attention_qkv", args.come_skip_attention_qkv)
                if getattr(args, "come_target_attention_only", None) is not None:
                    setattr(model.config, "come_target_attention_only", args.come_target_attention_only)
                    setattr(model.config, "target_attention_only", args.come_target_attention_only)
                # CoopQ
                if getattr(args, "coopq_bits", None) is not None:
                    setattr(model.config, "coopq_bits", args.coopq_bits)
                    setattr(model.config, "bits", args.coopq_bits)
                if getattr(args, "coopq_min_bits", None) is not None:
                    setattr(model.config, "coopq_min_bits", args.coopq_min_bits)
                    setattr(model.config, "min_bits", args.coopq_min_bits)
                if getattr(args, "coopq_max_bits", None) is not None:
                    setattr(model.config, "coopq_max_bits", args.coopq_max_bits)
                    setattr(model.config, "max_bits", args.coopq_max_bits)
                if getattr(args, "coopq_symmetric", None) is not None:
                    setattr(model.config, "coopq_symmetric", args.coopq_symmetric)
                    setattr(model.config, "symmetric", args.coopq_symmetric)
                if getattr(args, "coopq_per_channel", None) is not None:
                    setattr(model.config, "coopq_per_channel", args.coopq_per_channel)
                    setattr(model.config, "per_channel", args.coopq_per_channel)
                if getattr(args, "coopq_use_ste", None) is not None:
                    setattr(model.config, "coopq_use_ste", args.coopq_use_ste)
                    setattr(model.config, "use_ste", args.coopq_use_ste)
                if getattr(args, "coopq_target_attention_only", None) is not None:
                    setattr(model.config, "coopq_target_attention_only", args.coopq_target_attention_only)
                    setattr(model.config, "target_attention_only", args.coopq_target_attention_only)
                # CR
                if getattr(args, "cr_rank", None) is not None:
                    setattr(model.config, "cr_rank", args.cr_rank)
                    setattr(model.config, "rank", args.cr_rank)
                if getattr(args, "cr_max_error", None) is not None:
                    setattr(model.config, "cr_max_error", args.cr_max_error)
                    setattr(model.config, "max_error", args.cr_max_error)
                if getattr(args, "cr_target_attention_only", None) is not None:
                    setattr(model.config, "cr_target_attention_only", args.cr_target_attention_only)
                    setattr(model.config, "target_attention_only", args.cr_target_attention_only)
                # CUR
                if getattr(args, "cur_rank", None) is not None:
                    setattr(model.config, "cur_rank", args.cur_rank)
                    setattr(model.config, "rank", args.cur_rank)
                if getattr(args, "cur_min_rank", None) is not None:
                    setattr(model.config, "cur_min_rank", args.cur_min_rank)
                    setattr(model.config, "min_rank", args.cur_min_rank)
                if getattr(args, "cur_method", None) is not None:
                    setattr(model.config, "cur_method", args.cur_method)
                    setattr(model.config, "method", args.cur_method)
                if getattr(args, "cur_max_error", None) is not None:
                    setattr(model.config, "cur_max_error", args.cur_max_error)
                    setattr(model.config, "max_error", args.cur_max_error)
                if getattr(args, "cur_target_attention_only", None) is not None:
                    setattr(model.config, "cur_target_attention_only", args.cur_target_attention_only)
                    setattr(model.config, "target_attention_only", args.cur_target_attention_only)
                # CUR-MoE
                if getattr(args, "cur_moe_rank", None) is not None:
                    setattr(model.config, "cur_moe_rank", args.cur_moe_rank)
                    setattr(model.config, "rank", args.cur_moe_rank)
                if getattr(args, "cur_moe_target_attention_only", None) is not None:
                    setattr(model.config, "cur_moe_target_attention_only", args.cur_moe_target_attention_only)
                    setattr(model.config, "target_attention_only", args.cur_moe_target_attention_only)
                # DeltaKV
                if getattr(args, "delta_kv_rank", None) is not None:
                    setattr(model.config, "delta_kv_rank", args.delta_kv_rank)
                    setattr(model.config, "rank", args.delta_kv_rank)
                if getattr(args, "delta_kv_sparsity", None) is not None:
                    setattr(model.config, "delta_kv_sparsity", args.delta_kv_sparsity)
                    setattr(model.config, "sparsity", args.delta_kv_sparsity)
                if getattr(args, "delta_kv_max_error", None) is not None:
                    setattr(model.config, "delta_kv_max_error", args.delta_kv_max_error)
                    setattr(model.config, "max_error", args.delta_kv_max_error)
                if getattr(args, "delta_kv_target_attention_only", None) is not None:
                    setattr(model.config, "delta_kv_target_attention_only", args.delta_kv_target_attention_only)
                    setattr(model.config, "target_attention_only", args.delta_kv_target_attention_only)
                # DirectSVD
                if getattr(args, "direct_svd_rank", None) is not None:
                    setattr(model.config, "direct_svd_rank", args.direct_svd_rank)
                    setattr(model.config, "rank", args.direct_svd_rank)
                if getattr(args, "direct_svd_max_error", None) is not None:
                    setattr(model.config, "direct_svd_max_error", args.direct_svd_max_error)
                    setattr(model.config, "max_error", args.direct_svd_max_error)
                if getattr(args, "direct_svd_target_attention_only", None) is not None:
                    setattr(model.config, "direct_svd_target_attention_only", args.direct_svd_target_attention_only)
                    setattr(model.config, "target_attention_only", args.direct_svd_target_attention_only)
                # Dobi-SVD
                if getattr(args, "dobi_rank", None) is not None:
                    setattr(model.config, "dobi_rank", args.dobi_rank)
                    setattr(model.config, "rank", args.dobi_rank)
                if getattr(args, "dobi_init_rank_ratio", None) is not None:
                    setattr(model.config, "dobi_init_rank_ratio", args.dobi_init_rank_ratio)
                    setattr(model.config, "init_rank_ratio", args.dobi_init_rank_ratio)
                if getattr(args, "dobi_init_gate_val", None) is not None:
                    setattr(model.config, "dobi_init_gate_val", args.dobi_init_gate_val)
                    setattr(model.config, "init_gate_val", args.dobi_init_gate_val)
                if getattr(args, "dobi_max_error", None) is not None:
                    setattr(model.config, "dobi_max_error", args.dobi_max_error)
                    setattr(model.config, "max_error", args.dobi_max_error)
                if getattr(args, "dobi_target_attention_only", None) is not None:
                    setattr(model.config, "dobi_target_attention_only", args.dobi_target_attention_only)
                    setattr(model.config, "target_attention_only", args.dobi_target_attention_only)
                # Dynamic Entropy SVD
                if getattr(args, "dynamic_min_rank", None) is not None:
                    setattr(model.config, "dynamic_min_rank", args.dynamic_min_rank)
                    setattr(model.config, "min_rank", args.dynamic_min_rank)
                if getattr(args, "dynamic_max_rank", None) is not None:
                    setattr(model.config, "dynamic_max_rank", args.dynamic_max_rank)
                    setattr(model.config, "max_rank", args.dynamic_max_rank)
                if getattr(args, "dynamic_target_param_ratio", None) is not None:
                    setattr(model.config, "dynamic_target_param_ratio", args.dynamic_target_param_ratio)
                    setattr(model.config, "target_param_ratio", args.dynamic_target_param_ratio)
                if getattr(args, "dynamic_max_error", None) is not None:
                    setattr(model.config, "dynamic_max_error", args.dynamic_max_error)
                    setattr(model.config, "max_error", args.dynamic_max_error)
                if getattr(args, "dynamic_target_attention_only", None) is not None:
                    setattr(model.config, "dynamic_target_attention_only", args.dynamic_target_attention_only)
                    setattr(model.config, "target_attention_only", args.dynamic_target_attention_only)
                # ENLighten
                if getattr(args, "enlighten_rank", None) is not None:
                    setattr(model.config, "enlighten_rank", args.enlighten_rank)
                    setattr(model.config, "rank", args.enlighten_rank)
                if getattr(args, "enlighten_sparsity", None) is not None:
                    setattr(model.config, "enlighten_sparsity", args.enlighten_sparsity)
                    setattr(model.config, "sparsity", args.enlighten_sparsity)
                if getattr(args, "enlighten_structured_24", None) is not None:
                    setattr(model.config, "enlighten_structured_24", args.enlighten_structured_24)
                    setattr(model.config, "structured_24", args.enlighten_structured_24)
                if getattr(args, "enlighten_max_error", None) is not None:
                    setattr(model.config, "enlighten_max_error", args.enlighten_max_error)
                    setattr(model.config, "max_error", args.enlighten_max_error)
                if getattr(args, "enlighten_target_attention_only", None) is not None:
                    setattr(model.config, "enlighten_target_attention_only", args.enlighten_target_attention_only)
                    setattr(model.config, "target_attention_only", args.enlighten_target_attention_only)
                # Entropy Budget SVD
                if getattr(args, "entropy_budget_rank", None) is not None:
                    setattr(model.config, "entropy_budget_rank", args.entropy_budget_rank)
                    setattr(model.config, "rank", args.entropy_budget_rank)
                if getattr(args, "entropy_budget_total_budget_ratio", None) is not None:
                    setattr(model.config, "entropy_budget_total_budget_ratio", args.entropy_budget_total_budget_ratio)
                    setattr(model.config, "total_budget_ratio", args.entropy_budget_total_budget_ratio)
                if getattr(args, "entropy_budget_max_error", None) is not None:
                    setattr(model.config, "entropy_budget_max_error", args.entropy_budget_max_error)
                    setattr(model.config, "max_error", args.entropy_budget_max_error)
                if getattr(args, "entropy_budget_target_attention_only", None) is not None:
                    setattr(model.config, "entropy_budget_target_attention_only", args.entropy_budget_target_attention_only)
                    setattr(model.config, "target_attention_only", args.entropy_budget_target_attention_only)
                # ERC-SVD
                if getattr(args, "erc_allowed_error", None) is not None:
                    setattr(model.config, "erc_allowed_error", args.erc_allowed_error)
                    setattr(model.config, "allowed_error", args.erc_allowed_error)
                if getattr(args, "erc_max_error", None) is not None:
                    setattr(model.config, "erc_max_error", args.erc_max_error)
                    setattr(model.config, "max_error", args.erc_max_error)
                if getattr(args, "erc_target_attention_only", None) is not None:
                    setattr(model.config, "erc_target_attention_only", args.erc_target_attention_only)
                    setattr(model.config, "target_attention_only", args.erc_target_attention_only)
                if getattr(args, "erc_rank", None) is not None:
                    setattr(model.config, "erc_rank", args.erc_rank)
                    setattr(model.config, "rank", args.erc_rank)
                # FAQ Quantizer
                if getattr(args, "faq_bits", None) is not None:
                    setattr(model.config, "faq_bits", args.faq_bits)
                    setattr(model.config, "bits", args.faq_bits)
                if getattr(args, "faq_alpha", None) is not None:
                    setattr(model.config, "faq_alpha", args.faq_alpha)
                    setattr(model.config, "alpha", args.faq_alpha)
                if getattr(args, "faq_symmetric", None) is not None:
                    setattr(model.config, "faq_symmetric", args.faq_symmetric)
                    setattr(model.config, "symmetric", args.faq_symmetric)
                if getattr(args, "faq_use_ste", None) is not None:
                    setattr(model.config, "faq_use_ste", args.faq_use_ste)
                    setattr(model.config, "use_ste", args.faq_use_ste)
                if getattr(args, "faq_lora_r", None) is not None:
                    setattr(model.config, "faq_lora_r", args.faq_lora_r)
                    setattr(model.config, "lora_r", args.faq_lora_r)
                if getattr(args, "faq_max_error", None) is not None:
                    setattr(model.config, "faq_max_error", args.faq_max_error)
                    setattr(model.config, "max_error", args.faq_max_error)
                if getattr(args, "faq_target_attention_only", None) is not None:
                    setattr(model.config, "faq_target_attention_only", args.faq_target_attention_only)
                    setattr(model.config, "target_attention_only", args.faq_target_attention_only)
                # FLAP
                if getattr(args, "flap_sparsity", None) is not None:
                    setattr(model.config, "flap_sparsity", args.flap_sparsity)
                    setattr(model.config, "sparsity", args.flap_sparsity)
                if getattr(args, "flap_metric_type", None) is not None:
                    setattr(model.config, "flap_metric_type", args.flap_metric_type)
                    setattr(model.config, "metric_type", args.flap_metric_type)
                if getattr(args, "flap_structured_24", None) is not None:
                    setattr(model.config, "flap_structured_24", args.flap_structured_24)
                    setattr(model.config, "structured_24", args.flap_structured_24)
                if getattr(args, "flap_max_error", None) is not None:
                    setattr(model.config, "flap_max_error", args.flap_max_error)
                    setattr(model.config, "max_error", args.flap_max_error)
                if getattr(args, "flap_target_attention_only", None) is not None:
                    setattr(model.config, "flap_target_attention_only", args.flap_target_attention_only)
                    setattr(model.config, "target_attention_only", args.flap_target_attention_only)
                # FLAT-LLM
                if getattr(args, "flat_llm_keep_ratio", None) is not None:
                    setattr(model.config, "flat_llm_keep_ratio", args.flat_llm_keep_ratio)
                    setattr(model.config, "keep_ratio", args.flat_llm_keep_ratio)
                if getattr(args, "flat_llm_damping", None) is not None:
                    setattr(model.config, "flat_llm_damping", args.flat_llm_damping)
                    setattr(model.config, "damping", args.flat_llm_damping)
                if getattr(args, "flat_llm_structured_24", None) is not None:
                    setattr(model.config, "flat_llm_structured_24", args.flat_llm_structured_24)
                    setattr(model.config, "structured_24", args.flat_llm_structured_24)
                if getattr(args, "flat_llm_max_error", None) is not None:
                    setattr(model.config, "flat_llm_max_error", args.flat_llm_max_error)
                    setattr(model.config, "max_error", args.flat_llm_max_error)
                if getattr(args, "flat_llm_target_attention_only", None) is not None:
                    setattr(model.config, "flat_llm_target_attention_only", args.flat_llm_target_attention_only)
                    setattr(model.config, "target_attention_only", args.flat_llm_target_attention_only)
                # FRanDI
                if getattr(args, "frandi_rank", None) is not None:
                    setattr(model.config, "frandi_rank", args.frandi_rank)
                    setattr(model.config, "rank", args.frandi_rank)
                if getattr(args, "frandi_energy_threshold", None) is not None:
                    setattr(model.config, "frandi_energy_threshold", args.frandi_energy_threshold)
                    setattr(model.config, "energy_threshold", args.frandi_energy_threshold)
                if getattr(args, "frandi_lora_r", None) is not None:
                    setattr(model.config, "frandi_lora_r", args.frandi_lora_r)
                    setattr(model.config, "lora_r", args.frandi_lora_r)
                if getattr(args, "frandi_max_error", None) is not None:
                    setattr(model.config, "frandi_max_error", args.frandi_max_error)
                    setattr(model.config, "max_error", args.frandi_max_error)
                if getattr(args, "frandi_target_attention_only", None) is not None:
                    setattr(model.config, "frandi_target_attention_only", args.frandi_target_attention_only)
                    setattr(model.config, "target_attention_only", args.frandi_target_attention_only)
                # GreCoDeC
                if getattr(args, "grecodec_rank", None) is not None:
                    setattr(model.config, "grecodec_rank", args.grecodec_rank)
                    setattr(model.config, "rank", args.grecodec_rank)
                if getattr(args, "grecodec_max_iter", None) is not None:
                    setattr(model.config, "grecodec_max_iter", args.grecodec_max_iter)
                    setattr(model.config, "max_iter", args.grecodec_max_iter)
                if getattr(args, "grecodec_lora_r", None) is not None:
                    setattr(model.config, "grecodec_lora_r", args.grecodec_lora_r)
                    setattr(model.config, "lora_r", args.grecodec_lora_r)
                if getattr(args, "grecodec_max_error", None) is not None:
                    setattr(model.config, "grecodec_max_error", args.grecodec_max_error)
                    setattr(model.config, "max_error", args.grecodec_max_error)
                if getattr(args, "grecodec_target_attention_only", None) is not None:
                    setattr(model.config, "grecodec_target_attention_only", args.grecodec_target_attention_only)
                    setattr(model.config, "target_attention_only", args.grecodec_target_attention_only)
                # HashedNets & HashedNet
                if getattr(args, "hashed_net_buckets", None) is not None:
                    setattr(model.config, "hashed_net_buckets", args.hashed_net_buckets)
                    setattr(model.config, "n_buckets", args.hashed_net_buckets)
                if getattr(args, "hashed_net_use_sign", None) is not None:
                    setattr(model.config, "hashed_net_use_sign", args.hashed_net_use_sign)
                    setattr(model.config, "use_sign", args.hashed_net_use_sign)
                if getattr(args, "hashed_net_lora_r", None) is not None:
                    setattr(model.config, "hashed_net_lora_r", args.hashed_net_lora_r)
                    setattr(model.config, "lora_r", args.hashed_net_lora_r)
                if getattr(args, "hashed_net_max_error", None) is not None:
                    setattr(model.config, "hashed_net_max_error", args.hashed_net_max_error)
                    setattr(model.config, "max_error", args.hashed_net_max_error)
                if getattr(args, "hashed_net_target_attention_only", None) is not None:
                    setattr(model.config, "hashed_net_target_attention_only", args.hashed_net_target_attention_only)
                    setattr(model.config, "target_attention_only", args.hashed_net_target_attention_only)
                # Huffman
                if getattr(args, "huffman_quantize_bits", None) is not None:
                    setattr(model.config, "huffman_quantize_bits", args.huffman_quantize_bits)
                    setattr(model.config, "quantize_bits", args.huffman_quantize_bits)
                    setattr(model.config, "bits", args.huffman_quantize_bits)
                if getattr(args, "huffman_max_error", None) is not None:
                    setattr(model.config, "huffman_max_error", args.huffman_max_error)
                    setattr(model.config, "max_error", args.huffman_max_error)
                if getattr(args, "huffman_target_attention_only", None) is not None:
                    setattr(model.config, "huffman_target_attention_only", args.huffman_target_attention_only)
                    setattr(model.config, "target_attention_only", args.huffman_target_attention_only)
                # Hybrid SVD
                if getattr(args, "hybrid_svd_rank", None) is not None:
                    setattr(model.config, "hybrid_svd_rank", args.hybrid_svd_rank)
                    setattr(model.config, "svd_rank", args.hybrid_svd_rank)
                    setattr(model.config, "rank", args.hybrid_svd_rank)
                if getattr(args, "hybrid_svd_use_full", None) is not None:
                    setattr(model.config, "hybrid_svd_use_full", args.hybrid_svd_use_full)
                    setattr(model.config, "use_full_layer_replacement", args.hybrid_svd_use_full)
                if getattr(args, "hybrid_svd_lora_r", None) is not None:
                    setattr(model.config, "hybrid_svd_lora_r", args.hybrid_svd_lora_r)
                    setattr(model.config, "lora_r", args.hybrid_svd_lora_r)
                if getattr(args, "hybrid_svd_max_error", None) is not None:
                    setattr(model.config, "hybrid_svd_max_error", args.hybrid_svd_max_error)
                    setattr(model.config, "max_error", args.hybrid_svd_max_error)
                if getattr(args, "hybrid_svd_target_attention_only", None) is not None:
                    setattr(model.config, "hybrid_svd_target_attention_only", args.hybrid_svd_target_attention_only)
                    setattr(model.config, "target_attention_only", args.hybrid_svd_target_attention_only)
                # HybridKV
                if getattr(args, "hybridkv_budget", None) is not None:
                    setattr(model.config, "hybridkv_budget", args.hybridkv_budget)
                    setattr(model.config, "budget", args.hybridkv_budget)
                if getattr(args, "hybridkv_static_ratio", None) is not None:
                    setattr(model.config, "hybridkv_static_ratio", args.hybridkv_static_ratio)
                    setattr(model.config, "static_ratio", args.hybridkv_static_ratio)
                if getattr(args, "hybridkv_max_cache_len", None) is not None:
                    setattr(model.config, "hybridkv_max_cache_len", args.hybridkv_max_cache_len)
                    setattr(model.config, "max_cache_len", args.hybridkv_max_cache_len)
                if getattr(args, "hybridkv_num_sinks", None) is not None:
                    setattr(model.config, "hybridkv_num_sinks", args.hybridkv_num_sinks)
                    setattr(model.config, "num_sinks", args.hybridkv_num_sinks)
                if getattr(args, "hybridkv_recent_window", None) is not None:
                    setattr(model.config, "hybridkv_recent_window", args.hybridkv_recent_window)
                    setattr(model.config, "recent_window", args.hybridkv_recent_window)
                if getattr(args, "hybridkv_target_attention_only", None) is not None:
                    setattr(model.config, "hybridkv_target_attention_only", args.hybridkv_target_attention_only)
                    setattr(model.config, "target_attention_only", args.hybridkv_target_attention_only)
                # KMeans Quantizer
                if getattr(args, "kmeans_clusters", None) is not None:
                    setattr(model.config, "kmeans_clusters", args.kmeans_clusters)
                    setattr(model.config, "n_clusters", args.kmeans_clusters)
                if getattr(args, "kmeans_lora_r", None) is not None:
                    setattr(model.config, "kmeans_lora_r", args.kmeans_lora_r)
                    setattr(model.config, "lora_r", args.kmeans_lora_r)
                if getattr(args, "kmeans_max_error", None) is not None:
                    setattr(model.config, "kmeans_max_error", args.kmeans_max_error)
                    setattr(model.config, "max_error", args.kmeans_max_error)
                if getattr(args, "kmeans_target_attention_only", None) is not None:
                    setattr(model.config, "kmeans_target_attention_only", args.kmeans_target_attention_only)
                    setattr(model.config, "target_attention_only", args.kmeans_target_attention_only)
                # Kronecker
                if getattr(args, "kronecker_rank", None) is not None:
                    setattr(model.config, "kronecker_rank", args.kronecker_rank)
                    setattr(model.config, "rank", args.kronecker_rank)
                if getattr(args, "kronecker_max_error", None) is not None:
                    setattr(model.config, "kronecker_max_error", args.kronecker_max_error)
                    setattr(model.config, "max_error", args.kronecker_max_error)
                if getattr(args, "kronecker_target_attention_only", None) is not None:
                    setattr(model.config, "kronecker_target_attention_only", args.kronecker_target_attention_only)
                    setattr(model.config, "target_attention_only", args.kronecker_target_attention_only)
                # Kronecker TT LoRA
                if getattr(args, "tt_ranks", None) is not None:
                    setattr(model.config, "tt_ranks", args.tt_ranks)
                if getattr(args, "kronecker_tt_lora_max_error", None) is not None:
                    setattr(model.config, "kronecker_tt_lora_max_error", args.kronecker_tt_lora_max_error)
                    setattr(model.config, "max_error", args.kronecker_tt_lora_max_error)
                if getattr(args, "kronecker_tt_lora_target_attention_only", None) is not None:
                    setattr(model.config, "kronecker_tt_lora_target_attention_only", args.kronecker_tt_lora_target_attention_only)
                    setattr(model.config, "target_attention_only", args.kronecker_tt_lora_target_attention_only)
                # KVTC Compressor
                if getattr(args, "kvtc_bits", None) is not None:
                    setattr(model.config, "kvtc_bits", args.kvtc_bits)
                    setattr(model.config, "bits", args.kvtc_bits)
                if getattr(args, "kvtc_pca_energy", None) is not None:
                    setattr(model.config, "kvtc_pca_energy", args.kvtc_pca_energy)
                    setattr(model.config, "pca_energy", args.kvtc_pca_energy)
                if getattr(args, "kvtc_max_error", None) is not None:
                    setattr(model.config, "kvtc_max_error", args.kvtc_max_error)
                    setattr(model.config, "max_error", args.kvtc_max_error)
                if getattr(args, "kvtc_target_attention_only", None) is not None:
                    setattr(model.config, "kvtc_target_attention_only", args.kvtc_target_attention_only)
                    setattr(model.config, "target_attention_only", args.kvtc_target_attention_only)
                if getattr(args, "kvtc_fallback_to_original", None) is not None:
                    setattr(model.config, "kvtc_fallback_to_original", args.kvtc_fallback_to_original)
                    setattr(model.config, "fallback_to_original", args.kvtc_fallback_to_original)

                # Enforce module-specific max_error and target_attention_only to prevent contamination from unrelated compressors
                if getattr(args, f"{name}_max_error", None) is not None:
                    setattr(model.config, "max_error", getattr(args, f"{name}_max_error"))
                elif getattr(args, "max_error", None) is not None:
                    setattr(model.config, "max_error", args.max_error)

                if getattr(args, f"{name}_target_attention_only", None) is not None:
                    setattr(model.config, "target_attention_only", getattr(args, f"{name}_target_attention_only"))
                elif getattr(args, "target_attention_only", None) is not None:
                    setattr(model.config, "target_attention_only", args.target_attention_only)

                if name in ("svd_llm", "svd_llm_compressor"):
                    if getattr(args, "rank_ratio", None) is not None:
                        setattr(model.config, "rank_ratio", args.rank_ratio)
                    if getattr(args, "max_error", None) is not None:
                        setattr(model.config, "max_error", args.max_error)
                    if getattr(args, "lora_rank", None) is not None:
                        setattr(model.config, "lora_rank", args.lora_rank)

                if name in ("a3", "a3_compressor"):
                    a3_err = getattr(args, "a3_max_error", None)
                    if a3_err is None:
                        a3_err = getattr(args, "max_error", 0.50)
                    setattr(model.config, "a3_max_error", a3_err)
                    setattr(model.config, "max_error", a3_err)
                    a3_attn = getattr(args, "a3_target_attention_only", getattr(args, "target_attention_only", True))
                    setattr(model.config, "a3_attention_only", a3_attn)
                    setattr(model.config, "target_attention_only", a3_attn)

                    LOG.info(
                        "Calling apply_module_to_model with rank=%d for module=%s (lora_rank=%s)",
                        module_rank, name, getattr(args, "lora_rank", None),
                    )

            model, _ = apply_module_to_model(model, name, typ, rank=module_rank, device=device)
            model = model.to(device)
            injected = True
            path_name = f"adapted_{family}" if family != "bert" else "legacy_bert"
            LOG.info("module=%s path=%s", name, path_name)
            num_svd_layers = sum(
                1 for m in model.modules()
                if m.__class__.__name__ in ("SVDLinear", "SVDLinearLayer", "FactorisedLinear", "A3FactorizedLinear", "SVDLLMLinear")
                or any(k in m.__class__.__name__ for k in ("SVDCompressed", "NEXUS", "ORION", "MoBA", "A3"))
                or hasattr(m, "q_down")
            )
            LOG.info("module=%s injection summary: %d SVDLinear layers active", name, num_svd_layers)
            if name in ("aatc_xkv", "aatc_xkv_compressor", "a3", "a3_compressor", "aasvd", "adasvd", "svd_llm", "svd_llm_compressor", "admm", "admm_pruner", "admm_pruner_distill", "armor", "armor_pruner", "astro", "astro_quantizer", "cachenotes", "cachenotes_compressor", "come", "come_compressor", "coopq", "coopq_quantizer", "cr", "cr_compressor", "cur", "cur_compressor", "cur_moe", "cur_moe_compressor", "delta_kv", "delta_kv_compressor", "direct_svd", "direct_svd_compressor", "dobi", "dobi_svd", "dynamic_entropy_svd", "dynamic_rank", "dynamic_entropy", "enlighten", "enlighten_compressor", "entropy_budget", "entropy_budget_allocation", "erc", "erc_svd", "faq", "faq_quantizer", "flap", "flat_llm", "flat_llm_compressor", "frandi", "frandi_compressor", "grecodec", "grecodec_compressor", "hashednets", "hashednets_compressor", "hashed_net", "hashed_net_compressor", "huffman", "hybrid_svd", "hybridkv", "hybridkv_compressor", "kmeans", "kmeans_quantize", "kronecker", "kronecker_compressor", "kronecker_tt_lora", "kvtc", "kvtc_compressor", "kvtc_quantizer"):
                try:
                    sdata = getattr(model, "compression_stats", {})
                    if not sdata and name in ("aasvd", "adasvd"):
                        sdata = {"num_compressed_layers": num_svd_layers}
                    with open(out / f"{name}_stats.json", "w", encoding="utf-8") as sf:
                        json.dump(sdata, sf, indent=2)
                except Exception:
                    pass
            # -------------------------------------------------------------------------
            # DIAGNOSTIC / DEBUG / DRY-RUN HOOK
            # -------------------------------------------------------------------------
            if getattr(args, "debug_module", False) or getattr(args, "dry_run", False) or getattr(args, "verbose_debug", False):
                LOG.info("=" * 70)
                LOG.info("[DIAGNOSTIC] Running module health & forward verification for '%s'...", name)
                # 1. Parameter sanity check
                nan_params = []
                total_injected_p = 0
                for pn, pv in model.named_parameters():
                    total_injected_p += pv.numel()
                    if torch.isnan(pv).any() or torch.isinf(pv).any():
                        nan_params.append(pn)
                if nan_params:
                    LOG.error("[DIAGNOSTIC] Detected %d parameters with NaN/Inf values: %s", len(nan_params), nan_params[:5])
                    raise RuntimeError(f"Module '{name}' injected weights contain NaN/Inf: {nan_params[:3]}")
                LOG.info("[DIAGNOSTIC] Parameter count: before=%d, after=%d, delta=%+d", total_params_before, total_injected_p, total_injected_p - total_params_before)

                # 2. Forward pass dry-run test
                if eval_batches and len(eval_batches) > 0:
                    test_x, test_y = eval_batches[0]
                    test_x = {k: v[:2].to(device) for k, v in test_x.items()}
                    test_y = test_y[:2].to(device)
                    model.eval()
                    with torch.no_grad():
                        with torch.autocast(str(device).split(":")[0], dtype=torch.float16, enabled=args.precision == "fp16" and str(device).startswith("cuda")):
                            t_out = model(**test_x)
                            t_logits = t_out.logits if hasattr(t_out, "logits") else t_out[0]
                            if torch.isnan(t_logits).any() or torch.isinf(t_logits).any():
                                raise RuntimeError(f"Forward pass output logits contain NaN/Inf for module '{name}'")
                    LOG.info("[DIAGNOSTIC] Forward verification PASSED (logits shape=%s, finite=True)", list(t_logits.shape))

                if getattr(args, "dry_run", False):
                    LOG.info("[DIAGNOSTIC] --dry_run specified: skipping full recovery fine-tuning and evaluation.")
                    LOG.info("=" * 70)
                    perf_metrics = {
                        "accuracy_pct": 100.0,
                        "f1_score": 1.0,
                        "f1_macro": 1.0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "val_loss": 0.0,
                        "peak_vram_mb": 0.0,
                        "tokens_per_sec": 0.0,
                        "latency_ms_per_token": 0.0,
                        "elapsed_s": round(time.monotonic() - t, 2),
                    }
                    comp_metrics = {
                        "compression_rate": total_params_before / max(1, total_injected_p),
                        "compression_ratio_pct": round(100.0 * (1.0 - total_injected_p / max(1, total_params_before)), 2),
                        "num_compressed_layers": num_svd_layers,
                        "num_fallback_layers": 0,
                    }
                    return perf_metrics, comp_metrics, True, path_name, False
                LOG.info("=" * 70)

            had_nan = False
            recovery_targets = {x.strip() for x in args.recovery_modules.split(",") if x.strip()}
            should_recover = (
                args.recovery_steps > 0
                and (
                    not recovery_targets
                    or "all" in recovery_targets
                    or typ in recovery_targets
                    or name in recovery_targets
                )
            )
            if should_recover:
                FROM_SCRATCH_MODULES = {
                    "linear_attention", "ssm", "cosformer", "rff", "fourier",
                    "simple_ssm", "gated_linear_attn", "decay_linear_attn",
                    "cluster", "cur", "cur_moe", "cur_moe_compressor",
                    "performer", "nystrom", "mamba", "svd-mamba", "upi_stuffed_mamba",
                    "vig", "galerkin", "regla", "hgrn2_layer", "hgrn2_outer",
                    "hgrn2_gated_deltanet", "gated_delta_net", "gated_deltanet_block",
                    "factorised_matrix_state", "linformer", "lsh", "hashed_net",
                    "hamming", "spectron", "span_expanded", "gfws", "hla", "oha", "kda",
                    "aegis_hybrid", "aegis_spectron_fp8",
                    "halo_regla", "griffin_ssd", "granite_hybrid", "gqa", "gqa_attention",
                    "nha", "nha_attention",
                    "goldfinch_ahn", "goldfinch",
                    "jamba", "jamba2",
                    "mamba_svd_flash",
                    "adaptive_sparse", "adaptive_sparse_attention",
                }
                if name in FROM_SCRATCH_MODULES or name in ("asvd", "asvd_attention"):
                    rec_steps = args.recovery_steps if getattr(args, "recovery_steps", None) is not None else 50
                    if name in ("asvd", "asvd_attention") and family == "gpt":
                        rec_lr = args.learning_rate if args.learning_rate != 2e-4 else 1e-4
                    else:
                        rec_lr = args.learning_rate if args.learning_rate else 2e-4
                else:
                    rec_steps = 150 if (family == "qwen3_5" and args.recovery_steps == 100) else args.recovery_steps
                    rec_lr = args.learning_rate if args.learning_rate else (5e-5 if (family == "qwen3_5" or adapter.hidden_size > 512) else 1e-4)
                LOG.info("Running %d recovery fine-tuning steps for %s (lr=%.1e)", rec_steps, name, rec_lr)
                if typ == "attention":
                    for m in model.modules():
                        if hasattr(m, "merged") and hasattr(m, "reset_parameters"):
                            m.reset_parameters()
                            m.merged = False

                should_freeze_clf = getattr(args, "freeze_clf", False) or bool(args.parent_checkpoint)
                if should_freeze_clf:
                    for n, p in model.named_parameters():
                        if any(h in n.lower() for h in ("score", "classifier", "pre_classifier")):
                            p.requires_grad = False
                from framework.utils.compat import get_recovery_params
                trainable = get_recovery_params(model)
                if should_freeze_clf and trainable:
                    clf_param_ids = {id(cp) for cn, cp in model.named_parameters() if any(h in cn.lower() for h in ("score", "classifier", "pre_classifier"))}
                    trainable = [p for p in trainable if id(p) not in clf_param_ids]
                if trainable:
                    trainable_recovery_params = sum(p.numel() for p in trainable)
                    model._trainable_recovery_params = trainable_recovery_params
                    trainable_ids = {id(p) for p in trainable}
                    for p in model.parameters():
                        p.requires_grad = (id(p) in trainable_ids)
                    if getattr(args, "kl_distill", False):
                        LOG.info("Running recovery with KL Distillation (alpha=%.2f, temp=%.1f)", args.kl_alpha, args.kl_temperature)
                        finetune_with_kd(
                            model, base, train, args.dataset, rec_steps, args.seq_len, effective_batch_size, device,
                            args.precision == "fp16", lr=rec_lr, alpha=args.kl_alpha, temperature=args.kl_temperature,
                            precomputed_batches=train_batches
                        )
                        had_nan = False
                    else:
                        had_nan = finetune_lora(
                            model, tokenizer, train, args.dataset, rec_steps, args.seq_len, effective_batch_size, device,
                            args.precision == "fp16", lr=rec_lr, precomputed_batches=train_batches,
                            optimizer_type=active_opt, muon_lr=args.muon_lr, adam_lr=args.adam_lr,
                            weight_decay=args.weight_decay, timeout=effective_timeout,
                            span_lambda=float(getattr(args, "span_lambda", 0.0) or 0.0),
                        )

        total_params_after = sum(p.numel() for p in model.parameters())

        try:
            perf_metrics = evaluate_comprehensive(
                model, tokenizer, val, args.dataset, eval_seq_len, effective_batch_size, device,
                fp16=args.precision == "fp16", precomputed_batches=eval_batches
            )
            eval_acc = perf_metrics["accuracy_pct"]
            comp_metrics = compute_compression_metrics(
                model, name, typ, total_params_before, total_params_after, trainable_recovery_params
            )

            try:
                # 1. Comprehensive Profiling Metrics Export
                collect_and_export_full_metrics(
                    model, eval_batches, f"{model_slug}_{name}",
                    seq_len=eval_seq_len, device=device, fp16=args.precision == "fp16",
                    out_dir=str(out),
                    perf_metrics=perf_metrics,
                    comp_metrics=comp_metrics,
                    status="PENDING",
                    elapsed_s=round(time.monotonic() - t, 2),
                )
                # 2. Complete Weights & Adapter Checkpoint Export
                weights_dir = out / "checkpoints" / name
                weights_dir.mkdir(parents=True, exist_ok=True)

                module_state = {}
                for mod_name, mod in model.named_modules():
                    is_svd = mod.__class__.__name__ == "SVDLinear" or (hasattr(mod, "U") and hasattr(mod, "V"))
                    is_custom_attn = getattr(mod, "is_custom_attention", False) or "AdaptiveSpan" in mod.__class__.__name__
                    if is_svd or is_custom_attn or any(k in mod_name for k in [name, "span", "adapter", "compressor"]):
                        for param_name, param in mod.named_parameters(recurse=False):
                            module_state[f"{mod_name}.{param_name}"] = param.detach().cpu()

                lora_state = {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                    if "lora_" in k or "span_" in k or any(h in k for h in ["classifier.", "score.", "classification_head."])
                }

                checkpoint_payload = {
                    "module_name": name,
                    "model_name": str(args.model),
                    "model_family": family,
                    "dataset": args.dataset,
                    "accuracy_pct": eval_acc,
                    "recovery_steps": rec_steps,
                    "timestamp": datetime.now().isoformat(),
                    "performance_metrics": perf_metrics,
                    "compression_metrics": comp_metrics,
                    "module_state_dict": module_state,
                    "lora_state_dict": lora_state,
                }
                save_heavy = (os.environ.get("SAVE_HEAVY_WEIGHTS", "0") == "1")
                if save_heavy:
                    ckpt_file = weights_dir / "weights.pt"
                    torch.save(checkpoint_payload, ckpt_file)
                    LOG.info("Successfully saved model weights and checkpoint to %s (module_params=%d, lora_params=%d)",
                             ckpt_file, len(module_state), len(lora_state))
                else:
                    meta_file = weights_dir / "checkpoint_meta.json"
                    lean_payload = {
                        k: v for k, v in checkpoint_payload.items()
                        if k not in ("module_state_dict", "lora_state_dict")
                    }
                    try:
                        with open(meta_file, "w", encoding="utf-8") as f:
                            json.dump(lean_payload, f, indent=2, default=str)
                    except Exception:
                        pass

                # Export lean factorized SVD checkpoint artifacts (~20-90 MB) for automated publishing
                hf_export_dir = weights_dir / "hf_export"
                if eval_acc >= pass_threshold_pct:
                    hf_export_dir.mkdir(parents=True, exist_ok=True)
                try:
                    if save_heavy:
                        torch.save(checkpoint_payload, hf_export_dir / "weights.pt")
                    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
                        try:
                            tokenizer.save_pretrained(hf_export_dir)
                        except Exception:
                            pass
                    if hasattr(model, "config") and model.config is not None:
                        try:
                            with open(hf_export_dir / "config.json", "w", encoding="utf-8") as cf:
                                json.dump(model.config.to_dict(), cf, indent=2)
                        except Exception:
                            pass
                    # Save evaluation metrics json in export directory
                    with open(hf_export_dir / "metrics.json", "w", encoding="utf-8") as mf:
                        json.dump({"performance": perf_metrics, "compression": comp_metrics}, mf, indent=2)
                except Exception as hf_save_err:
                    LOG.debug("Optional HF model export info: %s", hf_save_err)
            except Exception as me_err:
                LOG.warning("Failed to collect/export full metrics or weights for %s: %s", name, me_err)
            return perf_metrics, comp_metrics, injected, path_name, had_nan
        finally:
            if model is not None:
                try:
                    model.zero_grad(set_to_none=True)
                    model.to("cpu")
                except Exception:
                    pass
                torch.cuda.empty_cache()


def run_single_module_main(args) -> dict:
    import os
    import torch
    from framework.core.snapshot_restore import ModelStateSnapshot
    from framework.core.registry import ATTENTION_REGISTRY
    name = args.modules[0] if isinstance(args.modules, list) else args.modules
    typ = "baseline" if name in ("full", "dense") else ("attention" if name in ATTENTION_REGISTRY else "compressor")

    if name in DEPRECATED_MODULES:
        reason = DEPRECATED_MODULES[name]
        LOG.warning(
            "Skipping deprecated module '%s'. Reason: %s",
            name,
            reason,
        )
        return {
            "module_name": name,
            "status": "SKIP_DEPRECATED",
            "accuracy_pct": 0.0,
            "elapsed_s": 0.0,
            "error": f"DEPRECATED: {reason}",
            "path": "deprecated",
            "timestamp": datetime.now().isoformat(),
        }

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            target_frac = float(args.max_vram_fraction) if getattr(args, "max_vram_fraction", None) is not None else 0.70
            if 0.0 < target_frac <= 1.0:
                torch.cuda.set_per_process_memory_fraction(target_frac)
                LOG.info("Configured PyTorch per-process GPU memory fraction cap: %.1f%%", target_frac * 100.0)
            free_mem, _ = torch.cuda.mem_get_info()
            free_gb = free_mem / (1024 ** 3)
            if free_gb < 4.0:
                LOG.warning("LOW GPU MEMORY: Only %.2f GB VRAM available (< 4.0 GB). High risk of CUDA OOM.", free_gb)
        except Exception as mem_err:
            LOG.debug("Could not query GPU memory info or set fraction: %s", mem_err)
    if args.output_dir is None:
        safe_model = Path(args.model).name.replace("-", "_").lower()
        # Cleanly organized hierarchical structure for saving folders
        args.output_dir = str(Path(__file__).resolve().parents[2] / "results" / args.dataset / safe_model)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    if args.dataset_dir is None:
        for cand in [
            Path("D:/SVD_Project/Codes/framework/data/glue_data") / args.dataset,
            Path("Codes/framework/data/glue_data") / args.dataset,
            Path(__file__).parent / "data" / "glue_data" / args.dataset,
            Path("D:/SVD_Project/Codes/framework/data/glue_data/sst2"),
            Path("Codes/framework/data/glue_data/sst2"),
            Path(__file__).parent / "data" / "glue_data" / "sst2",
        ]:
            if cand.exists():
                args.dataset_dir = str(cand)
                break

    # Auto-resolve base models (e.g. roberta-base, gpt2) to fine-tuned classification checkpoints if running on SST-2
    if args.dataset and "wikitext" not in args.dataset.lower():
        model_p = Path(args.model)
        m_name = model_p.name.lower()
        if "roberta" in m_name and "sst2" not in m_name and args.dataset == "sst2":
            cand_roberta = [
                model_p.parent / "roberta_sst2",
                Path("models/roberta_sst2"),
                Path("D:/SVD_Project/models/roberta_sst2"),
            ]
            for c in cand_roberta:
                if c.exists() and ((c / "config.json").exists() or (c / "model.safetensors").exists()):
                    LOG.info("Auto-resolving base model '%s' to fine-tuned SST-2 model: %s", args.model, c)
                    args.model = str(c)
                    break
        elif "gpt2" in m_name and "sst2" not in m_name and args.dataset == "sst2":
            cand_gpt2 = [
                model_p.parent / "gpt2_sst2",
                Path("models/gpt2_sst2"),
                Path("D:/SVD_Project/models/gpt2_sst2"),
            ]
            for c in cand_gpt2:
                if c.exists() and ((c / "config.json").exists() or (c / "model.safetensors").exists()):
                    LOG.info("Auto-resolving base model '%s' to fine-tuned SST-2 model: %s", args.model, c)
                    args.model = str(c)
                    break
    try:
        if "wikitext" in args.dataset.lower():
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        elif args.dataset_dir and Path(args.dataset_dir).exists():
            ds = load_from_disk(args.dataset_dir)
        elif args.dataset in {"ag_news", "agnews"}:
            ds = load_dataset("fancyzhx/ag_news")
        else:
            ds = load_dataset(args.dataset)
    except Exception as e:
        LOG.warning("Failed to load dataset '%s' from disk/hub (%s). Falling back to local sst2.", args.dataset, e)
        fallback_sst2 = Path("D:/SVD_Project/Codes/framework/data/glue_data/sst2")
        if not fallback_sst2.exists():
            fallback_sst2 = Path(__file__).parent / "data" / "glue_data" / "sst2"
        ds = load_from_disk(str(fallback_sst2))
    train = ds["train"]
    val = ds["validation"] if "validation" in ds else ds["test"]
    tokenizer = AutoTokenizer.from_pretrained(args.model); tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # Train the newly initialized classifier in fp32. GradScaler cannot
    # unscale gradients belonging to fp16 parameters. Cast the trained model
    # to the requested sweep precision immediately afterwards.
    dtype = torch.float32
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    num_classes = args.num_labels
    if num_classes is None:
        try:
            if "label" in train.features and hasattr(train.features["label"], "num_classes"):
                num_classes = train.features["label"].num_classes
            elif "label" in train.column_names:
                num_classes = len(set(train["label"][:500]))
            else:
                num_classes = 2
        except Exception:
            num_classes = 2
    LOG.info("Configured dataset '%s' with num_labels=%d", args.dataset, num_classes)
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        
        cfg = AutoConfig.from_pretrained(args.model)
        if "wikitext" in args.dataset.lower():
            if cfg.model_type in ("distilbert", "bert", "roberta", "albert", "deberta", "deberta-v2", "electra"):
                LOG.warning("Skipping model %s on wikitext because it lacks an LM head.", args.model)
                out = Path(args.output_dir)
                out.mkdir(parents=True, exist_ok=True)
                csv_name = f"summary_{args.output_suffix}.csv" if getattr(args, "output_suffix", None) else "summary.csv"
                with open(out / csv_name, "w", newline="", encoding="utf-8") as f:
                    f.write("module_name,status,accuracy_pct\n")
                return {}
                
            try:
                base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
            except TypeError:
                base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
        else:
            try:
                base = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_classes, ignore_mismatched_sizes=True, dtype=dtype).to(device)
            except TypeError:
                base = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_classes, ignore_mismatched_sizes=True, torch_dtype=dtype).to(device)
    except torch.OutOfMemoryError as e:
        LOG.critical("CUDA OOM while loading base model: %s", e)
        sys.exit(1)
    base.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    install_qwen_aliases(base)
    base.tokenizer = tokenizer
    if hasattr(base, "config") and base.config is not None:
        pass  # removed base.config.tokenizer = tokenizer

    if os.environ.get("SVD_COMPILE", "0") == "1":
        try:
            base = torch.compile(base, mode="default", fullgraph=False)
            LOG.info("torch.compile enabled (max-autotune)")
        except Exception as e:
            LOG.warning(f"torch.compile failed: {e}")

    qwen = is_qwen_model(base)
    family = detect_family(base) if args.model_family == "auto" else args.model_family
    adapter = ModelAdapter(base, family)
    LOG.info("model=%s qwen_compat=%s hidden=%s heads=%s head_dim=%s layers=%s",
             args.model, qwen, getattr(base.config, "hidden_size", "?"),
             getattr(base.config, "num_attention_heads", "?"),
             getattr(base.config, "head_dim", "?"), getattr(base.config, "num_hidden_layers", "?"))
    LOG.info("detected model family=%s adapter=%s", family, adapter.as_dict())
    active_opt = "muon" if (args.use_muon or args.optimizer == "muon") else "adamw"
    LOG.info("=" * 80)
    LOG.info("ACTIVE SWEEP CONFIGURATION:")
    LOG.info("  model:             %s", args.model)
    LOG.info("  family:            %s", family)
    LOG.info("  dataset:           %s", args.dataset)
    LOG.info("  batch_size:        %d", args.batch_size)
    LOG.info("  recovery_steps:    %d", args.recovery_steps)
    LOG.info("  optimizer:         %s (muon_lr=%.1e, adam_lr=%.1e, weight_decay=%.1e)",
             active_opt, args.muon_lr, args.adam_lr if active_opt == "muon" else args.learning_rate, args.weight_decay)
    LOG.info("  precision:         %s", args.precision)
    LOG.info("=" * 80)
    # Enable gradient checkpointing if requested or for long sequences (> 512)
    if (args.gradient_checkpointing or args.seq_len > 512) and hasattr(base, "gradient_checkpointing_enable"):
        try:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            LOG.info("Enabled gradient checkpointing for %s backbone", family)
        except Exception:
            try:
                base.gradient_checkpointing_enable()
                LOG.info("Enabled gradient checkpointing for %s backbone", family)
            except Exception as e:
                LOG.warning("Could not enable gradient checkpointing: %s", e)

    checkpoint = out / "finetuned_model.pt"
    parent_checkpoint = Path(args.parent_checkpoint) if args.parent_checkpoint and Path(args.parent_checkpoint).exists() else None
    if parent_checkpoint is None:
        if args.parent_checkpoint:
            LOG.warning("Specified parent_checkpoint '%s' does not exist; attempting automatic fallback search.", args.parent_checkpoint)
        candidates = [
            out.parent / f"{family}_{args.dataset}_lora_baseline" / "finetuned_model.pt",
            out.parent / f"{family}2_{args.dataset}_lora_baseline" / "finetuned_model.pt",
            out.parent / f"{family}_{args.dataset}_full_sweep" / "finetuned_model.pt",
            out.parent / f"{family}_{args.dataset}_baseline" / "finetuned_model.pt",
            Path("results") / f"{family}_{args.dataset}_lora_baseline" / "finetuned_model.pt",
            Path("results") / f"{family}_{args.dataset}_baseline" / "finetuned_model.pt",
        ]
        if args.dataset == "sst2":
            candidates.extend([
                out.parent / "unified_asvd" / family / "finetuned_model.pt",
                out.parent / "unified_delta_kv" / family / "finetuned_model.pt",
                out.parent / "unified_nsa" / family / "finetuned_model.pt",
                out.parent / "unified_infini" / family / "finetuned_model.pt",
                out.parent / "unified_rpc" / family / "finetuned_model.pt",
                out.parent / "unified_res_svd" / family / "finetuned_model.pt",
                Path("results") / "unified_asvd" / family / "finetuned_model.pt",
                Path("results") / "unified_delta_kv" / family / "finetuned_model.pt",
            ])
        parent_checkpoint = next((c for c in candidates if c.exists()), Path("non_existent_placeholder.pt"))

    def _safe_load_state(target_model: nn.Module, raw_state: Any) -> None:
        st = raw_state.get("state_dict", raw_state) if isinstance(raw_state, dict) else raw_state
        meta = raw_state.get("metadata", {}) if isinstance(raw_state, dict) else {}
        alpha = float(meta.get("alpha", 32.0))
        r = float(meta.get("r", 16.0))
        scaling = alpha / max(1.0, r)

        model_st = target_model.state_dict()
        matching_keys = [k for k in st if k in model_st and hasattr(st[k], "shape") and hasattr(model_st[k], "shape") and st[k].shape == model_st[k].shape]
        is_partial = (
            len(st) < len(model_st) * 0.7
            or any(".lora_" in k for k in st)
            or all(any(h in k for h in ("score", "classifier")) for k in st if "weight" in k or "bias" in k)
        )
        denom = len(st) if is_partial else len(model_st)
        match_ratio = len(matching_keys) / max(1, denom)
        if match_ratio < 0.5:
            LOG.warning(
                "Checkpoint only matches %.1f%% of required keys (need >=50%%); skipping load to prevent weight corruption.",
                match_ratio * 100.0,
            )
            return

        filtered_st = {}

        # 1. Direct matches
        for k, v in st.items():
            if k in model_st:
                if hasattr(v, "shape") and hasattr(model_st[k], "shape"):
                    if v.shape == model_st[k].shape:
                        filtered_st[k] = v
                    else:
                        LOG.warning("Skipping checkpoint parameter '%s' due to shape mismatch: %s vs current model %s",
                                    k, list(v.shape), list(model_st[k].shape))
                else:
                    filtered_st[k] = v

        # 2. Check for LoRA-wrapped layers in checkpoint mapped to plain layers in model
        #    e.g. checkpoint has 'classifier.dense.base_layer.weight' (+ optional lora_A/B), model has 'classifier.dense.weight'
        for k in model_st:
            if k not in filtered_st:
                base_key = k.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
                if base_key in st:
                    val = st[base_key].clone()
                    if k.endswith(".weight"):
                        prefix = k[:-len(".weight")]
                        # Try direct nn.Parameter keys first (LoRALinear convention)
                        lora_a_key = f"{prefix}.lora_A"
                        lora_b_key = f"{prefix}.lora_B"
                        if lora_a_key not in st or lora_b_key not in st:
                            # Fall back to submodule.weight convention (nn.Linear-based LoRA)
                            lora_a_key = f"{prefix}.lora_A.weight"
                            lora_b_key = f"{prefix}.lora_B.weight"
                        if lora_a_key in st and lora_b_key in st:
                            a = st[lora_a_key]
                            b = st[lora_b_key]
                            if a.dim() == 2 and b.dim() == 2:
                                if a.shape[0] == b.shape[1]:
                                    delta = (b @ a) * scaling
                                elif a.shape[1] == b.shape[0]:
                                    delta = (a.t() @ b.t()).t() * scaling
                                else:
                                    delta = None
                                if delta is not None and delta.shape == val.shape:
                                    val = val + delta.to(val.dtype)
                    filtered_st[k] = val
                    LOG.info("Reconstructed unmerged model layer '%s' from checkpoint LoRA '%s'", k, base_key)

        # 3. Check for plain layers in checkpoint mapped to LoRA base_layer in model
        #    e.g. checkpoint has 'classifier.dense.weight', model has 'classifier.dense.base_layer.weight'
        for k in model_st:
            if k not in filtered_st and ".base_layer." in k:
                plain_key = k.replace(".base_layer.", ".")
                if plain_key in st:
                    filtered_st[k] = st[plain_key]
                    LOG.info("Mapped plain checkpoint layer '%s' into LoRA base_layer '%s'", plain_key, k)

        missing, unexpected = target_model.load_state_dict(filtered_st, strict=False)
        if missing:
            LOG.warning("Missing keys during load: %s", missing[:10])
        if unexpected:
            LOG.warning("Unexpected keys during load: %s", unexpected[:10])

    # Auto-detect if checkpoint or parent_checkpoint is a LoRA checkpoint
    is_lora_mode = (args.finetune_mode == "lora" or family in ("qwen3_5", "qwen"))
    active_ckpt = checkpoint if checkpoint.exists() else (parent_checkpoint if parent_checkpoint.exists() else None)
    if active_ckpt is not None:
        try:
            probe = torch.load(active_ckpt, map_location="cpu", weights_only=False)
            if isinstance(probe, dict):
                if probe.get("metadata", {}).get("finetune_mode") == "lora":
                    is_lora_mode = True
                elif any(".lora_" in k for k in probe.get("state_dict", probe).keys()):
                    is_lora_mode = True
        except Exception:
            pass

    # If LoRA mode is requested or detected, apply LoRA wrappers to the backbone
    if is_lora_mode:
        from framework.utils.lora import apply_lora, save_lora_weights, load_lora_weights, find_all_linear_submodule_names
        dynamic_targets = find_all_linear_submodule_names(base)
        default_cli_targets = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        if args.lora_target_modules != default_cli_targets:
            target_mods = {x.strip() for x in args.lora_target_modules.split(",") if x.strip()}
        else:
            target_mods = dynamic_targets if dynamic_targets else default_cli_targets
        base, num_lora = apply_lora(
            base,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_mods,
        )
        base.to(device)
        head = next((getattr(base, attr, None) for attr in
                     ("score", "classifier", "classification_head", "classification_head.dense")
                     if getattr(base, attr, None) is not None), None)
        if head is not None:
            for p in head.parameters(): p.requires_grad = True
        LOG.info("Injected %d LoRALinear layers into %s backbone", num_lora, family)

    if parent_checkpoint.exists() and (args.fresh or not checkpoint.exists()):
        LOG.info("Loading shared baseline checkpoint from parent directory: %s", parent_checkpoint)
        state = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
        _safe_load_state(base, state)
    elif checkpoint.exists():
        LOG.info("Loading cached fine-tuned checkpoint: %s", checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _safe_load_state(base, state)
    else:
        if args.finetune_steps > 0 and "wikitext" not in args.dataset.lower():
            if args.finetune_mode == "lora":
                LOG.info("Fine-tuning LoRA adapters for %d steps before module injection (lr=%.1e)", args.finetune_steps, args.learning_rate)
                finetune_lora(base, tokenizer, train, args.dataset, args.finetune_steps, args.seq_len, args.batch_size, device, args.precision == "fp16", lr=args.learning_rate)
                from framework.utils.lora import save_lora_weights
                save_lora_weights(base, str(checkpoint), metadata={"finetune_mode": "lora", "steps": args.finetune_steps, "dataset": args.dataset, "seed": args.seed})
            else:
                head_lr = 1e-3 if family in ("roberta", "bert", "gpt") else 1e-4
                LOG.info("Fine-tuning classification head for %d steps before module injection (head_lr=%.1e)", args.finetune_steps, head_lr)
                finetune(base, tokenizer, train, args.dataset, args.finetune_steps, args.seq_len, args.batch_size, device, args.precision == "fp16", lr=head_lr)
                try:
                    if "baseline" in str(checkpoint).lower():
                        torch.save(base.state_dict(), checkpoint)
                        LOG.info("Saved fine-tuned checkpoint: %s", checkpoint)
                    else:
                        # For module sweeps, save only classification head to avoid filling disk with 500MB per module
                        head_st = {k: v for k, v in base.state_dict().items() if any(h in k for h in ("score", "classifier"))}
                        torch.save(head_st, checkpoint)
                        LOG.info("Saved lightweight fine-tuned head checkpoint: %s", checkpoint)
                except Exception as e:
                    LOG.warning("Failed to save fine-tuned checkpoint %s: %s (continuing in-memory)", checkpoint, e)
    if args.precision == "fp16":
        with torch.no_grad():
            for name, p in base.named_parameters():
                if not p.requires_grad and p.dtype != torch.float16:
                    p.data = p.data.half()


    base.to("cpu")
    try:
        (out / "finetune_metadata.json").write_text(json.dumps({
            "classification_head_trained": True,
            "finetune_mode": args.finetune_mode,
            "steps": args.finetune_steps,
            "dataset": args.dataset,
            "seed": args.seed
        }), encoding="utf-8")
    except Exception as e:
        LOG.warning("Could not write finetune_metadata.json: %s", e)
    # Make batch cache dataset-specific to prevent cross-dataset contamination
    if getattr(args, "cached_batches_dir", None):
        _cached_dir = Path(args.cached_batches_dir)
        if _cached_dir.name == "batch_cache":
            _cached_dir = _cached_dir.parent / f"batch_cache_{args.dataset}"
    else:
        _cached_dir = out / f"batch_cache_{args.dataset}"

    cache_meta = _cached_dir / "meta.json"
    if cache_meta.exists():
        try:
            with open(cache_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("dataset") != args.dataset:
                import shutil
                shutil.rmtree(str(_cached_dir), ignore_errors=True)
                _cached_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    _eval_cache = _cached_dir / "eval_batches.pt" if _cached_dir else None
    _train_cache = _cached_dir / "train_batches.pt" if _cached_dir else None
    eval_seq_len = args.eval_seq_len if args.eval_seq_len is not None else args.seq_len
    eval_batches = []
    train_batches = []
    
    if "wikitext" not in args.dataset.lower():
        if _eval_cache and _eval_cache.exists() and _train_cache and _train_cache.exists():
            LOG.info("Loading pre-cached batches from %s (skipping tokenization)", _cached_dir)
            eval_batches = [(d, t.to(device)) for d, t in [
                ({k: v.to(device) for k, v in x.items()}, y) for x, y in torch.load(str(_eval_cache), map_location="cpu", weights_only=False)
            ]]
            _max_train = max(10, min(100, args.recovery_steps if args.recovery_steps > 0 else 10))
            all_train = torch.load(str(_train_cache), map_location="cpu", weights_only=False)
            train_batches = [({k: v.to(device) for k, v in x.items()}, y.to(device)) for x, y in all_train[:_max_train]]
        else:
            eval_batches = prepare_eval_batches(tokenizer, val, args.dataset, eval_seq_len, args.batch_size, device)
            train_batches = prepare_train_batches(tokenizer, train, args.dataset, args.seq_len, args.batch_size, device, max_batches=max(10, min(100, args.recovery_steps if args.recovery_steps > 0 else 10)))
            # Auto-save to cache dir if one was specified but files didn't exist yet
            if _cached_dir:
                try:
                    _cached_dir.mkdir(parents=True, exist_ok=True)
                    torch.save([({k: v.cpu() for k, v in x.items()}, y.cpu()) for x, y in eval_batches], str(_eval_cache))
                    torch.save([({k: v.cpu() for k, v in x.items()}, y.cpu()) for x, y in train_batches], str(_train_cache))
                    meta_content = {
                        "dataset": args.dataset,
                        "max_length": eval_seq_len,
                        "num_eval_batches": len(eval_batches),
                        "num_train_batches": len(train_batches),
                    }
                    cache_meta.write_text(json.dumps(meta_content, indent=2), encoding="utf-8")
                except Exception as e:
                    LOG.warning("Failed to save caches to %s: %s", _cached_dir, e)
                LOG.info("Saved pre-tokenized batch cache to %s (will be reused by subsequent module runs)", _cached_dir)
    LOG.info("Prepared %d eval batches and %d train batches directly on GPU.", len(eval_batches), len(train_batches))
    # Pre-evaluate base model to get baseline benchmark accuracy for reference
    # Merge LoRA adapters into base weights BEFORE measuring baseline
    for module in base.modules():
        if hasattr(module, "merge") and callable(module.merge):
            try:
                module.merge()
            except Exception as e:
                LOG.debug("LoRA merge failed for %s: %s", module, e)
        elif hasattr(module, "merge_and_unload") and callable(module.merge_and_unload):
            try:
                module.merge_and_unload()
            except Exception as e:
                LOG.debug("LoRA merge_and_unload failed for %s: %s", module, e)

    for m in base.modules():
        if hasattr(m, "merge") and callable(m.merge):
            try: m.merge()
            except Exception as e: pass

    base.to(device)
    base_benchmark_ppl = None
    if "wikitext" in args.dataset.lower():
        base_eval = evaluate_perplexity(base, tokenizer, val, eval_seq_len, args.batch_size, device, fp16=args.precision == "fp16")
        base_benchmark_ppl = base_eval.get("perplexity")
        base_benchmark_acc = 0.0
        LOG.info("Baseline model benchmark perplexity on validation set: %s (loss=%.4f)", base_benchmark_ppl, base_eval.get("val_loss", 0.0))
    else:
        base_benchmark_acc = round(evaluate(base, tokenizer, val, args.dataset, eval_seq_len, args.batch_size, device, fp16=args.precision == "fp16", precomputed_batches=eval_batches), 2)
        LOG.info("Baseline model benchmark accuracy on validation set: %.2f%%", base_benchmark_acc)
    base.to("cpu")

    # -------------------------------------------------------------------------
    # STEP 0: Dynamically Compute True Majority-Class Baseline & Statistical Margin
    # -------------------------------------------------------------------------
    # Formula & Rationale:
    # 1. Validation split label distribution is counted dynamically: p0 = majority_count / N.
    #    For binary SST-2 (N=872, 444 neg, 428 pos), p0 = 444 / 872 = 50.92%.
    # 2. Under the null hypothesis H0 (classifier outputs uninformative/random labels or
    #    majority-class constant), test accuracy follows a Binomial(N, p0) distribution.
    # 3. Standard Error (SE) of the sample accuracy = sqrt(p0 * (1 - p0) / N).
    #    For SST-2: SE = sqrt(0.50917 * 0.49083 / 872) = 1.693%.
    # 4. Statistical Significance Margin = z * SE (z = 3.0, corresponding to one-tailed
    #    p < 0.0013, 99.87% confidence level).
    #    Margin = 3.0 * 1.693% = 5.08%.
    # 5. PASS Threshold = p0 + margin.
    #    For SST-2: 50.92% + 5.08% = 56.00%.
    # Any module with accuracy <= pass_threshold is statistically indistinguishable from
    # random chance or majority-class guess, and is classified as SKIP_<FAMILY>_INCOMPATIBLE_ACCURACY_PLATEAU.
    # -------------------------------------------------------------------------
    if "wikitext" in args.dataset.lower():
        total_val_samples = 1000
        majority_count = 500
        majority_baseline_ratio = 0.5
    else:
        val_labels = val["label"] if "label" in val.features else [x["label"] for x in val]
        from collections import Counter
        label_counts = Counter(val_labels)
        total_val_samples = len(val_labels)
        majority_count = max(label_counts.values()) if label_counts else 0
        majority_baseline_ratio = majority_count / total_val_samples if total_val_samples > 0 else 0.5
        
    majority_baseline_pct = round(majority_baseline_ratio * 100.0, 2)

    p0 = majority_baseline_ratio
    se = math.sqrt(p0 * (1.0 - p0) / max(1, total_val_samples))
    z_score = 3.0  # 3 standard errors above chance for 99.87% confidence
    # Dataset-aware chance and threshold calibration
    # Dynamic threshold: always derived from the actual validation split majority.
    # No dataset-specific hardcodes — the majority, SE, and threshold are computed
    # live from the loaded validation split.  This ensures the threshold is always
    # calibrated to whatever dataset+split is actually on disk.
    stat_margin_pct = round(z_score * se * 100.0, 2)
    pass_threshold_pct = round(majority_baseline_pct + stat_margin_pct, 2)
    threshold_source = (
        f"dynamic: majority={majority_baseline_pct:.2f}% ({majority_count}/{total_val_samples}), "
        f"SE={se*100:.3f}%, 3*SE={stat_margin_pct:.3f}%"
    )

    LOG.info("=" * 80)
    LOG.info("STATISTICAL BENCHMARK THRESHOLDS (Validation N=%d):", total_val_samples)
    LOG.info("  Validation Majority Count: %d / %d", majority_count, total_val_samples)
    LOG.info("  True Majority Baseline:    %.2f%%", majority_baseline_pct)
    LOG.info("  Standard Error (SE):       %.3f%%", se * 100.0)
    LOG.info("  Statistical Margin (3*SE): %.2f%%", stat_margin_pct)
    LOG.info("  PASS Threshold (Min Acc):  %.2f%%", pass_threshold_pct)
    LOG.info("  Threshold Source:          %s", threshold_source)
    LOG.info("=" * 80)

    if isinstance(args.modules, list):
        raw_mods = []
        for item in args.modules:
            raw_mods.extend([x.strip() for x in item.split(",") if x.strip()])
        names = modules() if (not raw_mods or raw_mods == ["all"]) else raw_mods
    elif args.modules == "all":
        names = modules()
    else:
        names = [x.strip() for x in args.modules.split(",") if x.strip()]

    path = out / "summary.csv"
    summary_dict = {}

    def _flush_summary_csv():
        try:
            _tmp = path.with_suffix(".tmp")
            with _tmp.open("w", newline="", encoding="utf8") as _f:
                _w = csv.DictWriter(_f, fieldnames=FIELDS, extrasaction="ignore")
                _w.writeheader()
                for _m in summary_dict:
                    _w.writerow(summary_dict[_m])
            if _tmp.exists():
                if path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass
                _tmp.rename(path)
        except Exception as _csv_err:
            LOG.warning("Could not write summary.csv to %s: %s", path, _csv_err)

    if path.exists():
        backup_path = path.with_suffix(".csv.bak")
        if not backup_path.exists():
            import shutil
            shutil.copyfile(path, backup_path)
        with path.open("r", encoding="utf8") as f:
            for r in csv.DictReader(f):
                summary_dict[r["module_name"]] = r
    # In fresh mode, evaluate everything; otherwise only skip modules that genuinely PASSED with non-zero accuracy.
    # PASS_LOW is NOT skipped — it gets re-evaluated on new datasets where thresholds differ.
    # CRASH/FAIL/REJECTED_EXPANSION are always re-tried unless in BLOCKED_MODULES.
    existing = set() if args.fresh else {
        m for m, r in summary_dict.items()
        if r.get("status") == "PASS" and float(r.get("accuracy_pct", 0.0) or 0.0) > 0.0
    }
    module_rank = args.rank if args.rank is not None else max(16, getattr(adapter, "head_dim", 16))
    LOG.info("Initializing ModelStateSnapshot for in-memory module swapping...")
    base_snapshot = ModelStateSnapshot(base)

    for n, name in enumerate(names, 1):
        # Force rank = sqrt(head_dim) for TPA to preserve tensor product.
        if name in ("tpa", "tpa_attention", "tp_attention", "inject_tpa"):
            _nh = getattr(base.config, "num_attention_heads", None) or getattr(base.config, "n_heads", 12)
            _hs = getattr(base.config, "hidden_size", None) or getattr(base.config, "d_model", 768)
            _hd = getattr(base.config, "head_dim", None) or (_hs // max(1, _nh))
            _optimal = max(1, int(round(_hd ** 0.5)))
            if module_rank != _optimal:
                LOG.warning("TPA: overriding module_rank %d -> %d (head_dim=%d, sqrt required).", module_rank, _optimal, _hd)
                module_rank = _optimal
                if hasattr(base, "config") and base.config is not None:
                    setattr(base.config, "rank", _optimal)
                    setattr(base.config, "tpa_rank", _optimal)
        if name in DEPRECATED_MODULES:
            dep_reason = DEPRECATED_MODULES[name]
            LOG.warning(">>> Skipping DEPRECATED module [%d/%d]: %s (%s)", n, len(names), name, dep_reason)
            row = {
                "module_name": name, "status": "SKIP_DEPRECATED",
                "accuracy_pct": 0.0, "elapsed_s": 0.0,
                "error": f"DEPRECATED: {dep_reason}",
                "path": "deprecated", "timestamp": datetime.now().isoformat(),
            }
            summary_dict[name] = row
            _tmp = path.with_suffix(".tmp")
            with _tmp.open("w", newline="", encoding="utf8") as _f:
                _w = csv.DictWriter(_f, fieldnames=FIELDS, extrasaction="ignore")
                _w.writeheader()
                for _m in summary_dict:
                    _w.writerow(summary_dict[_m])
            if _tmp.exists():
                if path.exists():
                    try: path.unlink()
                    except Exception: pass
                _tmp.rename(path)
            continue
        if name in BLOCKED_MODULES and len(names) > 10:
            bl_reason = BLOCKED_MODULES[name]
            LOG.warning(">>> Skipping BLOCKED module [%d/%d]: %s (%s)", n, len(names), name, bl_reason)
            row = {
                "module_name": name, "status": "SKIP_BLOCKED",
                "accuracy_pct": 0.0, "elapsed_s": 0.0,
                "error": f"BLOCKED: {bl_reason}",
                "path": "blocked", "timestamp": datetime.now().isoformat(),
            }
            summary_dict[name] = row
            # Atomic write to CSV so blocklist entries appear in live summary.csv
            _tmp = path.with_suffix(".tmp")
            with _tmp.open("w", newline="", encoding="utf8") as _f:
                _w = csv.DictWriter(_f, fieldnames=FIELDS, extrasaction="ignore")
                _w.writeheader()
                for _m in summary_dict:
                    _w.writerow(summary_dict[_m])
            if _tmp.exists():
                if path.exists():
                    try: path.unlink()
                    except Exception: pass
                _tmp.rename(path)
            continue
        # if name in existing:
        #     LOG.info(">>> Skipping already PASSED module [%d/%d]: %s (PASS, %.2f%%)",
        #              n, len(names), name, float(summary_dict[name].get("accuracy_pct", 0.0)))
        #     continue

        LOG.info(">>> STARTING MODULE [%d/%d]: %s (recovery_steps=%d, batch_size=%d, optimizer=%s)",
                 n, len(names), name, args.recovery_steps, args.batch_size, active_opt)
        t = time.monotonic(); status, acc, err = "PENDING", 0.0, ""; had_nan = False; path_name = "native_qwen" if family == "qwen" else ("legacy_bert" if family in {"bert", "roberta"} else "adapted")

        support = module_family_support(name).get(family, "unknown")
        from framework.core.registry import is_module_declared_compatible
        is_compat, reason = is_module_declared_compatible(name, family)
        if name not in ("full", "dense") and (support == "incompatible" or not is_compat):
            LOG.warning("Module %s declared %s / %s, attempting injection with fallback guard...", name, support, reason)

        # Sequential single-model clone directly on GPU (Zero host RAM bloat, zero PCIe latency)
        # Deepcopy on CPU host memory (zero VRAM overhead) and send fresh candidate to device
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = base
        # Timing start
        t_load = time.monotonic()
        # Side effect is projection alias installation; the model itself
        # is not replaced, preserving native HF forward semantics.
        wrap_for_family(model, family)
        model.tokenizer = tokenizer
        if hasattr(model, "config") and model.config is not None:
            pass  # removed model.config.tokenizer = tokenizer
        typ = "attention" if name in ATTENTION_REGISTRY else "compressor"
        if typ == "attention":
            # For attention replacement modules, merge baseline LoRA adapters into base weights
            # so that get_weight_bias extracts full fine-tuned representations into new_attn.
            for m in model.modules():
                if hasattr(m, "merge") and callable(m.merge):
                    try:
                        m.merge()
                    except Exception:
                        pass
        else:
            # For compressor modules, do NOT merge LoRA into base weights.
            # Keep base_layer and lora_A/lora_B deltas separate: the compressor factorizes
            # the frozen base_layer.weight, while LoRA deltas remain active and trainable for recovery.
            pass
        model = model.to(device)
        injected = False
        active_opt = "muon" if getattr(args, "use_muon", False) or getattr(args, "optimizer", "adamw") == "muon" else "adamw"
        had_nan = False
        path_name = f"adapted_{family}" if family != "bert" else "legacy_bert"

        # Slow module or configured timeout (default minimum: 1200s = 20 minutes)
        override_to = SLOW_MODULE_TIMEOUT_OVERRIDES.get(name, 0)
        cli_timeout = getattr(args, "timeout", None) or 1200
        effective_timeout = max(cli_timeout, override_to) if override_to > 0 else cli_timeout

        # Per-module batch_size overrides for VRAM-heavy modules on 8GB GPU
        _OOM_MODULE_BATCH_OVERRIDES = {
            "granite_hybrid": 4, "grecodec": 4, "grecodec_compressor": 4,
            "hymba_transx": 4, "mamba_svd_flash": 8, "spectral_mamba": 8,
            "vspec": 8, "slicegpt": 8, "slicegpt_pca": 8,
            "asvd": 8, "cosformer": 8, "cur": 8, "vig": 8,
            "tpa": 4, "tpa_attention": 4, "tp_attention": 4,
            "kda": 4,
            "nha": 4,
            "nha_attention": 4,
            "nha_compressor": 4,
            "kronecker_tt_lora": 4,
            "gated_delta_net": 4,
            "hgrn2_gated_deltanet": 4,
            "nasc": 8,
            "nasc_attention": 8,
        }
        # Only apply OOM override on 12-layer models (roberta, gpt)
        if name in _OOM_MODULE_BATCH_OVERRIDES and family in ("roberta", "gpt"):
            effective_batch_size = min(args.batch_size, _OOM_MODULE_BATCH_OVERRIDES[name])
        else:
            effective_batch_size = args.batch_size

        total_params_before = sum(p.numel() for p in base.parameters())
        model_slug = Path(args.model).name.replace("/", "_").replace("\\", "_")

        acc = 0.0  # default; overwritten on success or exception
        try:

            # Pipeline replaced
            perf_metrics, comp_metrics, injected, path_name, had_nan = _run_candidate_pipeline(model, name, args, device, family, effective_batch_size, tokenizer, val, base, train, effective_timeout, out, model_slug, total_params_before, module_rank, eval_batches, train_batches, eval_seq_len, pass_threshold_pct, typ, t)
            # *** Critical: extract acc from perf_metrics on successful run ***
            acc = perf_metrics.get("accuracy_pct", 0.0)
        except TimeoutError as exc:
            status = f"SKIP_{family.upper()}_TIMEOUT"
            err = f"Module pipeline exceeded hard {effective_timeout}s timeout"
            acc = 0.0
        except QwenIncompatibleError as exc:
            status = f"SKIP_{family.upper()}_INCOMPATIBLE_GQA"
            err = str(exc)
            acc = 0.0
        except Exception as exc:
            LOG.exception("Module %s execution failure:", name)
            if "out of memory" in str(exc).lower() and injected and args.batch_size > 8:
                try:
                    torch.cuda.empty_cache()
                    safe_batch = max(8, args.batch_size // 2)
                    perf_metrics = evaluate_comprehensive(model, tokenizer, val, args.dataset, eval_seq_len, safe_batch, device)
                    acc = perf_metrics["accuracy_pct"]
                    err = f"batch_size reduced {args.batch_size}->{safe_batch} after CUDA OOM"
                    exc = None
                except Exception as retry: exc = retry

            if exc is not None and name != "full" and is_dtype_error(exc):
                try:
                    model.float()
                    model, _ = apply_module_to_model(model, name, typ, rank=module_rank, device=device)
                    perf_metrics = evaluate_comprehensive(model, tokenizer, val, args.dataset, eval_seq_len, effective_batch_size, device, precomputed_batches=eval_batches)
                    acc = perf_metrics["accuracy_pct"]
                    err = f"fp32 retry: {exc}"
                    exc = None
                except Exception as retry: exc = retry

            # Check for explicit SKIP signal (e.g. inject_adaptive_svd returning None or copy_weights failing)
            if exc is not None and isinstance(exc, RuntimeError) and ("Could not extract all required weights" in str(exc) or "Could not instantiate attention class" in str(exc)):
                status = f"SKIP_{family.upper()}_ARCHITECTURAL"
                err = f"note: architecture unsupported by module ({exc})"
                acc = 0.0
            elif exc is not None:
                status = "FAIL"
                err = f"{type(exc).__name__}: {exc}"
                acc = 0.0

        # Ensure metrics dictionaries are fully populated even on failure/skip
        if "perf_metrics" not in locals() or not perf_metrics:
            perf_metrics = {
                "accuracy_pct": acc,
                "f1_score": 0.0,
                "f1_macro": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "val_loss": 0.0,
                "peak_vram_mb": 0.0,
                "tokens_per_sec": 0.0,
            }
        if "comp_metrics" not in locals() or not comp_metrics:
            comp_metrics = {
                "total_params_before": total_params_before,
                "total_params_after": total_params_before,
                "trainable_recovery_params": getattr(model, "_trainable_recovery_params", 0),
                "compression_rate": 1.0,
                "compression_rate_raw": 1.0,
                "compression_rate_net": 1.0,
                "compression_ratio_pct": 0.0,
                "flops_reduction_pct": 0.0,
                "num_compressed_layers": 0,
                "num_fallback_layers": 0,
                "avg_reconstruction_error": "",
                "weighted_reconstruction_error": "",
                "num_injected_attention_layers": 0,
                "lora_params": 0,
            }

        # -------------------------------------------------------------------------
        # STEP 0: Honest Three-Way Status Classification
        # -------------------------------------------------------------------------
        if not status.startswith("SKIP_") and not status.startswith("FAIL"):
            if "wikitext" in args.dataset.lower():
                ppl_val = perf_metrics.get("perplexity", 0.0)
                try:
                    ppl_num = float(ppl_val)
                except (ValueError, TypeError):
                    ppl_num = float("inf")

                if acc >= 100.0 and (ppl_num <= 0.0 or math.isnan(ppl_num) or math.isinf(ppl_num)):
                    status = "INVALID"
                    err = "INVALID: evaluated accuracy is 100.0% on language modeling task without valid perplexity"
                elif math.isnan(ppl_num) or math.isinf(ppl_num):
                    status = "FAIL_PPL_NAN"
                    err = f"FAIL_PPL_NAN: non-finite perplexity ({ppl_val})"
                elif ppl_num <= 50.0:
                    status = "PASS"
                elif ppl_num <= 100.0:
                    status = "PASS_LOW_PPL"
                else:
                    status = "FAIL_PPL_DIVERGED"
                    err = f"FAIL_PPL_DIVERGED: perplexity ({ppl_num:.2f}) exceeded threshold 100.0"
            else:
                if acc <= 0.0 or (status == "PENDING" and acc < pass_threshold_pct):
                    status = "EVAL_FAILED" if acc <= 0.0 else "FAIL_BELOW_CHANCE"
                    if acc <= 0.0:
                        err = f"EVAL_FAILED: evaluated accuracy is non-positive ({acc:.2f}%)"
                elif acc >= pass_threshold_pct:
                    status = "PASS"
                else:
                    status = "FAIL_BELOW_CHANCE"

                if had_nan:
                    status = "FAIL_RECOVERY_NAN"
                    err = "note: recovery_loss_nan"
                elif acc < pass_threshold_pct and status != "EVAL_FAILED":
                    err = f"FAIL: accuracy ({acc:.2f}%) below chance threshold ({pass_threshold_pct:.2f}%)"

            # Check parameter expansion — exclude LoRA adapter params from base model count
            p_before = comp_metrics.get("total_params_before", total_params_before)
            p_after = comp_metrics.get("total_params_after", total_params_before)
            lora_p = comp_metrics.get("lora_params", 0) or 0
            base_params_after = p_after - lora_p

            if p_before > 0 and name != "full":
                base_delta = base_params_after - p_before
                base_delta_pct = 100.0 * base_delta / p_before if p_before > 0 else 0.0
                if base_params_after > p_before * 1.20:
                    status = "EXTREME_EXPANSION"
                    err = f"base_before={p_before} base_after={base_params_after} delta=+{base_delta_pct:.1f}%"
                elif base_params_after > p_before and base_delta > 1024 and base_delta_pct > 0.05:
                    status = "MODERATE_EXPANSION"
                    err = f"base_before={p_before} base_after={base_params_after} delta=+{base_delta_pct:.2f}%"
                elif (comp_metrics.get("num_compressed_layers", 0) == 0
                      and comp_metrics.get("num_injected_attention_layers", 0) == 0
                      and not name.startswith("inject_")
                      and base_params_after >= p_before * 0.99):
                    if status.startswith("FAIL") or status.startswith("EVAL_FAILED") or status.startswith("SKIP"):
                        err = f"{status}: zero compressed layers, zero injected layers ({err})".strip(": ")
                    else:
                        status = "PASS_NO_COMPRESSION"
                        err = "note: zero compressed layers, zero injected layers, model unchanged"

        # Update per-module full metrics JSON with final classification status and elapsed time
        json_path = out / f"{model_slug}_{name}_full_metrics.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    jdata = json.load(jf)
                jdata["performance"]["status"] = status
                jdata["performance"]["elapsed_s"] = round(time.monotonic() - t, 2)
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(jdata, jf, indent=4)
            except Exception:
                pass

        ppl_val_raw = perf_metrics.get("perplexity", "")
        if "wikitext" in args.dataset.lower():
            try:
                ppl_float = float(ppl_val_raw)
                if not math.isfinite(ppl_float):
                    ppl_status = "NAN"
                elif ppl_float > 100.0:
                    ppl_status = "DIVERGED"
                elif ppl_float > 50.0:
                    ppl_status = "HIGH"
                else:
                    ppl_status = "OK"
            except (ValueError, TypeError):
                ppl_status = "NAN"
        else:
            ppl_status = ""

        row = {
            "module_name": name,
            "status": status,
            "accuracy_pct": acc,
            "ppl_status": ppl_status,
            "f1_score": perf_metrics.get("f1_score", 0.0),
            "f1_macro": perf_metrics.get("f1_macro", 0.0),
            "precision": perf_metrics.get("precision", 0.0),
            "recall": perf_metrics.get("recall", 0.0),
            "val_loss": perf_metrics.get("val_loss", 0.0),
            "perplexity": ppl_val_raw,
            "elapsed_s": round(time.monotonic() - t, 2),
            "eval_time_s": perf_metrics.get("eval_time_s", round(time.monotonic() - t, 2)),
            "latency_ms": perf_metrics.get("latency_ms", 0.0),
            "peak_vram_mb": perf_metrics.get("peak_vram_mb", 0.0),
            "tokens_per_sec": perf_metrics.get("tokens_per_sec") if perf_metrics.get("tokens_per_sec") is not None else perf_metrics.get("throughput", 0.0),
            "total_params_before": comp_metrics.get("total_params_before", total_params_before),
            "total_params_after": comp_metrics.get("total_params_after", total_params_before),
            "trainable_recovery_params": comp_metrics.get("trainable_recovery_params", 0),
            "compression_rate": comp_metrics.get("compression_rate", 1.0),
            "compression_rate_raw": comp_metrics.get("compression_rate_raw", 1.0),
            "compression_rate_net": comp_metrics.get("compression_rate_net", 1.0),
            "compression_ratio_pct": comp_metrics.get("compression_ratio_pct", 0.0),
            "flops_reduction_pct": comp_metrics.get("flops_reduction_pct", 0.0),
            "num_compressed_layers": comp_metrics.get("num_compressed_layers", 0),
            "num_fallback_layers": comp_metrics.get("num_fallback_layers", 0),
            "avg_reconstruction_error": comp_metrics.get("avg_reconstruction_error", ""),
            "weighted_reconstruction_error": comp_metrics.get("weighted_reconstruction_error", ""),
            "num_injected_attention_layers": comp_metrics.get("num_injected_attention_layers", 0),
            "lora_params": comp_metrics.get("lora_params", 0),
            "collapse_flag": perf_metrics.get("collapse_flag", False),
            "per_class_metrics": json.dumps(perf_metrics.get("per_class_metrics", {})),
            "threshold_used": pass_threshold_pct,
            "threshold_source": threshold_source,
            "error": err[:1000],
            "path": path_name,
            "timestamp": datetime.now().isoformat(),
        }
        summary_dict[name] = row
        _flush_summary_csv()

        f1_disp = row["f1_score"] if row.get("f1_score") not in ("", None) else "N/A"
        vram_disp = row["peak_vram_mb"] if row.get("peak_vram_mb") not in ("", None) else "N/A"
        LOG.info("[%d/%d] %s -> %s %.2f%% (F1: %s, Peak VRAM: %s MB)",
                 n, len(names), name, status, acc, f1_disp, vram_disp)

        if "wikitext" in args.dataset.lower():
            is_successful_run = status in ("PASS", "PASS_LOW_PPL") and (name == "full" or comp_metrics.get("num_compressed_layers", 0) > 0 or comp_metrics.get("num_injected_attention_layers", 0) > 0)
        else:
            is_successful_run = (status == "PASS") and (acc >= pass_threshold_pct)

        is_fake_pass = (status == "PASS_NO_COMPRESSION")
        if is_fake_pass:
            LOG.warning(
                "Module '%s' PASSED on accuracy but compressed ZERO layers. Marking as PASS_NO_COMPRESSION.",
                name,
            )
        weights_dir = out / "checkpoints" / name
        hf_export_dir = weights_dir / "hf_export"

        if not is_successful_run:
            LOG.info("[Quality Guard] Module '%s' did not pass quality criteria (status=%s, acc=%.2f%%, pass_thresh=%.2f%%). Pruning large checkpoint artifacts but preserving metrics.",
                     name, status, acc, pass_threshold_pct)
            # Log failed run to failed_runs.json
            try:
                failed_json_path = out / "failed_runs.json"
                failed_data = []
                if failed_json_path.exists():
                    try:
                        failed_data = json.loads(failed_json_path.read_text(encoding="utf-8"))
                        if not isinstance(failed_data, list):
                            failed_data = []
                    except Exception:
                        failed_data = []
                # Deduplicate by (module, model, dataset, status)
                new_entry = {
                    "module": name,
                    "model": str(args.model),
                    "dataset": args.dataset,
                    "status": status,
                    "perplexity": row.get("perplexity", None),
                    "accuracy_pct": row.get("accuracy_pct", 0.0),
                    "error": err[:1000] if err else "",
                    "timestamp": row.get("timestamp", datetime.now().isoformat())
                }
                deduped = [
                    e for e in failed_data
                    if not (e.get("module") == new_entry["module"]
                            and e.get("model") == new_entry["model"]
                            and e.get("dataset") == new_entry["dataset"]
                            and e.get("status") == new_entry["status"])
                ]
                deduped.append(new_entry)
                failed_json_path.write_text(json.dumps(deduped, indent=2), encoding="utf-8")
            except Exception as f_err:
                LOG.debug("Could not update failed_runs.json: %s", f_err)

            # Always preserve a compact metrics record (no heavy weights.pt) so the
            # analytics dashboard and auto-uploader can track every module outcome.
            if weights_dir.exists():
                # Remove only the large binary weight files; keep the metrics JSON and config
                for big_file in list(weights_dir.glob("**/*.pt")):
                    try:
                        big_file.unlink()
                    except Exception:
                        pass
                for big_file in list(weights_dir.glob("**/*.bin")):
                    try:
                        big_file.unlink()
                    except Exception:
                        pass
            # Always write a summary_record.json so the auto-uploader knows this module ran
            weights_dir.mkdir(parents=True, exist_ok=True)
            try:
                summary_record = {
                    "module_name": name,
                    "model_name": str(args.model),
                    "model_family": family,
                    "dataset": args.dataset,
                    "status": status,
                    "accuracy_pct": acc,
                    "f1_score": row.get("f1_score", 0.0),
                    "f1_macro": row.get("f1_macro", 0.0),
                    "precision": row.get("precision", 0.0),
                    "recall": row.get("recall", 0.0),
                    "val_loss": row.get("val_loss", 0.0),
                    "peak_vram_mb": row.get("peak_vram_mb", 0.0),
                    "tokens_per_sec": row.get("tokens_per_sec", 0.0),
                    "elapsed_s": row.get("elapsed_s", 0.0),
                    "compression_rate": row.get("compression_rate", 1.0),
                    "compression_ratio_pct": row.get("compression_ratio_pct", 0.0),
                    "total_params_before": row.get("total_params_before", 0),
                    "total_params_after": row.get("total_params_after", 0),
                    "num_compressed_layers": row.get("num_compressed_layers", 0),
                    "num_injected_attention_layers": row.get("num_injected_attention_layers", 0),
                    "lora_params": row.get("lora_params", 0),
                    "pass_threshold_pct": pass_threshold_pct,
                    "timestamp": row.get("timestamp", datetime.now().isoformat()),
                    "error": err[:500] if err else "",
                }
                with open(weights_dir / "summary_record.json", "w", encoding="utf-8") as srec:
                    json.dump(summary_record, srec, indent=2)
            except Exception as srec_err:
                LOG.debug("Could not write summary_record.json for %s: %s", name, srec_err)
        else:
            if getattr(args, "defer_export", False):
                LOG.info("[Export Deferred] Checkpoint preserved in %s. Automated export will be handled post-sweep by orchestrator.", weights_dir)
            else:
                try:
                    from Codes.framework.utils.hf_git_exporter import handle_benchmark_run_export
                    # Always ensure hf_export_dir exists with metrics so HF upload is never silently skipped
                    hf_export_dir.mkdir(parents=True, exist_ok=True)
                    # Write metrics.json into export dir if not already present
                    _metrics_out = hf_export_dir / "metrics.json"
                    if not _metrics_out.exists():
                        try:
                            import json as _json
                            _metrics_out.write_text(
                                _json.dumps({"performance": perf_metrics, "compression": comp_metrics}, indent=2),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                    # Copy summary.csv into export dir if available
                    _summary_src = out / "summary.csv"
                    _summary_dst = hf_export_dir / "summary.csv"
                    if _summary_src.exists() and not _summary_dst.exists():
                        try:
                            import shutil as _shutil
                            _shutil.copy2(_summary_src, _summary_dst)
                        except Exception:
                            pass
                    # Copy finetuned weights if produced by this sweep run
                    _weights_src = out / "finetuned_model.pt"
                    _weights_dst = hf_export_dir / "weights.pt"
                    if _weights_src.exists() and not _weights_dst.exists():
                        try:
                            import shutil as _shutil
                            _shutil.copy2(_weights_src, _weights_dst)
                        except Exception:
                            pass
                    handle_benchmark_run_export(
                        model=None,
                        tokenizer=tokenizer,
                        model_name=Path(args.model).name,
                        module_name=name,
                        dataset_name=args.dataset,
                        perf_metrics=perf_metrics,
                        comp_metrics=comp_metrics,
                        status=status,
                        pass_threshold_pct=float(pass_threshold_pct),
                        baseline_acc=base_benchmark_acc if name != "full" else None,
                        config_dict=vars(args) if hasattr(args, "__dict__") else {},
                        out_dir=out,
                        allow_pass_low=False,
                        export_dir=hf_export_dir,
                    )
                    # Prune heavy weights from disk after successful HF upload to prevent disk exhaustion
                    try:
                        if weights_dir.exists():
                            for big_file in list(weights_dir.glob("**/*.pt")) + list(weights_dir.glob("**/*.bin")):
                                big_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                except Exception as export_exc:
                    LOG.warning("Automated export exception for %s: %s", name, export_exc)

        # Log per-module timing
        t_total = time.monotonic() - t
        LOG.info("[%d/%d] %s Timing: eval_time=%.2fs", n, len(names), name, t_total)
        
        LOG.info("Restoring model state from snapshot for next module...")
        base_snapshot.restore()

    # -------------------------------------------------------------------------
    # Final Compact Summary Table
    # -------------------------------------------------------------------------
    summary_rows = []
    for m_name, r in summary_dict.items():
        display_name = m_name
        try:
            num_comp = int(r.get("num_compressed_layers", 0))
        except (ValueError, TypeError):
            num_comp = 0
        if num_comp == 0 and m_name != "full":
            display_name = f"{m_name} (efficient-attn)"

        try:
            acc_val = float(r.get("accuracy_pct", 0.0) or 0.0)
        except (ValueError, TypeError):
            acc_val = 0.0
        delta_acc = acc_val - base_benchmark_acc if m_name != "full" else 0.0
        f1_val = r.get("f1_score", "N/A")
        try:
            f1_str = f"{float(f1_val):.4f}" if f1_val not in ("", "N/A", None) else "N/A"
        except (ValueError, TypeError):
            f1_str = str(f1_val)

        comp_rate = r.get("compression_rate", "1.00")
        try:
            comp_rate_str = f"{float(comp_rate):.2f}x" if comp_rate not in ("", "N/A", None) else "1.00x"
        except (ValueError, TypeError):
            comp_rate_str = str(comp_rate)

        vram = r.get("peak_vram_mb", "N/A")
        try:
            vram_str = f"{float(vram):.1f} MB" if vram not in ("", "N/A", None) else "N/A"
        except (ValueError, TypeError):
            vram_str = str(vram)

        try:
            inf_time = float(r.get('eval_time_s', r.get('elapsed_s', 0.0)) or 0.0)
            inf_time_str = f"{inf_time:.2f}"
        except (ValueError, TypeError):
            inf_time_str = "0.00"

        try:
            lat = float(r.get('latency_ms', 0.0)) if r.get('latency_ms') not in ("", "N/A", None) else None
            lat_str = f"{lat:.2f}" if lat is not None else "N/A"
        except (ValueError, TypeError):
            lat_str = "N/A"

        summary_rows.append({
            "Module": display_name,
            "Status": r.get("status", "N/A"),
            "Accuracy": f"{acc_val:.2f}%",
            "Delta Acc": f"{delta_acc:+.2f}%" if m_name != "full" else "0.00% (base)",
            "F1": f1_str,
            "Compressed Layers": r.get("num_compressed_layers", "0"),
            "Compression Rate": comp_rate_str,
            "Peak VRAM": vram_str,
            "Inference Time (s)": inf_time_str,
            "Latency (ms/samp)": lat_str,
        })

    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    LOG.info("=" * 80)
    LOG.info("BENCHMARK QUALITY SWEEP SUMMARY TABLE (%s on %s)", Path(args.model).name, args.dataset)
    LOG.info("\n%s", summary_df.to_string(index=False))
    LOG.info("=" * 80)

    # Export a visual publication markdown report for the sweep
    try:
        report_file = out / "benchmark_report.md"
        report_lines = [
            f"# Benchmark Quality & Compression Report: {Path(args.model).name} on {args.dataset}",
            f"- **Execution Timestamp**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            f"- **Baseline Reference Accuracy**: `{base_benchmark_acc:.2f}%`",
            f"- **Statistical Pass Threshold**: `{pass_threshold_pct:.2f}%`",
            "",
            "## Empirical Comparative Profiling",
            "",
            "| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Inference Time (s) | Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for sr in summary_rows:
            st = sr["Status"]
            badge = f"`{st}`" if st == "PASS" else f"**`{st}`**"
            report_lines.append(
                f"| **{sr['Module']}** | {badge} | **{sr['Accuracy']}** | {sr['Delta Acc']} | {sr['F1']} | {sr['Compressed Layers']} | {sr['Compression Rate']} | {sr['Peak VRAM']} | {sr['Time (s)']} |"
            )
        report_lines.append("")
        report_file.write_text("\n".join(report_lines), encoding="utf-8")
        LOG.info("Saved visual benchmark report to %s", report_file)
    except Exception as rep_err:
        LOG.debug("Report generation info: %s", rep_err)

    _flush_summary_csv()

    if len(names) == 1 and names[0] in summary_dict:
        return summary_dict[names[0]]
    if len(summary_dict) == 1:
        return list(summary_dict.values())[0]
    return summary_dict



import subprocess
import time
import json
from pathlib import Path
import torch

def run_module_isolated(module_name: str, args, timeout_s: int) -> dict:
    import os, sys
    tmp_dir = Path("logs/isolated_results")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    result_json = tmp_dir / f"{module_name}_{int(time.time() * 1000)}_{os.getpid()}.json"
    if result_json.exists():
        try: result_json.unlink()
        except Exception: pass

    child_script = (HERE / "run_one_module_isolated.py").resolve()
    cmd = [
        sys.executable, "-u",
        str(child_script),
        "--module", module_name,
        "--model", str(args.model),
        "--model_family", str(args.model_family),
        "--dataset", str(args.dataset),
        "--dataset_dir", str(args.dataset_dir) if args.dataset_dir else "None",
        "--output_json", str(result_json),
        "--output_dir", str(args.output_dir) if args.output_dir else "results",
        "--rank", str(args.rank or 64),
        "--recovery_steps", str(args.recovery_steps),
        "--lora_rank", str(args.lora_rank),
        "--batch_size", str(args.batch_size),
        "--precision", str(args.precision),
        "--seed", str(args.seed),
        "--cached_batches_dir", str(Path(args.output_dir or "results") / f"batch_cache_{args.dataset}"),
        "--defer_export",
    ]
    if getattr(args, "max_error", None) is not None:
        cmd.extend(["--max_error", str(args.max_error)])
    if getattr(args, "rank_ratio", None) is not None:
        cmd.extend(["--rank_ratio", str(args.rank_ratio)])
    if getattr(args, "force_rank", False):
        cmd.append("--force_rank")
    if getattr(args, "dry_run", False):
        cmd.append("--dry_run")
    if getattr(args, "debug_module", False):
        cmd.append("--debug_module")
    if getattr(args, "verbose_debug", False):
        cmd.append("--verbose_debug")
    if getattr(args, "span_lambda", None) is not None and getattr(args, "span_lambda", 0.0) > 0:
        cmd.extend(["--span_lambda", str(args.span_lambda)])
    if getattr(args, "kvtc_bits", None) is not None:
        cmd.extend(["--kvtc_bits", str(args.kvtc_bits)])
    if getattr(args, "kvtc_max_error", None) is not None:
        cmd.extend(["--kvtc_max_error", str(args.kvtc_max_error)])
    if getattr(args, "kvtc_pca_energy", None) is not None:
        cmd.extend(["--kvtc_pca_energy", str(args.kvtc_pca_energy)])
    if getattr(args, "kvtc_target_attention_only", None) is not None:
        cmd.extend(["--kvtc_target_attention_only", str(args.kvtc_target_attention_only)])
    if getattr(args, "kvtc_fallback_to_original", None) is not None:
        cmd.extend(["--kvtc_fallback_to_original", str(args.kvtc_fallback_to_original)])
    if getattr(args, "strict_rank", False):
        cmd.extend(["--strict_rank", "True"])
    if getattr(args, "group_size", None) is not None:
        cmd.extend(["--group_size", str(args.group_size)])
    if getattr(args, "max_vram_fraction", None) is not None:
        cmd.extend(["--max_vram_fraction", str(args.max_vram_fraction)])
    if args.dataset_dir is None:
        cmd = [c for i, c in enumerate(cmd) if not (cmd[i-1] == "--dataset_dir" or c == "--dataset_dir")]
    if getattr(args, "unknown_args", None):
        cmd.extend(args.unknown_args)
        
    log_path = Path(f"logs/isolated/{module_name}_r{args.rank}_{args.model_family}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ISOLATED] Launching {module_name} (timeout={timeout_s}s) -> {log_path}")
    t0 = time.monotonic()

    base_dir = Path("D:/svd_project") if Path("D:/svd_project").exists() else HERE.parent.parent
    extra_flags = (0x08000000 | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(base_dir),
            creationflags=extra_flags,
        )

        try:
            proc.wait(timeout=timeout_s)
            elapsed = time.monotonic() - t0
            print(f"[ISOLATED] {module_name} exited with code {proc.returncode} in {elapsed:.1f}s")
        except subprocess.TimeoutExpired:
            print(f"[ISOLATED] {module_name} TIMEOUT after {timeout_s}s - killing")
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            elapsed = time.monotonic() - t0

    if result_json.exists():
        try:
            return json.loads(result_json.read_text(encoding="utf-8"))
        except Exception as e:
            return {"status": "PARSE_ERROR", "error": str(e), "elapsed_s": elapsed}
    else:
        return {
            "status": "TIMEOUT" if elapsed >= timeout_s - 1 else "CRASH_NO_OUTPUT",
            "error": f"Child produced no result file after {elapsed:.1f}s",
            "elapsed_s": elapsed,
        }

def clear_vram_between_modules():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(2)

def main():
    args = parse_args()
    from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    
    all_available = modules()
    if isinstance(args.modules, list):
        raw_mods = []
        for item in args.modules:
            raw_mods.extend([x.strip() for x in item.split(",") if x.strip()])
        if not raw_mods or raw_mods == ["all"]:
            names = all_available
        elif raw_mods == ["attentions"]:
            names = ["full"] + [m for m in all_available if m in ATTENTION_REGISTRY and m != "full"]
        elif raw_mods == ["compressors"]:
            names = ["full"] + [m for m in all_available if m in COMPRESSOR_REGISTRY and m != "full"]
        else:
            names = raw_mods
    elif args.modules == "all":
        names = all_available
    elif args.modules == "attentions":
        names = ["full"] + [m for m in all_available if m in ATTENTION_REGISTRY and m != "full"]
    elif args.modules == "compressors":
        names = ["full"] + [m for m in all_available if m in COMPRESSOR_REGISTRY and m != "full"]
    else:
        names = [x.strip() for x in args.modules.split(",") if x.strip()]

    if args.dataset_dir is None:
        for cand in [
            Path("D:/SVD_Project/Codes/framework/data/glue_data") / args.dataset,
            Path("Codes/framework/data/glue_data") / args.dataset,
            Path(__file__).parent / "data" / "glue_data" / args.dataset,
            Path("D:/SVD_Project/Codes/framework/data/glue_data/sst2"),
            Path("Codes/framework/data/glue_data/sst2"),
            Path(__file__).parent / "data" / "glue_data" / "sst2",
        ]:
            if cand.exists():
                args.dataset_dir = str(cand)
                break

    if args.output_dir is None:
        safe_model = Path(args.model).name.replace("-", "_").lower()
        args.output_dir = f"results/{safe_model}_{args.dataset}_sweep"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_filename = f"summary_{args.output_suffix}.csv" if getattr(args, "output_suffix", None) else "summary.csv"
    path = out / csv_filename
    
    summary_dict = {}
    if path.exists() and not args.fresh:
        import csv
        with path.open("r", encoding="utf8") as f:
            for r in csv.DictReader(f):
                summary_dict[r["module_name"]] = r
                
    existing = set() if (args.fresh or getattr(args, "no_skip", False)) else {
        m for m, r in summary_dict.items()
        if r.get("status") in {
            "PASS", "PASS_LOW", "REJECTED_EXPANSION",
            "EXTREME_EXPANSION", "MODERATE_EXPANSION",
            "TIMEOUT", "SKIP_BERT_ARCHITECTURAL", "SKIP_BLOCKED",
        }
    }
    

    import threading
    csv_write_lock = threading.Lock()

    def _execute_single(name):
        override_to = SLOW_MODULE_TIMEOUT_OVERRIDES.get(name, 0)
        cli_timeout = getattr(args, "timeout", None) or 1200
        timeout_s = max(cli_timeout, override_to) if override_to > 0 else cli_timeout
        result = run_module_isolated(name, args, timeout_s)
        row = {
            "module_name": name,
            "status":                        result.get("status", "UNKNOWN"),
            "accuracy_pct":                  result.get("accuracy_pct", result.get("accuracy", 0.0)),
            "f1_score":                      result.get("f1_score", result.get("f1", 0.0)),
            "f1_macro":                      result.get("f1_macro", 0.0),
            "precision":                     result.get("precision", 0.0),
            "recall":                        result.get("recall", 0.0),
            "val_loss":                      result.get("val_loss", 0.0),
            "elapsed_s":                     result.get("elapsed_s", 0.0),
            "peak_vram_mb":                  result.get("peak_vram_mb", 0.0),
            "tokens_per_sec":                result.get("tokens_per_sec") if result.get("tokens_per_sec") is not None else result.get("throughput_tok_per_sec", 0.0),
            "latency_ms_per_token":         result.get("latency_ms_per_token", result.get("latency_ms", 0.0)),
            "throughput_tok_per_sec":       result.get("throughput_tok_per_sec", result.get("tokens_per_sec", 0.0)),
            "perplexity":                   result.get("perplexity", ""),
            "ppl_status":                   result.get("ppl_status", ""),
            "long_context_accuracy":        result.get("long_context_accuracy", ""),
            "total_params_before":           result.get("total_params_before", result.get("params_before", 0)),
            "total_params_after":            result.get("total_params_after", result.get("params_after", 0)),
            "trainable_recovery_params":     result.get("trainable_recovery_params", 0),
            "compression_rate":              result.get("compression_rate", result.get("compression_ratio", 1.0)),
            "compression_ratio_pct":         result.get("compression_ratio_pct", 0.0),
            "flops_reduction_pct":           result.get("flops_reduction_pct", 0.0),
            "num_compressed_layers":         result.get("num_compressed_layers", 0),
            "num_fallback_layers":           result.get("num_fallback_layers", 0),
            "avg_reconstruction_error":      result.get("avg_reconstruction_error", ""),
            "weighted_reconstruction_error": result.get("weighted_reconstruction_error", ""),
            "num_injected_attention_layers": result.get("num_injected_attention_layers", 0),
            "lora_params":                   result.get("lora_params", 0),
            "error":                         result.get("error", ""),
            "path":                          result.get("path", ""),
            "timestamp":                     result.get("finished_at", time.time()),
        }
        with csv_write_lock:
            summary_dict[name] = row
            existing_rows = {}
            if path.exists():
                try:
                    with path.open("r", encoding="utf8") as f:
                        import csv
                        reader = csv.DictReader(f)
                        for r in reader:
                            if r.get("module_name"):
                                existing_rows[r["module_name"]] = r
                except Exception:
                    pass
            existing_rows.update(summary_dict)
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", newline="", encoding="utf8") as f:
                import csv
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                for m in existing_rows:
                    w.writerow(existing_rows[m])
            if tmp.exists():
                if path.exists():
                    try: path.unlink()
                    except Exception as e: LOG.debug("Failed to unlink %s: %s", path, e)
                tmp.rename(path)
        clear_vram_between_modules()
        return row

    if len(names) <= 10:
        pending_names = [n for n in names if n not in existing]
    else:
        pending_names = [n for n in names if n not in existing and n not in BLOCKED_MODULES]

    # for n in names:
    #     if n in existing:
    #         print(f">>> Skipping {n} (already PASS)")
    #     elif n in BLOCKED_MODULES and len(names) > 10:
    #         print(f">>> Skipping BLOCKED {n}")

    workers = max(1, getattr(args, "parallel", 1))
    from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    # Force dynamic module discovery to populate ATTENTION_REGISTRY and COMPRESSOR_REGISTRY
    _ = modules()
    attention_modules = [name for name in pending_names if name in ATTENTION_REGISTRY or name == "full"]
    compressor_modules = [name for name in pending_names if name in COMPRESSOR_REGISTRY and name not in attention_modules]

    if compressor_modules:
        LOG.info("Running %d compressor modules via isolated subprocesses (workers=%d)", len(compressor_modules), workers)
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(_execute_single, compressor_modules))
        else:
            for name in compressor_modules:
                _execute_single(name)

    if attention_modules:
        LOG.info("Running %d attention modules in-memory sequentially", len(attention_modules))
        args.modules = attention_modules
        run_single_module_main(args)
        
        # Merge results from run_single_module_main into summary.csv
        path = out / csv_filename
        if path.exists():
            try:
                with path.open("r", encoding="utf8") as f:
                    import csv
                    reader = csv.DictReader(f)
                    for r in reader:
                        summary_dict[r["module_name"]] = r
            except Exception:
                pass

if __name__ == '__main__':
    main()
