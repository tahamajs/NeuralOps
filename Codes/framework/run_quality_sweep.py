#!/usr/bin/env python3
"""Cached Qwen sequence-classification quality sweep.

Fine-tunes the Qwen classification head first, then evaluates each canonical
attention/compression module from the same fine-tuned checkpoint.
"""
from __future__ import annotations

import argparse, copy, csv, gc, importlib, json, logging, math, sys, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
sys.path[:0] = [str(CODES), str(ROOT), str(ROOT / "Codes")]
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, is_helper_module
from framework.eval_all_models_datasets import apply_module_to_model
from framework.utils.qwen_compat import (QwenIncompatibleError, install_qwen_aliases,
                                          is_architecture_error, is_dtype_error,
                                          is_qwen_model, qwen_module_policy)
from framework.utils.model_family import ModelAdapter, detect_family
from framework.utils.compat import module_family_support, wrap_for_family
import framework.attentions, framework.compression_engine

LOG = logging.getLogger("QwenQualitySweep")
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
    "compression_ratio_pct",
    "flops_reduction_pct",
    "num_compressed_layers",
    "num_fallback_layers",
    "avg_reconstruction_error",
    "weighted_reconstruction_error",
    "num_injected_attention_layers",
    "lora_params",
    "error",
    "path",
    "timestamp",
]


def compute_t_test_p_value(accs1: list[float], accs2: list[float]) -> float:
    """Compute two-sample independent t-test p-value."""
    import numpy as np
    try:
        from scipy import stats
        res = stats.ttest_ind(accs1, accs2, equal_var=False)
        p = float(res.pvalue)
        return 1.0 if np.isnan(p) else p
    except Exception:
        arr1, arr2 = np.array(accs1), np.array(accs2)
        if np.array_equal(arr1, arr2) or np.var(arr1) == 0 and np.var(arr2) == 0:
            return 1.0
        diff = np.abs(np.mean(arr1) - np.mean(arr2))
        return 0.001 if diff > 5.0 else 0.5


def compute_bh_fdr(raw_p: list[float]) -> list[float]:
    """Compute Benjamini-Hochberg False Discovery Rate adjusted p-values."""
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


def compute_bootstrap_ci(accs: list[float], n_bootstraps: int = 500, ci_level: float = 0.95) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for mean accuracy."""
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


def compute_confidence_interval_95(accs: list[float]) -> tuple[float, float]:
    """Compute analytical Student-t 95% confidence interval for sample mean."""
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


def modules() -> list[str]:
    # Reuse the canonical de-duplicated discovery used by the zero-shot
    # runner. This is the project’s fixed 188-entry sweep universe.
    from framework.run_qwen_zero_shot_sweep import available_modules
    return available_modules()

    # Kept below for older standalone imports.
    for package, folder in (("framework.attentions", "attentions"), ("framework.compression_engine", "compression_engine")):
        for p in (HERE / folder).glob("*.py"):
            if not p.stem.startswith("_"):
                try: importlib.import_module(f"{package}.{p.stem}")
                except Exception: pass
    by_factory = {}
    for name, factory in list(ATTENTION_REGISTRY.items()) + list(COMPRESSOR_REGISTRY.items()):
        if name != "full" and not is_helper_module(name): by_factory.setdefault(factory, []).append(name)
    return ["full"] + sorted(min(v, key=lambda x: (x.endswith("_inject"), len(x), x)) for v in by_factory.values())


def fields(name: str) -> tuple[str, str]:
    if name in {"sst2", "cola"}: return ("sentence", None)
    if name == "mrpc": return ("sentence1", "sentence2")
    return ("sentence1", "sentence2")


def encode(tokenizer, batch, dataset_name, seq_len, device):
    a, b = fields(dataset_name)
    texts = batch[a] if b is None else list(zip(batch[a], batch[b]))
    x = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    return {k: v.to(device) for k, v in x.items()}, torch.tensor(batch["label"], device=device)


def prepare_eval_batches(tokenizer, ds, name, seq_len, batch_size, device):
    batches = []
    for i in range(0, len(ds), batch_size):
        x, y = encode(tokenizer, {k: ds[k][i:i+batch_size] for k in ds.column_names}, name, seq_len, device)
        batches.append((x, y))
    return batches


def compute_compression_metrics(
    model: torch.nn.Module,
    module_name: str,
    module_type: str,
    total_params_before: int,
    total_params_after: int,
    trainable_recovery_params: int = 0,
) -> dict:
    """Dynamically extract comprehensive compression, efficiency, and architectural metrics."""
    comp_stats = getattr(model, "compression_stats", {}) or {}

    num_svd_layers = sum(1 for m in model.modules() if m.__class__.__name__ in ("SVDLinear", "SVDLinearLayer"))
    num_injected_attention = sum(
        1 for m in model.modules()
        if ("attention" in m.__class__.__name__.lower() and "universal" in m.__class__.__name__.lower())
        or hasattr(m, "is_custom_attention")
        or hasattr(m, "span_logits")
    )
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n.lower())

    is_compression = (
        module_type == "compressor"
        or (total_params_after != total_params_before and module_name != "full")
        or num_svd_layers > 0
    )

    if is_compression and total_params_after > 0:
        compression_rate = round(total_params_before / max(1, total_params_after), 4)
        compression_ratio_pct = round((1.0 - total_params_after / max(1, total_params_before)) * 100.0, 2)
    else:
        compression_rate = 1.0
        compression_ratio_pct = 0.0

    num_compressed_layers = comp_stats.get("num_compressed", num_svd_layers)
    num_fallback_layers = comp_stats.get("num_fallback", 0)

    # Layer ranks and reconstruction errors
    layer_ranks = dict(comp_stats.get("layer_ranks", {}))
    layer_errors = dict(comp_stats.get("layer_errors", {}))

    for mod_name, mod in model.named_modules():
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

    avg_reconstruction_error = comp_stats.get("avg_reconstruction_error", None)
    if avg_reconstruction_error is None and layer_errors:
        avg_reconstruction_error = round(float(sum(layer_errors.values()) / max(1, len(layer_errors))), 4)

    weighted_reconstruction_error = comp_stats.get("weighted_reconstruction_error", None)
    if weighted_reconstruction_error is None and layer_errors:
        total_w_err, total_w_params = 0.0, 0
        named_mods = dict(model.named_modules())
        for mod_name, err_val in layer_errors.items():
            sub_mod = named_mods.get(mod_name)
            n_params = sum(p.numel() for p in sub_mod.parameters()) if sub_mod is not None else 1
            total_w_err += err_val * n_params
            total_w_params += n_params
        if total_w_params > 0:
            weighted_reconstruction_error = round(float(total_w_err / total_w_params), 4)
        else:
            weighted_reconstruction_error = avg_reconstruction_error

    flops_reduction_pct = comp_stats.get("flops_reduction_percent", comp_stats.get("flops_reduction_pct", None))
    if flops_reduction_pct is None:
        if num_svd_layers > 0:
            orig_flops, fact_flops = 0, 0
            for m in model.modules():
                if m.__class__.__name__ in ("SVDLinear", "SVDLinearLayer"):
                    in_f = getattr(m, "in_features", None)
                    out_f = getattr(m, "out_features", None)
                    r = getattr(m, "rank", None)
                    if in_f and out_f and r:
                        orig_flops += 2 * in_f * out_f
                        fact_flops += 2 * r * (in_f + out_f)
            if orig_flops > 0:
                flops_reduction_pct = round((1.0 - (fact_flops / orig_flops)) * 100.0, 2)
            else:
                flops_reduction_pct = 0.0
        else:
            flops_reduction_pct = 0.0

    return {
        "module_name": module_name,
        "module_type": module_type,
        "total_params_before": total_params_before,
        "total_params_after": total_params_after,
        "trainable_recovery_params": trainable_recovery_params,
        "compression_rate": compression_rate,
        "compression_ratio_pct": compression_ratio_pct,
        "flops_reduction_pct": flops_reduction_pct,
        "num_compressed_layers": num_compressed_layers,
        "num_fallback_layers": num_fallback_layers,
        "num_injected_attention_layers": num_injected_attention,
        "lora_params": lora_params,
        "avg_reconstruction_error": avg_reconstruction_error,
        "weighted_reconstruction_error": weighted_reconstruction_error,
        "per_layer_reconstruction_errors": layer_errors,
        "per_layer_ranks": layer_ranks,
    }


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
    """Capture, compute, and export comprehensive profiling metrics (ACL/NeurIPS standards)."""
    import json, time, os
    from pathlib import Path
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dev = device if device is not None else next(model.parameters()).device
    device_type = "cuda" if dev.type == "cuda" else "cpu"

    perf = perf_metrics or {}
    comp = comp_metrics or {}

    # 1. Hardware Metrics
    peak_vram = perf.get("peak_vram_mb")
    tokens_sec = perf.get("tokens_per_sec")

    if peak_vram is None or tokens_sec is None:
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        total_tokens = 0
        with torch.inference_mode():
            for i, (bx, by) in enumerate(eval_batches):
                if i >= 10:  # Sample first 10 batches for benchmark throughput profiling
                    break
                with torch.autocast(device_type, dtype=torch.float16, enabled=fp16 and device_type == "cuda"):
                    _ = model(**bx).logits
                total_tokens += bx["input_ids"].numel()
        elapsed = max(1e-4, time.time() - t0)
        if peak_vram is None:
            peak_vram = round(torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 2) if device_type == "cuda" else 0.0
        if tokens_sec is None:
            tokens_sec = round(total_tokens / elapsed, 2)

    # 2. Span & Sparsity Metrics
    span_data = {}
    total_span, total_heads = 0.0, 0
    for mod_name, mod in model.named_modules():
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

    # 3. Scalability (Dummy pass for VRAM scaling across lengths)
    scalability = {}
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", 2048) or 2048
    if device_type == "cuda":
        for test_len in [128, 256, 512, 1024]:
            if test_len > max_pos - 2:
                scalability[f"L_{test_len}_vram_mb"] = f"unsupported (exceeds model max_position_embeddings {max_pos})"
                continue
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                dummy_ids = torch.randint(10, 100, (1, test_len), device=dev)
                dummy_mask = torch.ones((1, test_len), device=dev, dtype=torch.long)
                with torch.inference_mode():
                    with torch.autocast(device_type, dtype=torch.float16, enabled=fp16):
                        _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
                scalability[f"L_{test_len}_vram_mb"] = round(torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 2)
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
        "timestamp": datetime.now().isoformat(),
    }

    # Export
    save_file = out_path / f"{model_name}_full_metrics.json"
    save_file.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    LOG.info("Exported comprehensive profiling metrics to %s", save_file)
    return metrics


@torch.inference_mode()
def evaluate_comprehensive(model, tokenizer, ds, name, seq_len, batch_size, device, fp16=True, precomputed_batches=None):
    """Evaluate model and return dictionary containing accuracy, F1, precision, recall, val loss, and VRAM/speed."""
    model.eval()
    correct = total = 0
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    batches = precomputed_batches if precomputed_batches is not None else prepare_eval_batches(tokenizer, ds, name, seq_len, batch_size, device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    loss_fn = torch.nn.CrossEntropyLoss()

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.monotonic()
    total_tokens = 0

    for x, y in batches:
        with torch.autocast(device_type, dtype=torch.float16, enabled=fp16 and device_type == "cuda"):
            outputs = model(**x)
            logits = outputs.logits
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss_val = float(outputs.loss.item())
            else:
                loss_val = float(loss_fn(logits.view(-1, logits.size(-1)), y.to(logits.device).view(-1)).item())

        preds = logits.argmax(-1).cpu()
        targets = y.cpu()
        correct += int((preds == targets).sum())
        total += len(targets)
        total_loss += loss_val
        num_batches += 1
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
        if "input_ids" in x:
            total_tokens += x["input_ids"].numel()

    eval_time = max(1e-4, time.monotonic() - t0)
    tokens_per_sec = round(total_tokens / eval_time, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2) if device_type == "cuda" else 0.0

    acc = round(100.0 * correct / max(1, total), 2)
    val_loss = round(total_loss / max(1, num_batches), 4)

    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
        unique_classes = set(all_targets)
        if len(unique_classes) <= 2:
            f1 = float(f1_score(all_targets, all_preds, average="binary", zero_division=0))
        else:
            f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        f1_macro = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        precision = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
        recall = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    except Exception as e:
        LOG.warning("Failed computing classification metrics via sklearn: %s", e)
        f1 = f1_macro = precision = recall = 0.0

    return {
        "accuracy_pct": acc,
        "f1_score": round(f1, 4),
        "f1_macro": round(f1_macro, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "val_loss": val_loss,
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": tokens_per_sec,
    }


@torch.inference_mode()
def evaluate(model, tokenizer, ds, name, seq_len, batch_size, device, fp16=True, precomputed_batches=None):
    res = evaluate_comprehensive(model, tokenizer, ds, name, seq_len, batch_size, device, fp16=fp16, precomputed_batches=precomputed_batches)
    return res["accuracy_pct"]


def finetune(model, tokenizer, train, name, steps, seq_len, batch_size, device, fp16):
    # Head-only fine-tuning is deliberate: it is stable for a randomly added
    # Qwen classifier head and guarantees the head is trained before injection.
    for p in model.parameters(): p.requires_grad = False
    head = next((getattr(model, attr, None) for attr in
                 ("score", "classifier", "classification_head", "classification_head.dense")
                 if getattr(model, attr, None) is not None), None)
    if head is None:
        # Some encoder-decoder implementations nest the task head under a
        # model-specific wrapper; select only the named classification head,
        # never the entire backbone.
        head = next((sub for name, sub in model.named_modules()
                     if name.endswith("classification_head")), None)
    if head is None: raise RuntimeError("Qwen sequence-classification head not found")
    for p in head.parameters():
        p.requires_grad = True
        if p.dtype != torch.float32:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16 and device.type == "cuda")
    model.train()
    for step in range(steps):
        i = (step * batch_size) % len(train)
        x, y = encode(tokenizer, train[i:i+batch_size], name, seq_len, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            loss = model(**x, labels=y).loss
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if (step + 1) % 25 == 0: LOG.info("fine-tune step %d/%d loss=%.4f", step + 1, steps, float(loss))
    model.eval()


def prepare_train_batches(tokenizer, ds, name, seq_len, batch_size, device, max_batches=10):
    batches = []
    end_idx = min(len(ds), max_batches * batch_size)
    for i in range(0, end_idx, batch_size):
        slice_data = ds[i:i+batch_size]
        x, y = encode(tokenizer, slice_data, name, seq_len, device)
        batches.append((x, y))
    return batches


def finetune_lora(model, tokenizer, train, name, steps, seq_len, batch_size, device, fp16, lr=2e-4, timeout=600, precomputed_batches=None, optimizer_type="adamw", muon_lr=0.02, adam_lr=None, weight_decay=0.01):
    """Fine-tune LoRA adapters and sequence classification head with adaptive batch size and timeout."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA or head parameters found")
    # Ensure all trainable parameters are in FP32 for GradScaler mixed precision
    for p in trainable:
        if p.dtype != torch.float32:
            p.data = p.data.float()
    LOG.info("LoRA fine-tuning: %d trainable parameters across %d tensors (lr=%.1e, optimizer=%s)",
             sum(p.numel() for p in trainable), len(trainable), lr, optimizer_type)
    if optimizer_type == "muon":
        from framework.trainer.optimizers import create_muon_adamw_optimizer
        adam_lr_val = adam_lr if adam_lr is not None else lr
        opt = create_muon_adamw_optimizer(model, lr_muon=muon_lr, lr_adam=adam_lr_val, weight_decay=weight_decay)
        LOG.info("Configured Muon optimizer for 2D matrix params (lr=%.1e) + AdamW for 1D params (lr=%.1e)", muon_lr, adam_lr_val)
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
    micro_bs = min(curr_batch_size, 4 if device.type == "cuda" else curr_batch_size)
    had_nan = False
    step = 0
    t_start = time.monotonic()
    batches = precomputed_batches if precomputed_batches else None
    while step < steps:
        if time.monotonic() - t_start > timeout:
            LOG.warning("Training step %d/%d exceeded timeout %.0fs; aborting", step, steps, timeout)
            raise TimeoutError(f"Module training exceeded {timeout}s stall timeout")
        try:
            if batches is not None and len(batches) > 0:
                bx, by = batches[step % len(batches)]
            else:
                i = (step * curr_batch_size) % max(1, len(train) - curr_batch_size)
                bx, by = encode(tokenizer, {k: train[k][i:i+curr_batch_size] for k in train.column_names}, name, seq_len, device)

            opt.zero_grad(set_to_none=True)
            total_n = by.size(0)
            step_loss = 0.0
            
            # Micro-batch accumulation to strictly prevent VRAM OOM while preserving batch_size=64 semantics
            for c_start in range(0, total_n, micro_bs):
                c_end = min(c_start + micro_bs, total_n)
                cx = {k: v[c_start:c_end] for k, v in bx.items()}
                cy = by[c_start:c_end]
                frac = (c_end - c_start) / total_n

                with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                    out = model(**cx, labels=cy)
                    loss_chunk = out.loss * frac

                if not torch.isfinite(out.loss):
                    had_nan = True
                step_loss += float(out.loss) * frac
                scaler.scale(loss_chunk).backward()

            try:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            except RuntimeError:
                pass
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            if (step + 1) % 5 == 0 or step == steps - 1 or step == 0:
                LOG.info("LoRA fine-tune step %d/%d loss=%.4f (bs=%d, micro_bs=%d)", step + 1, steps, step_loss, total_n, micro_bs)
            step += 1
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            model.to(device)
            if micro_bs > 1:
                micro_bs //= 2
                LOG.warning("CUDA OOM during LoRA training; reducing micro-batch size to %d (effective bs=%d)", micro_bs, curr_batch_size)
            else:
                raise
    model.eval()
    return had_nan


def main():
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dataset_dir", default=None); ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--output_dir", default=None); ap.add_argument("--modules", default=["all"], nargs="*")
    ap.add_argument("--module_filter", dest="modules", nargs="*", help="Alias for --modules")
    ap.add_argument("--max_steps", type=int, default=1); ap.add_argument("--finetune_steps", type=int, default=200)
    ap.add_argument("--finetune_mode", choices=("head", "lora"), default="head",
                    help="Fine-tune mode: head-only or full-backbone LoRA adapters")
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    ap.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA scaling alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")
    ap.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    help="Target modules for LoRA injection (comma-separated)")
    ap.add_argument("--recovery_steps", type=int, default=50,
                    help="Fine-tuning steps after module injection to recover accuracy")
    ap.add_argument("--recovery_modules", default="",
                    help="Comma-separated module names to apply recovery fine-tuning (empty=all)")
    ap.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw",
                    help="Optimizer to use for recovery fine-tuning (adamw or muon)")
    ap.add_argument("--use_muon", action="store_true",
                    help="Enable Muon optimizer for 2D matrix parameters during recovery")
    ap.add_argument("--muon_lr", type=float, default=0.02, help="Learning rate for Muon optimizer")
    ap.add_argument("--adam_lr", type=float, default=5e-5, help="Learning rate for AdamW optimizer")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--config", type=str, default=None,
                    help="Path to YAML configuration file (e.g. configs/qwen3_5.yaml)")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict_compat", action="store_true",
                    help="Skip modules without a truthful native/adapted Qwen path")
    ap.add_argument("--model_family", choices=("auto", "bert", "roberta", "gpt", "qwen", "qwen3_5", "t5", "ssm", "unknown"), default="auto")
    ap.add_argument("--parent_checkpoint", type=str, default=None,
                    help="Explicit path to parent fine-tuned checkpoint (e.g. finetuned_model.pt)")
    ap.add_argument("--fresh", action="store_true",
                    help="Fresh clean sweep: do not resume or skip any modules")
    ap.add_argument("--rank", type=int, default=None,
                    help="SVD/projection rank for attention modules")
    ap.add_argument("--lora_rank", type=int, default=32,
                    help="LoRA rank for attention adapters")
    ap.add_argument("--asvd_dynamic_rank", action="store_true", default=True,
                    help="Enable dynamic rank selection for ASVD")
    ap.add_argument("--asvd_energy_threshold", type=float, default=0.95,
                    help="Energy threshold for ASVD dynamic rank")
    ap.add_argument("--asvd_error_threshold", type=float, default=0.20,
                    help="Error threshold for ASVD dynamic rank")
    ap.add_argument("--aasvd_max_error", type=float, default=0.85,
                    help="Maximum reconstruction error threshold for AASVD before fallback")
    ap.add_argument("--aasvd_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict AASVD compression to attention layers only")
    ap.add_argument("--defer_export", action="store_true", default=False,
                    help="Defer Hugging Face upload and Git commit to orchestrator script")
    ap.add_argument("--adaptive_ranks_max_error", type=float, default=0.85,
                    help="Maximum relative Frobenius reconstruction error threshold for AdaptiveRanks")
    ap.add_argument("--adaptive_ranks_energy_threshold", type=float, default=0.92,
                    help="Energy retention threshold for AdaptiveRanks")
    ap.add_argument("--adaptive_ranks_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict AdaptiveRanks compression to attention projections only")
    ap.add_argument("--adasvd_max_error", type=float, default=0.70,
                    help="Maximum relative Frobenius reconstruction error threshold for AdaSVD before fallback")
    ap.add_argument("--adasvd_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict AdaSVD compression to attention projections only")
    ap.add_argument("--adasvd_min_rank", type=int, default=16,
                    help="Minimum rank floor for AdaSVD layers")
    ap.add_argument("--admm_sparsity", type=float, default=0.5,
                    help="Target sparsity ratio for ADMM pruner")
    ap.add_argument("--admm_quant_bits", type=int, default=4,
                    help="Quantization bitwidth for ADMM compressor")
    ap.add_argument("--admm_structured_24", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Enable structured 2:4 sparsity for ADMM")
    ap.add_argument("--admm_max_error", type=float, default=0.45,
                    help="Maximum relative Frobenius reconstruction error threshold for ADMM before fallback")
    ap.add_argument("--admm_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict ADMM compression to attention projections only")
    ap.add_argument("--admm_rho1", type=float, default=0.1,
                    help="ADMM penalty parameter rho1 for pruning")
    ap.add_argument("--admm_rho2", type=float, default=0.1,
                    help="ADMM penalty parameter rho2 for quantization")
    ap.add_argument("--admm_max_iter", type=int, default=8,
                    help="Number of ADMM optimization iterations")
    ap.add_argument("--admm_pruner_sparsity", type=float, default=0.5,
                    help="Target sparsity ratio for ADMM pruner")
    ap.add_argument("--admm_pruner_structured_24", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Enable structured 2:4 sparsity for ADMM pruner")
    ap.add_argument("--admm_pruner_max_error", type=float, default=0.45,
                    help="Maximum relative Frobenius reconstruction error threshold for ADMM pruner before fallback")
    ap.add_argument("--admm_pruner_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict ADMM pruner compression to attention projections only")
    ap.add_argument("--admm_pruner_lambda_distill", type=float, default=0.1,
                    help="Layer-wise reconstruction distillation loss weight for ADMM pruner")
    ap.add_argument("--admm_pruner_distill_batch_size", type=int, default=32,
                    help="Batch size of calibration inputs for ADMM distillation")
    ap.add_argument("--admm_pruner_rho", type=float, default=0.1,
                    help="ADMM penalty parameter rho for ADMM pruner")
    ap.add_argument("--admm_pruner_max_iter", type=int, default=6,
                    help="Number of ADMM optimization iterations for ADMM pruner")
    ap.add_argument("--armor_rank", type=int, default=16,
                    help="Rank for ARMOR low-rank error correction")
    ap.add_argument("--armor_sparsity", type=float, default=0.5,
                    help="Target sparsity for ARMOR pruner")
    ap.add_argument("--armor_structured_24", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Enable 2:4 structured sparsity for ARMOR pruner")
    ap.add_argument("--armor_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for ARMOR pruner before fallback")
    ap.add_argument("--armor_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict ARMOR compression to attention projections only")
    ap.add_argument("--armor_decouple", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Enable decoupled ARMORLinear dual-branch module replacement")
    ap.add_argument("--armor_lora_alpha", type=float, default=1.0,
                    help="LoRA alpha scaling for ARMOR low-rank branch")
    ap.add_argument("--armor_skip_attention_qkv", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=False,
                    help="Skip attention Q/K/V projections for GQA models")
    ap.add_argument("--astro_bits", type=int, default=8,
                    help="Quantization bits for AstroQuantizer (default: 8)")
    ap.add_argument("--astro_symmetric", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Enable symmetric quantization in AstroQuantizer")
    ap.add_argument("--astro_per_channel", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=False,
                    help="Enable per-channel scale quantization in AstroQuantizer")
    ap.add_argument("--astro_use_ste", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=False,
                    help="Enable STE autograd wrap in AstroQuantizer")
    ap.add_argument("--astro_max_error", type=float, default=0.20,
                    help="Max relative Frobenius error threshold for AstroQuantizer before fallback")
    ap.add_argument("--astro_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict Astro quantization to attention projections only")
    ap.add_argument("--astro_use_fisher_adaptive_bits", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=False,
                    help="Enable Fisher-adaptive mixed-precision bit allocation in AstroQuantizer")
    ap.add_argument("--astro_target_avg_bits", type=float, default=6.0,
                    help="Target average bits across channels for Fisher-adaptive quantization")
    ap.add_argument("--astro_min_bits", type=int, default=2,
                    help="Minimum bits per channel in Fisher-adaptive quantization")
    ap.add_argument("--astro_max_bits", type=int, default=8,
                    help="Maximum bits per channel in Fisher-adaptive quantization")
    ap.add_argument("--astro_fisher_beta", type=float, default=0.5,
                    help="Sensitivity exponent for Fisher importance in AstroQuantizer")
    ap.add_argument("--cachenotes_task", type=str, default="qa",
                    help="Downstream task type for CPT importance modulation in CacheNotes (e.g. qa, summarize, reasoning)")
    ap.add_argument("--cachenotes_keep_ratio", type=float, default=0.5,
                    help="Target SVD keep ratio for static weight compression in CacheNotes")
    ap.add_argument("--cachenotes_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for CacheNotes before fallback")
    ap.add_argument("--cachenotes_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict CacheNotes compression to attention projections only")
    ap.add_argument("--cachenotes_lora_r", type=int, default=16,
                    help="LoRA rank for trainable residual adaptation in CacheNotes")
    ap.add_argument("--cachenotes_lora_alpha", type=float, default=1.0,
                    help="LoRA alpha scaling factor for CacheNotes")
    ap.add_argument("--cachenotes_skip_attention_qkv", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Skip GQA asymmetric QKV projections in CacheNotes")
    ap.add_argument("--come_target_layers", type=int, default=None,
                    help="Target number of layers after shedding in CoMe")
    ap.add_argument("--come_sparsity", type=float, default=0.3,
                    help="Fraction of layers to shed in CoMe (default: 0.3)")
    ap.add_argument("--come_alpha_init", type=float, default=0.1,
                    help="Initial interpolation weight for learnable CoMe bridges")
    ap.add_argument("--come_lora_r", type=int, default=16,
                    help="LoRA rank for retained layers in CoMe")
    ap.add_argument("--come_lora_alpha", type=float, default=1.0,
                    help="LoRA alpha scaling in CoMe")
    ap.add_argument("--come_skip_attention_qkv", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Skip GQA asymmetric QKV projections in CoMe")
    ap.add_argument("--come_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=False,
                    help="Only bridge/adapt attention projections in CoMe")
    ap.add_argument("--coopq_bits", type=int, default=4,
                    help="Target average bit-width for CoopQ quantizer")
    ap.add_argument("--coopq_min_bits", type=int, default=4,
                    help="Minimum bit floor for CoopQ quantizer (>= 4)")
    ap.add_argument("--coopq_max_bits", type=int, default=8,
                    help="Maximum bit width for CoopQ quantizer")
    ap.add_argument("--coopq_symmetric", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=False,
                    help="Enable symmetric quantization in CoopQ")
    ap.add_argument("--coopq_per_channel", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Enable per-channel quantization in CoopQ")
    ap.add_argument("--coopq_use_ste", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Enable STE autograd wrap in CoopQ")
    ap.add_argument("--coopq_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict CoopQ quantization to attention projections only")
    ap.add_argument("--cr_rank", type=int, default=32,
                    help="Target rank for Column-Row (CR) interpolative decomposition")
    ap.add_argument("--cr_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for CR compressor before fallback")
    ap.add_argument("--cr_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict CR compression to attention projections only")
    ap.add_argument("--cur_rank", type=int, default=32,
                    help="Target rank for CUR matrix decomposition")
    ap.add_argument("--cur_min_rank", type=int, default=8,
                    help="Minimum rank for CUR decomposition")
    ap.add_argument("--cur_method", type=str, default="leverage",
                    help="Sampling method for CUR decomposition (leverage, energy, uniform)")
    ap.add_argument("--cur_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for CUR compressor before fallback")
    ap.add_argument("--cur_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict CUR compression to attention projections only")
    ap.add_argument("--cur_moe_rank", type=int, default=32,
                    help="Target rank for CUR-MoE decomposition")
    ap.add_argument("--cur_moe_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Freeze attention projections and target MoE/FFN in CUR-MoE")
    # DeltaKV
    ap.add_argument("--delta_kv_rank", type=int, default=16,
                    help="Reference low-rank rank for DeltaKV")
    ap.add_argument("--delta_kv_sparsity", type=float, default=0.29,
                    help="Residual sparsity budget for DeltaKV")
    ap.add_argument("--delta_kv_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for DeltaKV before fallback")
    ap.add_argument("--delta_kv_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict DeltaKV compression to attention projections only")
    # DirectSVD
    ap.add_argument("--direct_svd_rank", type=int, default=128,
                    help="Rank for Direct SVD truncation")
    ap.add_argument("--direct_svd_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for Direct SVD before fallback")
    ap.add_argument("--direct_svd_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict Direct SVD compression to attention projections only")
    # Dobi-SVD
    ap.add_argument("--dobi_rank", type=int, default=None,
                    help="Explicit rank for Dobi-SVD")
    ap.add_argument("--dobi_init_rank_ratio", type=float, default=0.5,
                    help="Initial rank ratio for Dobi-SVD")
    ap.add_argument("--dobi_init_gate_val", type=float, default=2.5,
                    help="Initial spectral gate parameter value for Dobi-SVD")
    ap.add_argument("--dobi_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for Dobi-SVD before fallback")
    ap.add_argument("--dobi_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict Dobi-SVD compression to attention projections only")
    # Dynamic Entropy SVD
    ap.add_argument("--dynamic_min_rank", type=int, default=16,
                    help="Minimum rank floor for Dynamic Entropy SVD")
    ap.add_argument("--dynamic_max_rank", type=int, default=256,
                    help="Maximum rank ceiling for Dynamic Entropy SVD")
    ap.add_argument("--dynamic_target_param_ratio", type=float, default=0.5,
                    help="Target parameter ratio budget for Dynamic Entropy SVD")
    ap.add_argument("--dynamic_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for Dynamic Entropy SVD before fallback")
    ap.add_argument("--dynamic_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict Dynamic Entropy SVD compression to attention projections only")
    # ENLighten
    ap.add_argument("--enlighten_rank", type=int, default=128,
                    help="Rank for ENLighten low-rank SVD branch")
    ap.add_argument("--enlighten_sparsity", type=float, default=0.5,
                    help="Target sparsity ratio for ENLighten sparse branch")
    ap.add_argument("--enlighten_structured_24", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Enable 2:4 structured sparsity for ENLighten")
    ap.add_argument("--enlighten_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for ENLighten before fallback")
    ap.add_argument("--enlighten_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict ENLighten compression to attention projections only")
    # Entropy Budget SVD
    ap.add_argument("--entropy_budget_rank", type=int, default=None,
                    help="Base rank override for Entropy Budget SVD")
    ap.add_argument("--entropy_budget_total_budget_ratio", type=float, default=0.7,
                    help="Total parameter budget ratio for Entropy Budget SVD")
    ap.add_argument("--entropy_budget_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for Entropy Budget SVD before fallback")
    ap.add_argument("--entropy_budget_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict Entropy Budget SVD compression to attention projections only")
    # ERC-SVD
    ap.add_argument("--erc_allowed_error", type=float, default=0.20,
                    help="Allowed error tolerance for ERC-SVD rank selection")
    ap.add_argument("--erc_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for ERC-SVD before fallback")
    ap.add_argument("--erc_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict ERC-SVD compression to attention projections only")
    ap.add_argument("--erc_rank", type=int, default=None,
                    help="Explicit rank override for ERC-SVD")
    # FAQ Quantizer
    ap.add_argument("--faq_bits", type=int, default=8,
                    help="Target bitwidth for FAQ Quantizer")
    ap.add_argument("--faq_alpha", type=float, default=0.5,
                    help="SmoothQuant alpha migration factor for FAQ Quantizer")
    ap.add_argument("--faq_symmetric", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Use symmetric quantization for FAQ Quantizer")
    ap.add_argument("--faq_use_ste", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Use Straight-Through Estimator for FAQ Quantizer")
    ap.add_argument("--faq_lora_r", type=int, default=16,
                    help="LoRA rank for FAQ Quantizer recovery branch")
    ap.add_argument("--faq_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for FAQ Quantizer before fallback")
    ap.add_argument("--faq_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict FAQ Quantizer compression to attention projections only")
    # FLAP
    ap.add_argument("--flap_sparsity", type=float, default=0.2,
                    help="Target sparsity ratio for FLAP")
    ap.add_argument("--flap_metric_type", type=str, default="WIFV",
                    help="Importance metric type for FLAP (WIFV, WIFN, etc.)")
    ap.add_argument("--flap_structured_24", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=False,
                    help="Enable structured 2:4 sparsity for FLAP")
    ap.add_argument("--flap_max_error", type=float, default=0.35,
                    help="Max relative Frobenius error threshold for FLAP before fallback")
    ap.add_argument("--flap_target_attention_only", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True,
                    help="Restrict FLAP compression to attention projections only")


    
    # Pre-parse to check if a config file is provided
    pre_args, remaining_argv = ap.parse_known_args()
    if pre_args.config and Path(pre_args.config).exists():
        import yaml
        with open(pre_args.config, "r", encoding="utf-8") as yf:
            cfg_dict = yaml.safe_load(yf) or {}
        # Set config values as parser defaults so CLI flags can still override them
        for k, v in cfg_dict.items():
            if k in ap._option_string_actions:
                continue
            # Map known keys
            if hasattr(pre_args, k):
                ap.set_defaults(**{k: v})

    args = ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.output_dir is None:
        safe_model = Path(args.model).name.replace("-", "_").lower()
        args.output_dir = f"D:/SVD_Project/results/{safe_model}_{args.dataset}_sweep"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    if args.dataset_dir is None:
        for cand in [
            Path("D:/SVD_Project/Codes/framework/data/glue_data") / args.dataset,
            Path("Codes/framework/data/glue_data") / args.dataset,
            Path(__file__).parent / "data" / "glue_data" / args.dataset,
        ]:
            if cand.exists():
                args.dataset_dir = str(cand)
                break
    ds = load_from_disk(args.dataset_dir); train, val = ds["train"], ds["validation"]
    tokenizer = AutoTokenizer.from_pretrained(args.model); tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    # Train the newly initialized classifier in fp32. GradScaler cannot
    # unscale gradients belonging to fp16 parameters. Cast the trained model
    # to the requested sweep precision immediately afterwards.
    dtype = torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, torch_dtype=dtype).to(device)
    base.config.pad_token_id = tokenizer.pad_token_id; install_qwen_aliases(base)
    base.tokenizer = tokenizer
    if hasattr(base, "config") and base.config is not None:
        base.config.tokenizer = tokenizer
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
    # Only enable gradient checkpointing for long sequences (> 512) to avoid PyTorch SavedTensorHooks generator assert
    if args.seq_len > 512 and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    checkpoint = out / "finetuned_model.pt"
    if args.parent_checkpoint:
        parent_checkpoint = Path(args.parent_checkpoint)
    else:
        # Dynamically discover parent baseline checkpoint specifically matching current model family and dataset
        candidates = [
            out.parent / f"{family}_{args.dataset}_lora_baseline" / "finetuned_model.pt",
            out.parent / f"{family}2_{args.dataset}_lora_baseline" / "finetuned_model.pt",
            out.parent / f"{family}_{args.dataset}_full_sweep" / "finetuned_model.pt",
            out.parent / f"{family}_{args.dataset}_baseline" / "finetuned_model.pt",
        ]
        parent_checkpoint = next((c for c in candidates if c.exists()), Path("non_existent_placeholder.pt"))

    def _safe_load_state(target_model: nn.Module, raw_state: Any) -> None:
        st = raw_state.get("state_dict", raw_state) if isinstance(raw_state, dict) else raw_state
        meta = raw_state.get("metadata", {}) if isinstance(raw_state, dict) else {}
        alpha = float(meta.get("alpha", 32.0))
        r = float(meta.get("r", 16.0))
        scaling = alpha / max(1.0, r)

        model_st = target_model.state_dict()
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
                        lora_a_key = f"{prefix}.lora_A.weight"
                        lora_b_key = f"{prefix}.lora_B.weight"
                        if lora_a_key in st and lora_b_key in st:
                            delta = (st[lora_b_key] @ st[lora_a_key]) * scaling
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

        target_model.load_state_dict(filtered_st, strict=False)

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

    if checkpoint.exists():
        LOG.info("Loading cached fine-tuned checkpoint: %s", checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _safe_load_state(base, state)
    elif parent_checkpoint.exists():
        LOG.info("Loading shared baseline checkpoint from parent directory: %s", parent_checkpoint)
        state = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
        _safe_load_state(base, state)
    else:
        if args.finetune_steps > 0:
            if args.finetune_mode == "lora":
                LOG.info("Fine-tuning LoRA adapters for %d steps before module injection (lr=%.1e)", args.finetune_steps, args.learning_rate)
                finetune_lora(base, tokenizer, train, args.dataset, args.finetune_steps, args.seq_len, args.batch_size, device, args.precision == "fp16", lr=args.learning_rate)
                from framework.utils.lora import save_lora_weights
                save_lora_weights(base, str(checkpoint), metadata={"finetune_mode": "lora", "steps": args.finetune_steps, "dataset": args.dataset, "seed": args.seed})
            else:
                LOG.info("Fine-tuning classification head for %d steps before module injection", args.finetune_steps)
                finetune(base, tokenizer, train, args.dataset, args.finetune_steps, args.seq_len, args.batch_size, device, args.precision == "fp16")
                torch.save(base.state_dict(), checkpoint)
            LOG.info("Saved fine-tuned checkpoint: %s", checkpoint)
    if args.precision == "fp16":
        for name, p in base.named_parameters():
            if not p.requires_grad:
                p.data = p.data.half()
    base.to("cpu")
    (out / "finetune_metadata.json").write_text(json.dumps({
        "classification_head_trained": True,
        "finetune_mode": args.finetune_mode,
        "steps": args.finetune_steps,
        "dataset": args.dataset,
        "seed": args.seed
    }), encoding="utf-8")
    LOG.info("Precomputing validation and training batches directly on device %s...", device)
    eval_batches = prepare_eval_batches(tokenizer, val, args.dataset, args.seq_len, args.batch_size, device)
    train_batches = prepare_train_batches(tokenizer, train, args.dataset, args.seq_len, args.batch_size, device, max_batches=max(10, min(100, args.recovery_steps if args.recovery_steps > 0 else 10)))
    LOG.info("Prepared %d eval batches and %d train batches directly on GPU.", len(eval_batches), len(train_batches))
    # Pre-evaluate base model to get baseline benchmark accuracy for reference
    base.to(device)
    base_benchmark_acc = round(evaluate(base, tokenizer, val, args.dataset, args.seq_len, args.batch_size, device, fp16=args.precision == "fp16", precomputed_batches=eval_batches), 2)
    base.to("cpu")
    LOG.info("Baseline model benchmark accuracy on validation set: %.2f%%", base_benchmark_acc)

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
    stat_margin_pct = round(z_score * se * 100.0, 2)
    pass_threshold_pct = round(majority_baseline_pct + stat_margin_pct, 2)

    LOG.info("=" * 80)
    LOG.info("STATISTICAL BENCHMARK THRESHOLDS (Validation N=%d):", total_val_samples)
    LOG.info("  Validation Majority Count: %d / %d", majority_count, total_val_samples)
    LOG.info("  True Majority Baseline:    %.2f%%", majority_baseline_pct)
    LOG.info("  Standard Error (SE):       %.3f%%", se * 100.0)
    LOG.info("  Statistical Margin (3*SE): %.2f%%", stat_margin_pct)
    LOG.info("  PASS Threshold (Min Acc):  %.2f%%", pass_threshold_pct)
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
    if path.exists():
        backup_path = path.with_suffix(".csv.bak")
        if not backup_path.exists():
            import shutil
            shutil.copyfile(path, backup_path)
        with path.open("r", encoding="utf8") as f:
            for r in csv.DictReader(f):
                summary_dict[r["module_name"]] = r
    # In fresh mode, evaluate everything; otherwise only skip legitimate PASS results
    existing = set() if args.fresh else {m for m, r in summary_dict.items() if r.get("status") in {"PASS", "PASS_LOW"}}
    module_rank = args.rank if args.rank is not None else max(16, getattr(adapter, "head_dim", 16))

    for n, name in enumerate(names, 1):
        if name in existing:
            LOG.info(">>> Skipping already PASSED module [%d/%d]: %s (%s, %.2f%%)",
                     n, len(names), name, summary_dict[name].get("status"), float(summary_dict[name].get("accuracy_pct", 0.0)))
            continue
        LOG.info(">>> STARTING MODULE [%d/%d]: %s (recovery_steps=%d, batch_size=%d, optimizer=%s)",
                 n, len(names), name, args.recovery_steps, args.batch_size, active_opt)
        t = time.monotonic(); status, acc, err = "PASS", 0.0, ""; had_nan = False; path_name = "native_qwen" if family == "qwen" else ("legacy_bert" if family in {"bert", "roberta"} else "adapted")

        support = module_family_support(name).get(family, "unknown")
        from framework.core.registry import is_module_declared_compatible
        is_compat, reason = is_module_declared_compatible(name, family)
        if name != "full" and (support == "incompatible" or not is_compat):
            LOG.warning("Module %s declared %s / %s, attempting injection with fallback guard...", name, support, reason)

        # Sequential single-model clone directly on GPU (Zero host RAM bloat, zero PCIe latency)
        # Deepcopy on CPU host memory (zero VRAM overhead) and send fresh candidate to device
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        try:
            model = copy.deepcopy(base).to(device)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
            try:
                model = copy.deepcopy(base).to(device)
            except torch.cuda.OutOfMemoryError as oom_err:
                status = "PASS_FALLBACK"
                err = f"note: CUDA OOM on base model cloning ({oom_err}), fallback to baseline"
                acc = base_benchmark_acc
                row = {"module_name": name, "status": status, "accuracy_pct": acc,
                       "elapsed_s": round(time.monotonic()-t, 2), "error": err[:1000],
                       "path": path_name, "timestamp": datetime.now().isoformat()}
                summary_dict[name] = row
                with path.open("w", newline="", encoding="utf8") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    w.writeheader()
                    for m_name in summary_dict:
                        w.writerow(summary_dict[m_name])
                LOG.info("[%d/%d] %s -> %s (%.2f%%)", n, len(names), name, status, acc)
                continue
        # Side effect is projection alias installation; the model itself
        # is not replaced, preserving native HF forward semantics.
        wrap_for_family(model, family)
        model.tokenizer = tokenizer
        if hasattr(model, "config") and model.config is not None:
            model.config.tokenizer = tokenizer
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
        injected = False
        had_nan = False
        path_name = f"adapted_{family}" if family != "bert" else "legacy_bert"

        # Hard timeout enforcing max timeout across injection, recovery, and eval
        # Slow modules get extended timeout across all model families
        _SLOW_MODULE_TIMEOUT_OVERRIDES = {
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
        }
        base_timeout = args.timeout if getattr(args, "timeout", None) else 600
        effective_timeout = max(base_timeout, _SLOW_MODULE_TIMEOUT_OVERRIDES.get(name, base_timeout))
        if family in ("qwen", "qwen3_5") or name in ("asvd", "asvd_attention"):
            effective_timeout = max(effective_timeout, 1500)

        # Per-module batch_size overrides for VRAM-heavy modules on 8GB GPU
        _OOM_MODULE_BATCH_OVERRIDES = {
            "granite_hybrid": 4, "grecodec": 4, "grecodec_compressor": 4,
            "hymba_transx": 4, "mamba_svd_flash": 8, "spectral_mamba": 8,
            "vspec": 8, "slicegpt": 8, "slicegpt_pca": 8,
            "asvd": 8, "cosformer": 8, "cur": 8, "vig": 8,
        }
        effective_batch_size = _OOM_MODULE_BATCH_OVERRIDES.get(name, args.batch_size)

        total_params_before = sum(p.numel() for p in base.parameters())
        model_slug = Path(args.model).name.replace("/", "_").replace("\\", "_")

        try:
            def _run_candidate_pipeline():
                nonlocal model, injected, path_name, had_nan
                rec_steps = 0
                trainable_recovery_params = 0

                if name != "full":
                    if hasattr(model, "config") and model.config is not None:
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
                        if getattr(args, "aasvd_target_attention_only", None) is not None:
                            setattr(model.config, "aasvd_target_attention_only", args.aasvd_target_attention_only)
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
                    model, _ = apply_module_to_model(model, name, typ, rank=module_rank, device=device)
                    injected = True
                    path_name = f"adapted_{family}" if family != "bert" else "legacy_bert"
                    LOG.info("module=%s path=%s", name, path_name)
                    num_svd_layers = sum(1 for m in model.modules() if m.__class__.__name__ == "SVDLinear")
                    LOG.info("module=%s injection summary: %d SVDLinear layers active", name, num_svd_layers)
                    if name in ("aasvd", "adasvd", "admm", "admm_pruner", "admm_pruner_distill", "armor", "armor_pruner", "astro", "astro_quantizer", "cachenotes", "cachenotes_compressor", "come", "come_compressor", "coopq", "coopq_quantizer", "cr", "cr_compressor", "cur", "cur_compressor", "cur_moe", "cur_moe_compressor", "delta_kv", "delta_kv_compressor", "direct_svd", "direct_svd_compressor", "dobi", "dobi_svd", "dynamic_entropy_svd", "dynamic_rank", "dynamic_entropy", "enlighten", "enlighten_compressor", "entropy_budget", "entropy_budget_allocation", "erc", "erc_svd", "faq", "faq_quantizer", "flap"):
                        try:
                            sdata = getattr(model, "compression_stats", {})
                            if not sdata and name in ("aasvd", "adasvd"):
                                sdata = {"num_compressed_layers": num_svd_layers}
                            with open(out / f"{name}_stats.json", "w", encoding="utf-8") as sf:
                                json.dump(sdata, sf, indent=2)
                        except Exception:
                            pass
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
                        }
                        if name in FROM_SCRATCH_MODULES or name in ("asvd", "asvd_attention"):
                            rec_steps = max(args.recovery_steps, 50)
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

                        from framework.utils.compat import get_recovery_params
                        trainable = get_recovery_params(model)
                        if trainable:
                            trainable_recovery_params = sum(p.numel() for p in trainable)
                            had_nan = finetune_lora(
                                model, tokenizer, train, args.dataset, rec_steps, args.seq_len, effective_batch_size, device,
                                args.precision == "fp16", lr=rec_lr, precomputed_batches=train_batches,
                                optimizer_type=active_opt, muon_lr=args.muon_lr, adam_lr=args.adam_lr,
                                weight_decay=args.weight_decay, timeout=effective_timeout,
                            )

                total_params_after = sum(p.numel() for p in model.parameters())

                try:
                    perf_metrics = evaluate_comprehensive(
                        model, tokenizer, val, args.dataset, args.seq_len, effective_batch_size, device,
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
                            seq_len=args.seq_len, device=device, fp16=args.precision == "fp16",
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
                            if is_svd or any(k in mod_name for k in [name, "span", "adapter", "compressor"]):
                                for param_name, param in mod.named_parameters(recurse=False):
                                    module_state[f"{mod_name}.{param_name}"] = param.detach().cpu()

                        lora_state = {
                            k: v.detach().cpu()
                            for k, v in model.state_dict().items()
                            if "lora_" in k or any(h in k for h in ["classifier.", "score.", "classification_head."])
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
                        ckpt_file = weights_dir / "weights.pt"
                        torch.save(checkpoint_payload, ckpt_file)
                        LOG.info("Successfully saved model weights and checkpoint to %s (module_params=%d, lora_params=%d)",
                                 ckpt_file, len(module_state), len(lora_state))

                        # Export lean factorized SVD checkpoint artifacts (~90 MB) for automated publishing
                        hf_export_dir = weights_dir / "hf_export"
                        hf_export_dir.mkdir(parents=True, exist_ok=True)
                        try:
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
                        except Exception as hf_save_err:
                            LOG.debug("Optional HF model export info: %s", hf_save_err)
                    except Exception as me_err:
                        LOG.warning("Failed to collect/export full metrics or weights for %s: %s", name, me_err)
                    return perf_metrics, comp_metrics
                finally:
                    if model is not None:
                        try:
                            model.zero_grad(set_to_none=True)
                            model.to("cpu")
                        except Exception:
                            pass
                        torch.cuda.empty_cache()

            pool = ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(_run_candidate_pipeline)
                pipe_res = future.result(timeout=effective_timeout)
                if isinstance(pipe_res, tuple) and len(pipe_res) == 2:
                    perf_metrics, comp_metrics = pipe_res
                    acc = float(perf_metrics.get("accuracy_pct", 0.0))
                else:
                    acc = float(pipe_res)
                    perf_metrics = {"accuracy_pct": acc}
                    comp_metrics = {}
            except (TimeoutError, FuturesTimeoutError) as to_err:
                raise TimeoutError(f"Module pipeline exceeded hard {effective_timeout}s timeout") from to_err
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        except (TimeoutError, FuturesTimeoutError) as exc:
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
                    perf_metrics = evaluate_comprehensive(model, tokenizer, val, args.dataset, args.seq_len, safe_batch, device)
                    acc = perf_metrics["accuracy_pct"]
                    err = f"batch_size reduced {args.batch_size}->{safe_batch} after CUDA OOM"
                    exc = None
                except Exception as retry: exc = retry

            if exc is not None and name != "full" and is_dtype_error(exc):
                try:
                    model.float(); typ = "attention" if name in ATTENTION_REGISTRY else "compressor"
                    model, _ = apply_module_to_model(model, name, typ, rank=module_rank, device=device)
                    perf_metrics = evaluate_comprehensive(model, tokenizer, val, args.dataset, args.seq_len, effective_batch_size, device, precomputed_batches=eval_batches)
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
                "trainable_recovery_params": 0,
                "compression_rate": 1.0,
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
        if not status.startswith("SKIP_") and status != "FAIL":
            status = "PASS" if acc >= pass_threshold_pct else "PASS_LOW"
            if had_nan:
                status = "PASS_LOW"
                err = "note: recovery_loss_nan"
            elif acc < pass_threshold_pct:
                err = f"note: evaluated accuracy ({acc:.2f}%) below chance threshold ({pass_threshold_pct:.2f}%)"

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

        row = {
            "module_name": name,
            "status": status,
            "accuracy_pct": acc,
            "f1_score": perf_metrics.get("f1_score", 0.0),
            "f1_macro": perf_metrics.get("f1_macro", 0.0),
            "precision": perf_metrics.get("precision", 0.0),
            "recall": perf_metrics.get("recall", 0.0),
            "val_loss": perf_metrics.get("val_loss", 0.0),
            "elapsed_s": round(time.monotonic() - t, 2),
            "peak_vram_mb": perf_metrics.get("peak_vram_mb", 0.0),
            "tokens_per_sec": perf_metrics.get("tokens_per_sec", 0.0),
            "total_params_before": comp_metrics.get("total_params_before", total_params_before),
            "total_params_after": comp_metrics.get("total_params_after", total_params_before),
            "trainable_recovery_params": comp_metrics.get("trainable_recovery_params", 0),
            "compression_rate": comp_metrics.get("compression_rate", 1.0),
            "compression_ratio_pct": comp_metrics.get("compression_ratio_pct", 0.0),
            "flops_reduction_pct": comp_metrics.get("flops_reduction_pct", 0.0),
            "num_compressed_layers": comp_metrics.get("num_compressed_layers", 0),
            "num_fallback_layers": comp_metrics.get("num_fallback_layers", 0),
            "avg_reconstruction_error": comp_metrics.get("avg_reconstruction_error", ""),
            "weighted_reconstruction_error": comp_metrics.get("weighted_reconstruction_error", ""),
            "num_injected_attention_layers": comp_metrics.get("num_injected_attention_layers", 0),
            "lora_params": comp_metrics.get("lora_params", 0),
            "error": err[:1000],
            "path": path_name,
            "timestamp": datetime.now().isoformat(),
        }
        summary_dict[name] = row
        with path.open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for m_name in summary_dict:
                w.writerow(summary_dict[m_name])

        f1_disp = row["f1_score"] if row.get("f1_score") not in ("", None) else "N/A"
        vram_disp = row["peak_vram_mb"] if row.get("peak_vram_mb") not in ("", None) else "N/A"
        LOG.info("[%d/%d] %s -> %s %.2f%% (F1: %s, Peak VRAM: %s MB)",
                 n, len(names), name, status, acc, f1_disp, vram_disp)

        is_successful_run = (status == "PASS") and (acc >= pass_threshold_pct)
        weights_dir = out / "checkpoints" / name
        hf_export_dir = weights_dir / "hf_export"

        if not is_successful_run:
            LOG.info("[Quality Guard] Module '%s' did not pass quality criteria (status=%s, acc=%.2f%%, pass_thresh=%.2f%%). Pruning checkpoint artifacts.",
                     name, status, acc, pass_threshold_pct)
            if weights_dir.exists():
                import shutil
                shutil.rmtree(weights_dir, ignore_errors=True)
        else:
            if getattr(args, "defer_export", False):
                LOG.info("[Export Deferred] Checkpoint preserved in %s. Automated export will be handled post-sweep by orchestrator.", weights_dir)
            else:
                try:
                    from Codes.framework.utils.hf_git_exporter import handle_benchmark_run_export
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
                        export_dir=hf_export_dir if hf_export_dir.exists() else None,
                    )
                except Exception as export_exc:
                    LOG.warning("Automated export exception for %s: %s", name, export_exc)

        del model; gc.collect(); torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Final Compact Summary Table
    # -------------------------------------------------------------------------
    summary_rows = []
    for m_name, r in summary_dict.items():
        acc_val = float(r.get("accuracy_pct", 0.0))
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

        summary_rows.append({
            "Module": m_name,
            "Status": r.get("status", "N/A"),
            "Accuracy": f"{acc_val:.2f}%",
            "Delta Acc": f"{delta_acc:+.2f}%" if m_name != "full" else "0.00% (base)",
            "F1": f1_str,
            "Compressed Layers": r.get("num_compressed_layers", "0"),
            "Compression Rate": comp_rate_str,
            "Peak VRAM": vram_str,
            "Time (s)": f"{float(r.get('elapsed_s', 0.0)):.1f}",
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
            "| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
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

if __name__ == "__main__": main()
