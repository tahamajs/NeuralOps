#!/usr/bin/env python3
"""
Efficient Attention Research Framework — Full Pipeline Runner
=============================================================
Single entry-point for:
  - Training    (fine-tune any HF model on GLUE / custom data)
  - Evaluation  (standalone eval of a saved checkpoint)
  - Compression (SVD / compressor-registry methods)
  - Sweep       (multi-seed, multi-attn-type grid run)
  - List        (print all registered attention / compressor keys)

Usage examples
--------------
# Basic training
python main.py train --attn_type full --task sst2 --epochs 3

# Compression then evaluate
python main.py compress --attn_type full --task sst2 --compressor svd
python main.py eval    --attn_type full --task sst2 --model_path results/...

# Grid sweep (all attention types x seeds)
python main.py sweep --attn_types full nha lsh --seeds 42 123 --task sst2

# List registry
python main.py list
"""

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 console output on Windows to prevent UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
    pass

# ---------------------------------------------------------------------------
# Bootstrap: add framework root to sys.path and configure HF mirror/cache
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_CODES = _HERE.parent  # /Users/tahamajs/Documents/uni/Project/Codes
if str(_CODES) not in sys.path:
    sys.path.insert(0, str(_CODES))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Mirror and cache defaults
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Point to local cache if present — checked in priority order, first match wins
_potential_caches = [
    _CODES.parent / "cache",                                 # <project>/cache  (local override)
    Path.home() / ".cache" / "huggingface" / "hub",         # ~/.cache/hf/hub  (standard)
    Path.home() / ".cache" / "huggingface",                  # ~/.cache/hf
]
for pc in _potential_caches:
    if pc.exists():
        os.environ["HF_HOME"] = str(pc)
        os.environ["HF_HUB_CACHE"] = str(pc)
        os.environ["TRANSFORMERS_CACHE"] = str(pc)
        break

# ---------------------------------------------------------------------------
# Lazy heavy imports (torch / transformers loaded only when needed)
# ---------------------------------------------------------------------------

def _import_framework():
    """Import and return all framework components needed for the pipeline."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from framework.core.registry import (
        ATTENTION_REGISTRY,
        COMPRESSOR_REGISTRY,
        list_all_components,
    )
    from framework.core.utils import clear_memory, count_parameters, get_device, set_seed
    from framework.experiment_utils.config import ExperimentConfig, build_config_from_args
    from framework.trainer.callbacks import MemoryCallback, SVDAttentionCallback, TimeCallback
    from framework.trainer.metrics import get_task_metrics
    from framework.utils.replace_attention import replace_attention
    import framework.compression_engine  # auto-register all compressors
    from framework.pipeline.result_writer import ResultWriter

    return dict(
        torch=torch,
        AutoModelForSequenceClassification=AutoModelForSequenceClassification,
        AutoTokenizer=AutoTokenizer,
        DataCollatorWithPadding=DataCollatorWithPadding,
        Trainer=Trainer,
        TrainingArguments=TrainingArguments,
        ATTENTION_REGISTRY=ATTENTION_REGISTRY,
        COMPRESSOR_REGISTRY=COMPRESSOR_REGISTRY,
        list_all_components=list_all_components,
        clear_memory=clear_memory,
        count_parameters=count_parameters,
        get_device=get_device,
        set_seed=set_seed,
        ExperimentConfig=ExperimentConfig,
        get_task_metrics=get_task_metrics,
        replace_attention=replace_attention,
        ResultWriter=ResultWriter,
        MemoryCallback=MemoryCallback,
        SVDAttentionCallback=SVDAttentionCallback,
        TimeCallback=TimeCallback,
    )


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str, exp_name: str) -> logging.Logger:
    """Configure root logger to write to console + rotating file."""
    import logging
    from logging.handlers import RotatingFileHandler

    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{exp_name}_{ts}.log")

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8", errors="replace")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger("framework.main")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(cfg, tokenizer):
    """Load, tokenise, and return a HuggingFace DatasetDict."""
    try:
        from framework.data.glue_loader import (
            GLUE_TASK_KEYS,
            get_num_labels_and_is_regression,
            load_glue_local,
        )
        from framework.data.preprocessing import preprocess_function as build_preprocess

        raw = load_glue_local(cfg.task, cfg.glue_base_path)
        num_labels, is_regression = get_num_labels_and_is_regression(cfg.task)
        sent1_key, sent2_key = GLUE_TASK_KEYS[cfg.task]
        preprocess_fn = build_preprocess(
            tokenizer, cfg.task, cfg.max_len, sent1_key, sent2_key, is_regression
        )
        dataset = raw.map(preprocess_fn, batched=True, remove_columns=raw["train"].column_names)
        
        # Crop dataset to max_train_samples (e.g. 1000) if requested for fast benchmarking
        max_samples = getattr(cfg, "max_train_samples", None)
        if max_samples is not None and "train" in dataset:
            actual_count = min(len(dataset["train"]), int(max_samples))
            dataset["train"] = dataset["train"].select(range(actual_count))

        cols = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in dataset["train"].column_names:
            cols.append("token_type_ids")
        dataset.set_format(type="torch", columns=cols)
        return dataset, num_labels, is_regression
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {e}") from e


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


import json
from transformers import TrainerCallback

class LiveHistoryCallback(TrainerCallback):
    def __init__(self, out_dir):
        self.history_file = Path(out_dir) / "history" / "log_history.json"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.log_history:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(state.log_history, f, indent=2, default=str)

def run_train(cfg, fw: dict, logger: logging.Logger) -> Dict[str, Any]:
    """Full training pipeline: data → model → train → eval → save."""
    torch = fw["torch"]
    fw["set_seed"](cfg.seed)
    device = fw["get_device"]()
    logger.info("Device: %s | Seed: %s", device, cfg.seed)

    exp_name = cfg.exp_name or f"{cfg.attn_type}_{cfg.task}_s{cfg.seed}"
    out_dir = Path(cfg.output_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokeniser ──────────────────────────────────────────────────────────
    logger.info("Loading tokeniser: %s", cfg.model_name)
    try:
        tokenizer = fw["AutoTokenizer"].from_pretrained(cfg.model_name, use_fast=True)
    except Exception:
        tokenizer = fw["AutoTokenizer"].from_pretrained(cfg.model_name, use_fast=True, local_files_only=True)

    # ── Data ───────────────────────────────────────────────────────────────
    logger.info("Loading data for task: %s", cfg.task)
    dataset, num_labels, is_regression = load_data(cfg, tokenizer)
    logger.info(
        "Train: %d | Val: %d",
        len(dataset["train"]),
        len(dataset.get("validation", [])),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info("Loading model: %s  (num_labels=%d)", cfg.model_name, num_labels)
    try:
        model = fw["AutoModelForSequenceClassification"].from_pretrained(
            cfg.model_name, num_labels=num_labels
        )
    except Exception:
        model = fw["AutoModelForSequenceClassification"].from_pretrained(
            cfg.model_name, num_labels=num_labels, local_files_only=True
        )

    # Optional: swap attention layers
    if cfg.attn_type not in ("full", "baseline", "default", "none"):
        if cfg.attn_type in fw["ATTENTION_REGISTRY"]:
            logger.info("Replacing attention layers with: %s", cfg.attn_type)
            model = fw["replace_attention"](model, cfg.attn_type)
        else:
            logger.warning(
                "Attention type '%s' not in registry — using default MHA. "
                "Available: %s",
                cfg.attn_type,
                fw["ATTENTION_REGISTRY"].keys(),
            )
            
    # Full end-to-end parameter fine-tuning enabled for all attention mechanisms

    model.to(device)
    n_params = fw["count_parameters"](model)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # ── Training args ──────────────────────────────────────────────────────
    metric_map = {"cola": "matthews_correlation", "stsb": "pearson",
                  "mrpc": "f1", "qqp": "f1"}
    metric_for_best = f"eval_{metric_map.get(cfg.task, 'accuracy')}"

    training_args = fw["TrainingArguments"](
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=min(500, int(0.1 * len(dataset["train"]) / max(1, cfg.batch_size) * cfg.epochs)),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",  # STORAGE OPTIMIZATION: Do not save heavy intermediate checkpoints
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # Windows PyTorch CUDA stability fix (prevents WinError 1455 paging error)
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=cfg.max_grad_norm,
        gradient_checkpointing=cfg.gradient_checkpointing,
        save_safetensors=False,
    )
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    compute_metrics = fw["get_task_metrics"](cfg.task)

    callbacks = [fw["MemoryCallback"](), fw["TimeCallback"](), LiveHistoryCallback(out_dir)]
    if getattr(cfg, "compressor", None):
        callbacks.append(fw["SVDAttentionCallback"]())

    trainer = fw["Trainer"](
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        data_collator=fw["DataCollatorWithPadding"](tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training: %s", exp_name)
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    logger.info("Training finished in %.1fs", elapsed)

    eval_result = trainer.evaluate()
    logger.info("Evaluation results:")
    for k, v in eval_result.items():
        logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    # ── Save Results (Storage Optimized) ──────────────────────────────────
    rw = fw["ResultWriter"](out_dir)
    rw.save_config(cfg)
    rw.save_metrics(eval_result, tag="eval_final")
    rw.save_history(trainer.state.log_history)
    rw.save_system_info(elapsed_s=elapsed, n_params=n_params, device=str(device))
    rw.save_plots(trainer.state.log_history)
    rw.write_summary_md(exp_name, cfg, eval_result, elapsed)

    # Storage Zero-Bloat Guarantee: Clean all weight binaries and checkpoints
    import shutil
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    shutil.rmtree(out_dir / "model", ignore_errors=True)
    for p in out_dir.glob("*.bin"):
        p.unlink(missing_ok=True)
    for p in out_dir.glob("*.safetensors"):
        p.unlink(missing_ok=True)
    for p in out_dir.glob("*.pt"):
        p.unlink(missing_ok=True)

    if getattr(cfg, "save_model", False):
        model_dir = out_dir / "model"
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

    fw["clear_memory"]()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Results saved → %s", out_dir)
    return {"eval": eval_result, "train_runtime": elapsed, "output_dir": str(out_dir)}


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def run_eval(cfg, fw: dict, logger: logging.Logger, model_path: str) -> Dict[str, Any]:
    """Load a saved checkpoint and evaluate it on the validation split."""
    fw["set_seed"](cfg.seed)
    device = fw["get_device"]()

    tokenizer = fw["AutoTokenizer"].from_pretrained(model_path, use_fast=True)
    dataset, num_labels, _ = load_data(cfg, tokenizer)

    try:
        model = fw["AutoModelForSequenceClassification"].from_pretrained(model_path)
    except Exception:
        logger.warning("Could not load from %s, falling back to base model", model_path)
        model = fw["AutoModelForSequenceClassification"].from_pretrained(
            cfg.model_name, num_labels=num_labels
        )
    model.to(device)

    training_args = fw["TrainingArguments"](
        output_dir="./tmp_eval",
        per_device_eval_batch_size=cfg.batch_size * 2,
        remove_unused_columns=False,
        report_to="none",
    )
    eval_trainer = fw["Trainer"](
        model=model,
        args=training_args,
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        data_collator=fw["DataCollatorWithPadding"](tokenizer),
        compute_metrics=fw["get_task_metrics"](cfg.task),
    )
    result = eval_trainer.evaluate()
    result = {k.replace("eval_", ""): v for k, v in result.items()}
    logger.info("Evaluation result: %s", result)

    exp_name = cfg.exp_name or f"eval_{cfg.attn_type}_{cfg.task}"
    out_dir = Path(cfg.output_dir) / exp_name
    rw = fw["ResultWriter"](out_dir)
    rw.save_metrics(result, tag="standalone_eval")
    rw.write_summary_md(exp_name, cfg, result, elapsed_s=0.0)
    return result


# ---------------------------------------------------------------------------
# Compression pipeline
# ---------------------------------------------------------------------------

def run_compress(cfg, fw: dict, logger: logging.Logger) -> Dict[str, Any]:
    """Apply a registered compressor to the base model and report size reduction."""
    from framework.core.registry import COMPRESSOR_REGISTRY

    raw_name = getattr(cfg, "compressor", None) or getattr(cfg, "attn_type", "svd")
    compressor_name = str(raw_name).lower().strip()
    if compressor_name.startswith("compress_"):
        compressor_name = compressor_name[len("compress_"):]

    if compressor_name not in COMPRESSOR_REGISTRY:
        if f"compress_{compressor_name}" in COMPRESSOR_REGISTRY:
            compressor_name = f"compress_{compressor_name}"
        else:
            raise ValueError(
                f"Unknown compressor '{compressor_name}' (raw='{raw_name}'). "
                f"Available: {sorted(COMPRESSOR_REGISTRY.keys())}"
            )

    fw["set_seed"](cfg.seed)
    torch = fw["torch"]
    device = fw["get_device"]()
    exp_name = cfg.exp_name or f"compress_{compressor_name}_{cfg.task}"
    out_dir = Path(cfg.output_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model for compression: %s", cfg.model_name)
    try:
        model = fw["AutoModelForSequenceClassification"].from_pretrained(cfg.model_name)
        tokenizer = fw["AutoTokenizer"].from_pretrained(cfg.model_name)
    except Exception:
        model = fw["AutoModelForSequenceClassification"].from_pretrained(cfg.model_name, local_files_only=True)
        tokenizer = fw["AutoTokenizer"].from_pretrained(cfg.model_name, local_files_only=True)
    model.to(device)
    model.tokenizer = tokenizer

    orig_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    logger.info("Applying compressor: %s", compressor_name)
    compressor_cls = COMPRESSOR_REGISTRY.resolve(compressor_name)
    rank = getattr(cfg, "compression_rank", 64)
    try:
        compressor = compressor_cls(model, rank=rank)
    except TypeError:
        try:
            compressor = compressor_cls(model, cfg)
        except TypeError:
            try:
                compressor = compressor_cls(model)
            except TypeError:
                compressor = compressor_cls(model, rank=rank, tokenizer=tokenizer)
    if hasattr(compressor, "compress"):
        res = compressor.compress()
        compressed_model = res[0] if isinstance(res, tuple) else res
    elif callable(compressor) and not isinstance(compressor, torch.nn.Module):
        res = compressor()
        compressed_model = res[0] if isinstance(res, tuple) else res
    else:
        compressed_model = getattr(compressor, "model", model)

    comp_mb = sum(p.numel() * p.element_size() for p in compressed_model.parameters()) / 1024**2
    ratio = orig_mb / comp_mb if comp_mb > 0 else 1.0

    # Evaluate compressed model on task validation split
    eval_acc = 0.0
    eval_f1 = 0.0
    eval_loss = 0.0
    try:
        dataset, num_labels, _ = load_data(cfg, tokenizer)
        eval_args = fw["TrainingArguments"](
            output_dir=str(out_dir / "tmp_eval"),
            per_device_eval_batch_size=cfg.batch_size * 2,
            remove_unused_columns=False,
            report_to="none",
            save_strategy="no",
        )
        eval_trainer = fw["Trainer"](
            model=compressed_model,
            args=eval_args,
            eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
            data_collator=fw["DataCollatorWithPadding"](tokenizer),
            compute_metrics=fw["get_task_metrics"](cfg.task),
        )
        eval_res = eval_trainer.evaluate()
        eval_acc = eval_res.get("eval_accuracy", eval_res.get("eval_score", 0.0))
        eval_f1 = eval_res.get("eval_f1_macro", eval_acc)
        eval_loss = eval_res.get("eval_loss", 0.0)
        import shutil
        shutil.rmtree(out_dir / "tmp_eval", ignore_errors=True)
    except Exception as e:
        logger.warning("Could not evaluate compressed model on validation set: %s", e)

    if getattr(cfg, "save_model", False):
        model_dir = out_dir / "model"
        compressed_model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

    stats = {
        "compressor": compressor_name,
        "original_size_mb": round(orig_mb, 2),
        "compressed_size_mb": round(comp_mb, 2),
        "reduction_ratio": round(ratio, 3),
        "compression_rank": rank,
        "accuracy": eval_acc,
        "f1": eval_f1,
        "loss": eval_loss,
    }
    logger.info("Compression stats: %s", stats)

    rw = fw["ResultWriter"](out_dir)
    rw.save_config(cfg)
    rw.save_metrics(stats, tag="compression")
    rw.write_summary_md(exp_name, cfg, stats, elapsed_s=0.0)
    return stats


# ---------------------------------------------------------------------------
# Sweep pipeline
# ---------------------------------------------------------------------------

def run_sweep(args, fw: dict, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Run a grid sweep: attn_types × seeds, save aggregated results."""
    import numpy as np

    attn_types: List[str] = args.attn_types
    seeds: List[int] = args.seeds
    logger.info(
        "Starting sweep: %d attn_types × %d seeds = %d runs",
        len(attn_types), len(seeds), len(attn_types) * len(seeds),
    )

    all_results: List[Dict[str, Any]] = []
    for attn_type in attn_types:
        for seed in seeds:
            run_name = f"{attn_type}_s{seed}"
            logger.info("─" * 60)
            logger.info("Run: %s", run_name)

            cfg = fw["ExperimentConfig"]()
            cfg.model_name = args.model_name
            cfg.task       = args.task
            cfg.epochs     = args.epochs
            cfg.batch_size = args.batch_size
            cfg.lr         = args.lr
            cfg.attn_type  = attn_type
            cfg.seed       = seed
            cfg.output_dir = args.output_dir
            cfg.exp_name   = run_name
            cfg.max_len    = args.max_len
            cfg.grad_accum = args.grad_accum
            cfg.max_grad_norm = 1.0
            cfg.gradient_checkpointing = False
            cfg.glue_base_path = args.glue_base_path

            try:
                result = run_train(cfg, fw, logger)
                result["attn_type"] = attn_type
                result["seed"]      = seed
                result["run_name"]  = run_name
                all_results.append(result)
            except Exception as e:
                logger.error("Run %s failed: %s", run_name, e, exc_info=True)
                all_results.append({"run_name": run_name, "error": str(e)})

    # Aggregate across seeds per attn_type
    sweep_out = Path(args.output_dir) / f"sweep_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_out.mkdir(parents=True, exist_ok=True)

    by_attn: Dict[str, List] = {}
    for r in all_results:
        k = r.get("attn_type", "unknown")
        by_attn.setdefault(k, []).append(r)

    summary = {}
    for attn_type, runs in by_attn.items():
        good = [r for r in runs if "error" not in r]
        if not good:
            summary[attn_type] = {"error": "all runs failed"}
            continue
        # Collect numeric eval keys
        eval_keys = {k for r in good for k in r.get("eval", {}) if isinstance(r["eval"].get(k), float)}
        summary[attn_type] = {}
        for k in eval_keys:
            vals = [r["eval"][k] for r in good if k in r.get("eval", {})]
            summary[attn_type][f"{k}_mean"] = float(np.mean(vals))
            summary[attn_type][f"{k}_std"]  = float(np.std(vals))
        summary[attn_type]["n_runs"] = len(good)

    # Save sweep results
    with open(sweep_out / "all_runs.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(sweep_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Write markdown leaderboard
    _write_sweep_markdown(sweep_out, args.task, summary, attn_types)

    logger.info("Sweep complete → %s", sweep_out)
    return all_results


def _write_sweep_markdown(out_dir: Path, task: str, summary: dict, attn_types: List[str]) -> None:
    """Write a human-readable sweep summary to RESULTS.md."""
    lines = [
        f"# Sweep Summary — Task: {task}",
        f"Generated: {datetime.now().isoformat()}\n",
        "## Leaderboard\n",
    ]

    # Find all metric keys
    all_keys = sorted({k for v in summary.values() if isinstance(v, dict)
                       for k in v if k.endswith("_mean")})
    header = "| Attention Type | " + " | ".join(k.replace("_mean", "") for k in all_keys) + " |"
    sep    = "|" + "|".join(["---"] * (len(all_keys) + 1)) + "|"
    lines += [header, sep]
    for at in attn_types:
        v = summary.get(at, {})
        if "error" in v:
            row_vals = ["ERROR"] * len(all_keys)
        else:
            row_vals = [f"{v.get(k, 0):.4f} ± {v.get(k.replace('_mean','_std'), 0):.4f}"
                        for k in all_keys]
        lines.append(f"| {at} | " + " | ".join(row_vals) + " |")

    (out_dir / "RESULTS.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# List registries
# ---------------------------------------------------------------------------

def run_list(fw: Optional[dict] = None) -> None:
    """Print all registered components grouped by track."""
    import framework.compression_engine
    import framework.attentions
    from framework.eval_all_models_datasets import TRACK_MODULES
    from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

    print(f"\n=== ATTENTION MODULES ({len(ATTENTION_REGISTRY)} total) ===")
    for track, data in TRACK_MODULES.items():
        attns = data.get("attentions", [])
        if attns:
            print(f"\n  {track}:")
            for i, a in enumerate(attns):
                print(f"    {a}", end="\n" if (i+1) % 5 == 0 else "  ")
            print()

    print(f"\n=== COMPRESSION ENGINES ({len(COMPRESSOR_REGISTRY)} total) ===")
    for key in sorted(COMPRESSOR_REGISTRY.keys()):
        print(f"  {key}")



# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the full CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="Efficient Attention Framework — Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── Shared args ────────────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model_name",    default="distilbert-base-uncased")
    shared.add_argument("--attn_type",     default="full")
    shared.add_argument("--task",    "-t", default="sst2")
    shared.add_argument("--max_len",       type=int,   default=128)
    shared.add_argument("--batch_size",    type=int,   default=32)
    shared.add_argument("--epochs",        type=int,   default=3)
    shared.add_argument("--lr",            type=float, default=2e-5)
    shared.add_argument("--grad_accum",    type=int,   default=1)
    shared.add_argument("--seed",          type=int,   default=42)
    shared.add_argument("--output_dir",    default="./results")
    shared.add_argument("--log_dir",       default="./logs")
    shared.add_argument("--exp_name",      default=None)
    shared.add_argument("--glue_base_path", default=None)
    shared.add_argument("--config",        default=None, help="YAML config file")

    # train
    train_p = sub.add_parser("train", parents=[shared], help="Fine-tune a model")
    train_p.add_argument("--compressor",          default=None)
    train_p.add_argument("--compression_rank",    type=int, default=64)
    train_p.add_argument("--max_grad_norm",       type=float, default=1.0)
    train_p.add_argument("--gradient_checkpointing", action="store_true")
    train_p.add_argument("--early_stopping_patience", type=int, default=None)

    # eval
    eval_p = sub.add_parser("eval", parents=[shared], help="Evaluate a saved checkpoint")
    eval_p.add_argument("--model_path", required=True)

    # compress
    comp_p = sub.add_parser("compress", parents=[shared], help="Compress a model")
    comp_p.add_argument("--compressor",       required=True)
    comp_p.add_argument("--compression_rank", type=int, default=64)

    # sweep
    sweep_p = sub.add_parser("sweep", parents=[shared], help="Grid sweep")
    sweep_p.add_argument("--attn_types", nargs="+", default=["full"])
    sweep_p.add_argument("--seeds",      nargs="+", type=int, default=[42])

    # list
    sub.add_parser("list", help="List all registered components")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point — parse CLI args, import framework, run pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        run_list()
        return

    # ── Optional YAML config override ──────────────────────────────────────
    cfg_overrides: Dict[str, Any] = {}
    if getattr(args, "config", None):
        import yaml
        with open(args.config) as f:
            cfg_overrides = yaml.safe_load(f) or {}

    # ── Logger ─────────────────────────────────────────────────────────────
    exp_name = getattr(args, "exp_name", None) or (
        f"{getattr(args, 'attn_type', 'run')}_{getattr(args, 'task', 'task')}"
    )
    logger = setup_logger(getattr(args, "log_dir", "./logs"), exp_name)

    logger.info("=" * 60)
    logger.info("Efficient Attention Framework  |  command=%s", args.command)
    logger.info("Python %s | Platform: %s", sys.version.split()[0], platform.platform())
    logger.info("=" * 60)

    # ── Framework ──────────────────────────────────────────────────────────
    logger.info("Importing framework...")
    fw = _import_framework()

    # ── Build ExperimentConfig ─────────────────────────────────────────────
    if args.command != "sweep":
        cfg = fw["ExperimentConfig"]()
        for field in vars(cfg):
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # ── Dispatch ───────────────────────────────────────────────────────────
    try:
        if args.command == "train":
            result = run_train(cfg, fw, logger)
            print(f"\n✅ Training complete → {result['output_dir']}")

        elif args.command == "eval":
            result = run_eval(cfg, fw, logger, args.model_path)
            print(f"\n✅ Evaluation: {result}")

        elif args.command == "compress":
            result = run_compress(cfg, fw, logger)
            print(f"\n✅ Compression: {result}")

        elif args.command == "sweep":
            results = run_sweep(args, fw, logger)
            print(f"\n✅ Sweep complete — {len(results)} runs finished")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
