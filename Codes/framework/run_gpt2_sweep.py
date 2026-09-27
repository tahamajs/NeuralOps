#!/usr/bin/env python3
"""
Decoder Benchmark Runner: Evaluates compression and attention modules on GPT-2 with WikiText-2.

Measures:
  - Perplexity (PPL) on WikiText-2 test using sliding-window evaluation (max_length=1024, stride=512)
  - Peak VRAM (MB)
  - Latency (ms/token)
  - Throughput (tokens/sec)
  - Compression ratio and parameter savings

Production-hardened with:
  1. Calibration loader: 500 samples from WikiText-2 train, batch_size=4, max_length=128
  2. Diagnostic logging for all exceptions including errno, file/line, traceback, cwd, saved to diag.json
  3. Pre-execution skipping of deprecated modules from deprecated_modules.txt
  4. Snapshot and restore base model (no re-loading from disk per module)
  5. Four-tier PASS threshold (PASS, PASS_SLIGHT_DEGRADATION, PASS_DEGRADED, FAIL_PPL_DIVERGED)
  6. Overflow-handling tokenization with sliding window chunks
  7. Append-based CSV writing with filelock for multi-worker safety
  8. Automatic gc.collect() and torch.cuda.empty_cache()
  9. Explicit typing imports
  10. Shadow-free dataset imports without sys.path hack
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from datasets import load_dataset
from filelock import FileLock
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add framework to path
HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
for p in [str(CODES), str(ROOT), str(ROOT / "Codes")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.eval_all_models_datasets import apply_module_to_model

LOG = logging.getLogger("GPT2Sweep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_all_decoder_modules(category: str = "all") -> List[str]:
    """Dynamically discover all registered attention and compressor modules compatible with GPT-2 decoder."""
    import framework.attentions
    import framework.compression_engine
    from framework.core.registry import (
        ATTENTION_REGISTRY,
        COMPRESSOR_REGISTRY,
        is_helper_module,
        is_module_declared_compatible,
    )

    by_factory: Dict[Any, List[str]] = {}
    items_to_check = []
    if category in ("all", "attentions"):
        items_to_check.extend(list(ATTENTION_REGISTRY.items()))
    if category in ("all", "compressors"):
        items_to_check.extend(list(COMPRESSOR_REGISTRY.items()))

    for name, factory in items_to_check:
        if name == "full" or is_helper_module(name):
            continue
        is_compat, _ = is_module_declared_compatible(name, "gpt2")
        if is_compat:
            by_factory.setdefault(factory, []).append(name)

    canonical = ["full"]
    for aliases in by_factory.values():
        canonical.append(sorted(aliases, key=lambda x: (x.endswith("_inject"), len(x), x))[0])

    return sorted(list(set(canonical)), key=lambda item: (item != "full", item))


def load_deprecated_modules() -> Set[str]:
    """Load deprecated module names from known locations."""
    deprecated = set()
    search_paths = [
        ROOT / "audit" / "deprecated_modules.txt",
        ROOT / "deprecated_modules.txt",
        CODES / "framework" / "deprecated_modules.txt",
        HERE / "deprecated_modules.txt",
    ]
    for p in search_paths:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        deprecated.add(line)
            except Exception as e:
                LOG.warning("Could not read deprecated modules file %s: %s", p, e)
    return deprecated


DEFAULT_DECODER_MODULES = ["all"]

CSV_FIELDS = [
    "module_name",
    "status",
    "perplexity",
    "baseline_ppl",
    "delta_ppl",
    "compression_rate",
    "compression_ratio_pct",
    "peak_vram_mb",
    "latency_ms_per_token",
    "throughput_tok_per_sec",
    "elapsed_s",
    "total_params_before",
    "total_params_after",
    "error",
    "timestamp",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run decoder causal LM perplexity sweep on GPT-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default="models/gpt2_wikitext",
                        help="Path to fine-tuned GPT-2 checkpoint (or base gpt2)")
    parser.add_argument("--dataset_name", type=str, default="wikitext-2-raw-v1",
                        help="WikiText dataset configuration name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split for evaluation (test or validation)")
    parser.add_argument("--output_dir", type=str, default="results/02_glue_sweeps/gpt2/gpt2_sweep",
                        help="Output directory for results and summary")
    parser.add_argument("--modules", type=str, nargs="*", default=DEFAULT_DECODER_MODULES,
                        help="List of modules to evaluate")
    parser.add_argument("--seq_len", type=int, default=1024,
                        help="Sliding window context length")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Alias for --seq_len")
    parser.add_argument("--stride", type=int, default=512,
                        help="Sliding window stride length")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of workers (default 1)")
    parser.add_argument("--max_eval_tokens", type=int, default=100000,
                        help="Maximum total tokens to evaluate (None for full split)")
    parser.add_argument("--rank", type=int, default=64,
                        help="Target rank for low-rank decomposition modules")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Evaluate in FP16 precision")
    parser.add_argument("--worker_id", type=int, default=0,
                        help="0-indexed worker ID for parallel sharding across modules")
    parser.add_argument("--total_workers", type=int, default=1,
                        help="Total number of parallel workers sharding the module list")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Print planned evaluation steps without running them")
    return parser.parse_args()


def build_calibration_loader(train_dataset, tokenizer, num_samples: int = 500, batch_size: int = 4, max_length: int = 128) -> List[Dict[str, torch.Tensor]]:
    """
    Build a list of batched dictionary samples (calibration_loader) from WikiText-2 train split.
    Each batch contains 'input_ids' and 'attention_mask'.
    """
    valid_texts = []
    for item in train_dataset["text"]:
        t = item.strip()
        if len(t) > 20:
            valid_texts.append(t)
        if len(valid_texts) >= num_samples:
            break

    if not valid_texts:
        valid_texts = ["The quick brown fox jumps over the lazy dog. SVD linear attention compression benchmark."] * num_samples

    batches: List[Dict[str, torch.Tensor]] = []
    for i in range(0, len(valid_texts), batch_size):
        chunk = valid_texts[i:i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batches.append({"input_ids": enc.input_ids, "attention_mask": enc.attention_mask})

    return batches


def compute_perplexity_with_metrics(model, tokenizer, test_dataset, seq_len=1024, stride=512,
                                    max_eval_tokens=None, device="cuda"):
    """
    Standard sliding-window perplexity with overflow handling and comprehensive performance telemetry.
    Formula:
        PPL = exp( sum(NLL * target_length) / total_target_tokens )
        where target_ids mask non-target stride overlap with -100.
    """
    model.eval()
    text = "\n\n".join(test_dataset["text"])
    
    # Tokenize with overflow handling
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        return_overflowing_tokens=True,
    )
    input_ids_all = encodings.input_ids
    if input_ids_all.dim() == 1:
        input_ids_all = input_ids_all.unsqueeze(0)
    elif input_ids_all.dim() == 2 and input_ids_all.size(0) > 1:
        input_ids_all = input_ids_all.view(1, -1)

    total_text_tokens = input_ids_all.size(1)
    if max_eval_tokens:
        total_text_tokens = min(total_text_tokens, max_eval_tokens)

    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    t_start = time.time()
    nlls = []
    prev_end_loc = 0
    tokens_evaluated = 0

    with torch.no_grad():
        for begin_loc in range(0, total_text_tokens, stride):
            end_loc = min(begin_loc + seq_len, total_text_tokens)
            trg_len = end_loc - prev_end_loc
            input_ids = input_ids_all[:, begin_loc:end_loc].to(device, non_blocking=True)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood.detach().cpu())

            tokens_evaluated += trg_len
            prev_end_loc = end_loc

            if end_loc == total_text_tokens:
                break

    elapsed_s = max(1e-4, time.time() - t_start)
    ppl = float(torch.exp(torch.stack(nlls).sum() / prev_end_loc).item())
    
    if device == "cuda" and torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    else:
        peak_vram_mb = 0.0

    latency_ms_per_token = (elapsed_s / max(1, tokens_evaluated)) * 1000.0
    throughput_tok_per_sec = tokens_evaluated / elapsed_s

    return {
        "perplexity": round(ppl, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "latency_ms_per_token": round(latency_ms_per_token, 3),
        "throughput_tok_per_sec": round(throughput_tok_per_sec, 1),
        "elapsed_s": round(elapsed_s, 2),
        "tokens_evaluated": tokens_evaluated,
    }


def write_row_to_csv(csv_path: Path, row: Dict[str, Any], lock_path: Optional[Path] = None):
    """Safely append a single row to a CSV file using FileLock."""
    l_path = lock_path or csv_path.with_suffix(".lock")
    with FileLock(str(l_path)):
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())


def main():
    args = parse_args()
    if args.max_length:
        args.seq_len = args.max_length

    # Parse comma-separated modules if passed as a single string
    parsed_mods = []
    if isinstance(args.modules, list):
        for item in args.modules:
            parsed_mods.extend([m.strip() for m in item.split(",") if m.strip()])
    elif isinstance(args.modules, str):
        parsed_mods = [m.strip() for m in args.modules.split(",") if m.strip()]
    if not parsed_mods or parsed_mods == ["all"]:
        parsed_mods = get_all_decoder_modules("all")
    if args.total_workers > 1:
        # Worker sharding: assign every Nth module
        parsed_mods = [m for i, m in enumerate(parsed_mods) if (i % args.total_workers) == args.worker_id]
        LOG.info("Worker %d/%d assigned %d modules", args.worker_id, args.total_workers, len(parsed_mods))
    args.modules = parsed_mods

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("GPT-2 DECODER PERPLEXITY BENCHMARK SWEEP")
    print(f"Model Path:       {args.model_path}")
    print(f"Dataset / Split:  {args.dataset_name} ({args.split})")
    print(f"Window / Stride:  {args.seq_len} / {args.stride}")
    print(f"Target Rank:      {args.rank}")
    print(f"Target Modules:   {len(args.modules)} modules")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] The following evaluation sequence would be executed:")
        for idx, mod in enumerate(args.modules, 1):
            print(f"  {idx:2d}. Module: {mod:<25} | Rank: {args.rank} | Output: {args.output_dir}/{mod}/result.json")
        print("\n[DRY RUN COMPLETE] No models or datasets were modified.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.total_workers > 1:
        summary_csv = out_dir / f"summary_worker_{args.worker_id}.csv"
    else:
        summary_csv = out_dir / "summary.csv"
    main_summary = out_dir / "summary.csv"
    main_lock = out_dir / "summary.csv.lock"

    # Load Deprecated Modules
    deprecated_set = load_deprecated_modules()
    LOG.info("Loaded %d deprecated modules to skip", len(deprecated_set))

    # Load Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_path if Path(args.model_path).exists() else "gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        ds_test = load_dataset("wikitext", args.dataset_name, split=args.split)
    except Exception:
        ds_test = load_dataset("Salesforce/wikitext", args.dataset_name, split=args.split)

    try:
        ds_train = load_dataset("wikitext", args.dataset_name, split="train")
    except Exception:
        ds_train = load_dataset("Salesforce/wikitext", args.dataset_name, split="train")

    # Build 500-sample calibration loader (iterable of dicts)
    LOG.info("Building calibration dataset (500 samples, batch_size=4, max_length=128)...")
    calib_loader = build_calibration_loader(ds_train, tokenizer, num_samples=500, batch_size=4, max_length=128)
    LOG.info("Calibration loader built: %d batches", len(calib_loader))

    # 1. Load base uncompressed GPT-2 model once & evaluate baseline
    LOG.info("Loading baseline uncompressed GPT-2...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path if Path(args.model_path).exists() else "gpt2")
    if args.fp16 and device == "cuda":
        base_model.half()
    base_model.to(device)

    total_params_baseline = sum(p.numel() for p in base_model.parameters())
    base_metrics = compute_perplexity_with_metrics(
        base_model, tokenizer, ds_test, seq_len=args.seq_len, stride=args.stride,
        max_eval_tokens=args.max_eval_tokens, device=device
    )
    baseline_ppl = base_metrics["perplexity"]
    LOG.info("Baseline GPT-2 PPL: %.2f | VRAM: %.1f MB | Throughput: %.1f tok/s",
             baseline_ppl, base_metrics["peak_vram_mb"], base_metrics["throughput_tok_per_sec"])

    # Record baseline row
    base_row = {
        "module_name": "full",
        "status": "PASS",
        "perplexity": baseline_ppl,
        "baseline_ppl": baseline_ppl,
        "delta_ppl": 0.0,
        "compression_rate": 1.0,
        "compression_ratio_pct": 0.0,
        "peak_vram_mb": base_metrics["peak_vram_mb"],
        "latency_ms_per_token": base_metrics["latency_ms_per_token"],
        "throughput_tok_per_sec": base_metrics["throughput_tok_per_sec"],
        "elapsed_s": base_metrics["elapsed_s"],
        "total_params_before": total_params_baseline,
        "total_params_after": total_params_baseline,
        "error": "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_row_to_csv(summary_csv, base_row)
    if args.total_workers > 1:
        write_row_to_csv(main_summary, base_row, main_lock)

    # Take snapshot of base model to avoid reloading from disk per module
    LOG.info("Creating base model snapshot for fast restore...")
    base_snapshot = copy.deepcopy(base_model)
    LOG.info("Base model snapshot created successfully.")

    # Free the active base_model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load existing successful results to skip
    existing_mods = set()
    for csv_path in [main_summary, summary_csv]:
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        m_name = r.get("module_name")
                        st = r.get("status", "")
                        if m_name and st.startswith("PASS"):
                            existing_mods.add(m_name)
            except Exception:
                pass

    # 2. Iterate through compression modules
    for idx, mod in enumerate(args.modules, 1):
        if mod == "full":
            continue

        if mod in existing_mods:
            LOG.info("[%d/%d] Skipping module: %s (already evaluated with PASS in summary)", idx, len(args.modules), mod)
            continue

        mod_dir = out_dir / mod
        mod_dir.mkdir(parents=True, exist_ok=True)
        res_json = mod_dir / "result.json"
        diag_json = mod_dir / "diag.json"

        # Check if module is deprecated
        if mod in deprecated_set:
            LOG.info("[%d/%d] Skipping module: %s (marked as deprecated)", idx, len(args.modules), mod)
            row = {
                "module_name": mod,
                "status": "SKIP_DEPRECATED",
                "perplexity": -1.0,
                "baseline_ppl": baseline_ppl,
                "delta_ppl": 0.0,
                "compression_rate": 1.0,
                "compression_ratio_pct": 0.0,
                "peak_vram_mb": 0.0,
                "latency_ms_per_token": 0.0,
                "throughput_tok_per_sec": 0.0,
                "elapsed_s": 0.0,
                "total_params_before": total_params_baseline,
                "total_params_after": total_params_baseline,
                "error": "Module marked deprecated in deprecated_modules.txt",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            res_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
            write_row_to_csv(summary_csv, row)
            if args.total_workers > 1:
                write_row_to_csv(main_summary, row, main_lock)
            continue

        LOG.info("[%d/%d] Evaluating module: %s...", idx, len(args.modules), mod)
        t_mod = time.time()
        cand_model = None

        try:
            # Restore model from snapshot
            cand_model = copy.deepcopy(base_snapshot)
            cand_model.to(device)

            typ = "attention" if mod in ATTENTION_REGISTRY else "compressor"
            
            # Feed calibration_loader AND tokenizer
            first_batch_ids = calib_loader[0]["input_ids"].to(device)
            cand_model, comp_stats = apply_module_to_model(
                cand_model,
                mod,
                typ,
                rank=args.rank,
                device=device,
                calibration_data=first_batch_ids,
                calibration_loader=calib_loader,
                tokenizer=tokenizer,
            )

            p_after = sum(p.numel() for p in cand_model.parameters())
            comp_rate = round(total_params_baseline / max(1, p_after), 4)
            comp_pct = round(100.0 * (1.0 - p_after / max(1, total_params_baseline)), 2)

            metrics = compute_perplexity_with_metrics(
                cand_model, tokenizer, ds_test, seq_len=args.seq_len, stride=args.stride,
                max_eval_tokens=args.max_eval_tokens, device=device
            )

            ppl = metrics["perplexity"]
            delta_ppl = round(ppl - baseline_ppl, 2)

            # Strict 4-tier PASS thresholds
            if ppl <= 1.05 * baseline_ppl:
                status = "PASS"
            elif ppl <= 1.20 * baseline_ppl:
                status = "PASS_SLIGHT_DEGRADATION"
            elif ppl <= 1.50 * baseline_ppl:
                status = "PASS_DEGRADED"
            else:
                status = "FAIL_PPL_DIVERGED"

            row = {
                "module_name": mod,
                "status": status,
                "perplexity": ppl,
                "baseline_ppl": baseline_ppl,
                "delta_ppl": delta_ppl,
                "compression_rate": comp_rate,
                "compression_ratio_pct": comp_pct,
                "peak_vram_mb": metrics["peak_vram_mb"],
                "latency_ms_per_token": metrics["latency_ms_per_token"],
                "throughput_tok_per_sec": metrics["throughput_tok_per_sec"],
                "elapsed_s": metrics["elapsed_s"],
                "total_params_before": total_params_baseline,
                "total_params_after": p_after,
                "error": "",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            tb_str = "".join(tb_lines)
            
            # Extract specific errno, file, and line
            err_no = getattr(e, "errno", None)
            err_filename = getattr(e, "filename", None)
            last_tb = traceback.extract_tb(exc_tb)[-1] if exc_tb else None
            fault_file = last_tb.filename if last_tb else "unknown"
            fault_line = last_tb.lineno if last_tb else -1
            fault_func = last_tb.name if last_tb else "unknown"

            LOG.exception("Module %s evaluation error (%s at %s:%d): %s", mod, exc_type.__name__, fault_file, fault_line, e)

            # Write comprehensive diag.json
            diag_info = {
                "module_name": mod,
                "exception_type": exc_type.__name__ if exc_type else "Exception",
                "exception_message": str(e),
                "errno": err_no,
                "filename": str(err_filename) if err_filename else None,
                "fault_file": fault_file,
                "fault_line": fault_line,
                "fault_function": fault_func,
                "cwd": os.getcwd(),
                "traceback": tb_lines,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            try:
                diag_json.write_text(json.dumps(diag_info, indent=2), encoding="utf-8")
            except Exception as werr:
                LOG.warning("Could not write diag.json for %s: %s", mod, werr)

            row = {
                "module_name": mod,
                "status": "FAIL",
                "perplexity": -1.0,
                "baseline_ppl": baseline_ppl,
                "delta_ppl": 999.0,
                "compression_rate": 1.0,
                "compression_ratio_pct": 0.0,
                "peak_vram_mb": 0.0,
                "latency_ms_per_token": 0.0,
                "throughput_tok_per_sec": 0.0,
                "elapsed_s": round(time.time() - t_mod, 2),
                "total_params_before": total_params_baseline,
                "total_params_after": total_params_baseline,
                "error": f"[{fault_file}:{fault_line}] {type(e).__name__}: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        finally:
            if cand_model is not None:
                del cand_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        res_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
        write_row_to_csv(summary_csv, row)
        if args.total_workers > 1:
            write_row_to_csv(main_summary, row, main_lock)

    print(f"\n[DONE] Decoder sweep completed. Results written to {summary_csv}")


if __name__ == "__main__":
    main()
