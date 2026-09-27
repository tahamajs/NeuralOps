#!/usr/bin/env python3
"""
Decoder Benchmark Runner: Evaluates compression modules on GPT-2 with WikiText-2.

Measures:
  - Perplexity (PPL) on WikiText-2 test using sliding-window evaluation (max_length=1024, stride=512)
  - Peak VRAM (MB)
  - Latency (ms/token)
  - Throughput (tokens/sec)
  - Compression ratio and parameter savings

Usage:
    python run_gpt2_sweep.py --dry_run
    python run_gpt2_sweep.py --model_path models/gpt2_wikitext --rank 64
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

# Sanitize sys.path before importing 'datasets' package so local Codes/framework/datasets doesn't shadow HuggingFace datasets
sys.path = [p for p in sys.path if not p.rstrip("/\\").replace("\\", "/").endswith("Codes/framework") and not p.rstrip("/\\").replace("\\", "/").endswith("framework") and p != ""]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add framework to path
HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
sys.path[:0] = [str(CODES), str(ROOT), str(ROOT / "Codes")]

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
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Print planned evaluation steps without running them")
    return parser.parse_args()


def compute_perplexity_with_metrics(model, tokenizer, test_dataset, seq_len=1024, stride=512,
                                    max_eval_tokens=None, device="cuda"):
    """
    Standard sliding-window perplexity with comprehensive performance telemetry.
    
    Formula:
        PPL = exp( sum(NLL * target_length) / total_target_tokens )
        where target_ids mask non-target stride overlap with -100.
    """
    model.eval()
    text = "\n\n".join(test_dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    total_text_tokens = encodings.input_ids.size(1)
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
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device, non_blocking=True)
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
    summary_csv = out_dir / "summary.csv"

    # Load Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_path if Path(args.model_path).exists() else "gpt2")
    try:
        ds = load_dataset("wikitext", args.dataset_name, split=args.split)
    except Exception:
        ds = load_dataset("Salesforce/wikitext", args.dataset_name, split=args.split)

    # 1. Evaluate baseline uncompressed GPT-2
    LOG.info("Evaluating baseline uncompressed GPT-2...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path if Path(args.model_path).exists() else "gpt2")
    if args.fp16 and device == "cuda":
        base_model.half()
    base_model.to(device)

    total_params_baseline = sum(p.numel() for p in base_model.parameters())
    base_metrics = compute_perplexity_with_metrics(
        base_model, tokenizer, ds, seq_len=args.seq_len, stride=args.stride,
        max_eval_tokens=args.max_eval_tokens, device=device
    )
    baseline_ppl = base_metrics["perplexity"]
    LOG.info("Baseline GPT-2 PPL: %.2f | VRAM: %.1f MB | Throughput: %.1f tok/s",
             baseline_ppl, base_metrics["peak_vram_mb"], base_metrics["throughput_tok_per_sec"])

    results = []

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
    results.append(base_row)

    # Free base model before loop
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load existing results if summary.csv exists
    existing_mods = set()
    if summary_csv.exists():
        try:
            with open(summary_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    results.append(r)
                    existing_mods.add(r.get("module_name"))
        except Exception:
            pass

    # 2. Iterate through compression modules
    for idx, mod in enumerate(args.modules, 1):
        if mod == "full":
            continue

        if mod in existing_mods:
            LOG.info("[%d/%d] Skipping module: %s (already evaluated in %s)", idx, len(args.modules), mod, summary_csv)
            continue

        LOG.info("[%d/%d] Evaluating module: %s...", idx, len(args.modules), mod)
        t_mod = time.time()
        mod_dir = out_dir / mod
        mod_dir.mkdir(parents=True, exist_ok=True)
        res_json = mod_dir / "result.json"

        try:
            # Fresh clone from checkpoint
            cand_model = AutoModelForCausalLM.from_pretrained(args.model_path if Path(args.model_path).exists() else "gpt2")
            if args.fp16 and device == "cuda":
                cand_model.half()
            cand_model.to(device)

            typ = "attention" if mod in ATTENTION_REGISTRY else "compressor"
            cand_model, comp_stats = apply_module_to_model(
                cand_model, mod, typ, rank=args.rank, device=device
            )

            p_after = sum(p.numel() for p in cand_model.parameters())
            comp_rate = round(total_params_baseline / max(1, p_after), 4)
            comp_pct = round(100.0 * (1.0 - p_after / max(1, total_params_baseline)), 2)

            metrics = compute_perplexity_with_metrics(
                cand_model, tokenizer, ds, seq_len=args.seq_len, stride=args.stride,
                max_eval_tokens=args.max_eval_tokens, device=device
            )

            delta_ppl = round(metrics["perplexity"] - baseline_ppl, 2)
            status = "PASS" if metrics["perplexity"] < baseline_ppl * 3.0 else "PASS_DEGRADED"

            row = {
                "module_name": mod,
                "status": status,
                "perplexity": metrics["perplexity"],
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

            del cand_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            LOG.exception("Module %s evaluation error: %s", mod, e)
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
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        results.append(row)
        res_json.write_text(json.dumps(row, indent=2), encoding="utf-8")

        # Update summary CSV incrementally
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print(f"\n[DONE] Decoder sweep completed. Results written to {summary_csv}")


if __name__ == "__main__":
    main()
