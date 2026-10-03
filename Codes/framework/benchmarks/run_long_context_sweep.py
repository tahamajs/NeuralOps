#!/usr/bin/env python3
"""
Long-Context Sweep Runner: Evaluates compression modules on Byte-Level IMDb (4096 tokens).

Evaluates:
  - Sequence length: 4096 bytes
  - Batch size: 2 (safely fits 8 GB GPU VRAM)
  - Metrics: Classification Accuracy (%), Peak VRAM (MB), Latency (ms/sample), Compression Rate

Usage:
    python run_long_context_sweep.py --dry_run
    python run_long_context_sweep.py --model_path models/distilbert_sst2 --rank 64
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

# Framework import paths
HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
sys.path[:0] = [str(CODES), str(ROOT), str(ROOT / "Codes")]

from framework.core.registry import ATTENTION_REGISTRY
from framework.eval_all_models_datasets import apply_module_to_model
from framework.data_utils.load_byte_imdb import get_byte_imdb_dataloader

LOG = logging.getLogger("LongContextSweep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_LONG_CONTEXT_MODULES = [
    "full",
    "a3",
    "aasvd",
    "adaptive",
    "adaptive_svd",
    "adasvd",
    "aegis_hybrid",
    "aegis_v6",
    "cur_moe",
    "hybrid_svd",
    "kmeans",
    "kronecker",
    "lola",
    "minimal_svd",
    "res_svd",
    "salt",
    "soa_attn",
    "spectral_residual_lora",
    "spectron",
    "svd_compressed",
    "svd_linear",
    "swift_svd",
]

CSV_FIELDS = [
    "module_name",
    "status",
    "accuracy_pct",
    "baseline_acc",
    "delta_acc",
    "compression_rate",
    "compression_ratio_pct",
    "peak_vram_mb",
    "latency_ms_per_sample",
    "tokens_per_sec",
    "elapsed_s",
    "total_params_before",
    "total_params_after",
    "error",
    "timestamp",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 4096-token long-context evaluation sweep on Byte-Level IMDb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default="models/distilbert_sst2",
                        help="Path to pre-trained/fine-tuned classification model")
    parser.add_argument("--dataset_dir", type=str, default="Codes/framework/data/byte_imdb",
                        help="Path to pre-cached byte-imdb dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/05_perplexity_seqlen/long_context",
                        help="Directory to save sweep results and summary.csv")
    parser.add_argument("--modules", type=str, nargs="*", default=DEFAULT_LONG_CONTEXT_MODULES,
                        help="Modules to evaluate")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Evaluation batch size (kept small for 4096 tokens to avoid OOM)")
    parser.add_argument("--seq_len", type=int, default=4096,
                        help="Target sequence length in tokens/bytes")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Alias for --seq_len")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Parallel workers (default 1)")
    parser.add_argument("--max_eval_samples", type=int, default=500,
                        help="Number of test samples to evaluate (None for full split)")
    parser.add_argument("--rank", type=int, default=64,
                        help="Compression rank")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 precision")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Dry run: print planned steps without executing")
    return parser.parse_args()


def adapt_model_pos_embeddings(model: nn.Module, target_seq_len: int = 4096) -> None:
    """Expands position embeddings to target_seq_len via linear interpolation if needed."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    max_pos = getattr(cfg, "max_position_embeddings", None)
    if max_pos is not None and max_pos < target_seq_len:
        # DistilBERT
        if hasattr(model, "distilbert") and hasattr(model.distilbert, "embeddings"):
            pos_emb = model.distilbert.embeddings.position_embeddings
            old_w = pos_emb.weight.data  # [max_pos, dim]
            # Interpolate: [1, dim, max_pos] -> [1, dim, target_seq_len]
            new_w = torch.nn.functional.interpolate(
                old_w.t().unsqueeze(0), size=target_seq_len, mode="linear", align_corners=False
            ).squeeze(0).t()
            new_pos_emb = torch.nn.Embedding(target_seq_len, old_w.shape[1], device=old_w.device, dtype=old_w.dtype)
            new_pos_emb.weight.data.copy_(new_w)
            model.distilbert.embeddings.position_embeddings = new_pos_emb
            if hasattr(model.distilbert.embeddings, "register_buffer"):
                model.distilbert.embeddings.register_buffer(
                    "position_ids", torch.arange(target_seq_len).expand((1, -1)).to(old_w.device)
                )
            cfg.max_position_embeddings = target_seq_len
        # RoBERTa / BERT
        elif hasattr(model, "roberta") and hasattr(model.roberta, "embeddings"):
            pos_emb = model.roberta.embeddings.position_embeddings
            old_w = pos_emb.weight.data
            new_w = torch.nn.functional.interpolate(
                old_w.t().unsqueeze(0), size=target_seq_len + 2, mode="linear", align_corners=False
            ).squeeze(0).t()
            new_pos_emb = torch.nn.Embedding(target_seq_len + 2, old_w.shape[1], device=old_w.device, dtype=old_w.dtype)
            new_pos_emb.weight.data.copy_(new_w)
            model.roberta.embeddings.position_embeddings = new_pos_emb
            cfg.max_position_embeddings = target_seq_len + 2


def evaluate_long_context(model, dataloader, max_samples=None, device="cuda"):
    """
    Evaluates sequence classification on long context with VRAM and latency tracking.
    """
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    correct = 0
    total = 0
    t0 = time.time()
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_tokens += input_ids.numel()

            if max_samples and total >= max_samples:
                break

    elapsed_s = max(1e-4, time.time() - t0)
    acc = round(100.0 * correct / max(1, total), 2)

    if device == "cuda" and torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    else:
        peak_vram_mb = 0.0

    latency_ms = round((elapsed_s / max(1, total)) * 1000.0, 2)
    tokens_per_sec = round(total_tokens / elapsed_s, 1)

    return {
        "accuracy_pct": acc,
        "peak_vram_mb": peak_vram_mb,
        "latency_ms_per_sample": latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "elapsed_s": round(elapsed_s, 2),
        "total_evaluated": total,
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
    args.modules = parsed_mods or DEFAULT_LONG_CONTEXT_MODULES

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("LONG-CONTEXT SWEEP: BYTE-LEVEL IMDB (4096 TOKENS)")
    print(f"Model Path:      {args.model_path}")
    print(f"Dataset Dir:     {args.dataset_dir}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Modules:         {len(args.modules)} modules")
    print(f"Max Samples:     {args.max_eval_samples}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Planned evaluation sequence:")
        for idx, mod in enumerate(args.modules, 1):
            print(f"  {idx:2d}. Module: {mod:<25} | Rank: {args.rank} | Output: {args.output_dir}/{mod}/result.json")
        print("\n[DRY RUN COMPLETE] No hardware sweeps executed.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"

    # Prepare DataLoader
    loader = get_byte_imdb_dataloader(
        split="test",
        batch_size=args.batch_size,
        max_length=args.seq_len,
        dataset_dir=args.dataset_dir,
    )

    # 1. Baseline Evaluation
    LOG.info("Evaluating baseline uncompressed model on 4096-token Byte IMDb...")
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    adapt_model_pos_embeddings(base_model, target_seq_len=args.seq_len)
    if args.fp16 and device == "cuda":
        base_model.half()
    base_model.to(device)

    total_params_baseline = sum(p.numel() for p in base_model.parameters())
    base_metrics = evaluate_long_context(base_model, loader, max_samples=args.max_eval_samples, device=device)
    baseline_acc = base_metrics["accuracy_pct"]

    LOG.info("Baseline 4096-token Acc: %.2f%% | VRAM: %.1f MB | Latency: %.1f ms/sample",
             baseline_acc, base_metrics["peak_vram_mb"], base_metrics["latency_ms_per_sample"])

    results = []
    base_row = {
        "module_name": "full",
        "status": "PASS",
        "accuracy_pct": baseline_acc,
        "baseline_acc": baseline_acc,
        "delta_acc": 0.0,
        "compression_rate": 1.0,
        "compression_ratio_pct": 0.0,
        "peak_vram_mb": base_metrics["peak_vram_mb"],
        "latency_ms_per_sample": base_metrics["latency_ms_per_sample"],
        "tokens_per_sec": base_metrics["tokens_per_sec"],
        "elapsed_s": base_metrics["elapsed_s"],
        "total_params_before": total_params_baseline,
        "total_params_after": total_params_baseline,
        "error": "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    results.append(base_row)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Iterate through modules
    for idx, mod in enumerate(args.modules, 1):
        if mod == "full":
            continue

        LOG.info("[%d/%d] Evaluating long-context module: %s...", idx, len(args.modules), mod)
        t_mod = time.time()
        mod_dir = out_dir / mod
        mod_dir.mkdir(parents=True, exist_ok=True)
        res_json = mod_dir / "result.json"

        try:
            cand_model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
            adapt_model_pos_embeddings(cand_model, target_seq_len=args.seq_len)
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

            metrics = evaluate_long_context(cand_model, loader, max_samples=args.max_eval_samples, device=device)
            delta_acc = round(metrics["accuracy_pct"] - baseline_acc, 2)
            status = "PASS" if metrics["accuracy_pct"] >= 50.0 else "PASS_LOW"

            row = {
                "module_name": mod,
                "status": status,
                "accuracy_pct": metrics["accuracy_pct"],
                "baseline_acc": baseline_acc,
                "delta_acc": delta_acc,
                "compression_rate": comp_rate,
                "compression_ratio_pct": comp_pct,
                "peak_vram_mb": metrics["peak_vram_mb"],
                "latency_ms_per_sample": metrics["latency_ms_per_sample"],
                "tokens_per_sec": metrics["tokens_per_sec"],
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
            LOG.exception("Long-context module %s error: %s", mod, e)
            row = {
                "module_name": mod,
                "status": "FAIL",
                "accuracy_pct": 0.0,
                "baseline_acc": baseline_acc,
                "delta_acc": -baseline_acc,
                "compression_rate": 1.0,
                "compression_ratio_pct": 0.0,
                "peak_vram_mb": 0.0,
                "latency_ms_per_sample": 0.0,
                "tokens_per_sec": 0.0,
                "elapsed_s": round(time.time() - t_mod, 2),
                "total_params_before": total_params_baseline,
                "total_params_after": total_params_baseline,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        results.append(row)
        res_json.write_text(json.dumps(row, indent=2), encoding="utf-8")

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print(f"\n[DONE] Long-context sweep finished. Results: {summary_csv}")


if __name__ == "__main__":
    main()
