#!/usr/bin/env python3
"""
Rebuild stale summary CSVs from authoritative result.json files in each module subdirectory.
Fixes stale CRASH records caused by old Windows runs being overwritten by correct GPU results.
"""
import csv
import json
import glob
import os
from pathlib import Path

SWEEP_DIR = Path("results/07_misc_tests/05_perplexity_seqlen/gpt2_wikitext2_sweep")

FIELDNAMES = [
    "module_name","status","accuracy_pct","f1_score","f1_macro","precision","recall",
    "val_loss","elapsed_s","peak_vram_mb","tokens_per_sec","latency_ms_per_token",
    "throughput_tok_per_sec","perplexity","long_context_accuracy","total_params_before",
    "total_params_after","trainable_recovery_params","compression_rate","compression_ratio_pct",
    "flops_reduction_pct","num_compressed_layers","num_fallback_layers",
    "avg_reconstruction_error","weighted_reconstruction_error",
    "num_injected_attention_layers","lora_params","error","path","timestamp"
]

# Find all module result.json files
result_files = sorted(SWEEP_DIR.glob("*/result.json"))
print(f"Found {len(result_files)} result.json files in {SWEEP_DIR}")

rows = []
for rfile in result_files:
    try:
        data = json.loads(rfile.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  SKIP {rfile}: {e}")
        continue
    
    module = data.get("module_name", rfile.parent.name)
    row = {f: "" for f in FIELDNAMES}
    row["module_name"]              = module
    row["status"]                   = data.get("status", "UNKNOWN")
    row["accuracy_pct"]             = data.get("accuracy_pct", 0.0)
    row["f1_score"]                 = data.get("f1_score", 0.0)
    row["f1_macro"]                 = data.get("f1_macro", 0.0)
    row["precision"]                = data.get("precision", 0.0)
    row["recall"]                   = data.get("recall", 0.0)
    row["val_loss"]                 = data.get("val_loss", 0.0)
    row["elapsed_s"]                = data.get("eval_elapsed_s", data.get("wall_elapsed_s", 0.0))
    row["peak_vram_mb"]             = data.get("peak_vram_mb", 0.0)
    row["tokens_per_sec"]           = data.get("throughput_tok_per_sec", 0.0)
    row["latency_ms_per_token"]     = data.get("latency_ms_per_token", 0.0)
    row["throughput_tok_per_sec"]   = data.get("throughput_tok_per_sec", 0.0)
    row["perplexity"]               = data.get("perplexity", "")
    row["long_context_accuracy"]    = data.get("long_context_accuracy", "")
    row["total_params_before"]      = data.get("total_params_before", 0)
    row["total_params_after"]       = data.get("total_params_after", 0)
    row["trainable_recovery_params"]= data.get("trainable_recovery_params", 0)
    row["compression_rate"]         = data.get("compression_rate", 1.0)
    row["compression_ratio_pct"]    = data.get("compression_ratio_pct", 0.0)
    row["flops_reduction_pct"]      = data.get("flops_reduction_pct", 0.0)
    row["num_compressed_layers"]    = data.get("num_compressed_layers", 0)
    row["num_fallback_layers"]      = data.get("num_fallback_layers", 0)
    row["avg_reconstruction_error"] = data.get("avg_reconstruction_error", "")
    row["weighted_reconstruction_error"] = data.get("weighted_reconstruction_error", "")
    row["num_injected_attention_layers"] = data.get("num_injected_attention_layers", 0)
    row["lora_params"]              = data.get("lora_params", 0)
    row["error"]                    = data.get("error", "")
    row["path"]                     = str(rfile.parent)
    row["timestamp"]                = data.get("timestamp", "")
    
    rows.append(row)
    print(f"  [{row['status']:6s}] {module:40s} PPL={row['perplexity']}")

# Write a single consolidated summary CSV
out_csv = SWEEP_DIR / "MASTER_SUMMARY_REBUILT.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {len(rows)} rows to {out_csv}")

# Also fix the stale per-module summary CSVs that still show CRASH
stale_csvs = list(SWEEP_DIR.glob("summary_gpt2_wt2_*.csv"))
print(f"\nFixing {len(stale_csvs)} stale per-module summary CSVs...")
for stale in stale_csvs:
    module_hint = stale.stem.replace("summary_gpt2_wt2_", "")
    # Find matching row
    match = next((r for r in rows if r["module_name"] == module_hint), None)
    if match and match["status"] != "CRASH":
        with open(stale, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerow(match)
        print(f"  FIXED: {stale.name} -> status={match['status']}, PPL={match['perplexity']}")
    else:
        print(f"  SKIP:  {stale.name} (no matching result.json or already correct)")

print("\nDone!")
