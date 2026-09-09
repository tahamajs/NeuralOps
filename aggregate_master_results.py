#!/usr/bin/env python3
"""
Master Results Aggregator - runs on GPU server.
Scans ALL results/ dirs, produces one big JSON + CSV summary,
and saves it to results/MASTER_SUMMARY_<timestamp>.json.
"""
import os, sys, csv, json, glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(os.environ.get("SVD_PROJECT_ROOT", "D:/SVD_Project"))
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "master_aggregated_results.json"
OUT_CSV  = RESULTS / "master_aggregated_results.csv"

FIELDS = [
    "module_name", "model_family", "status", "accuracy_pct", "f1_score", "f1_macro",
    "precision", "recall", "val_loss", "elapsed_s", "peak_vram_mb", "tokens_per_sec",
    "total_params_before", "total_params_after", "trainable_recovery_params",
    "compression_rate", "compression_ratio_pct", "num_compressed_layers",
    "num_injected_attention_layers", "lora_params", "error", "timestamp", "source_csv"
]

def main():
    print(f"[{datetime.now().isoformat()}] Scanning {RESULTS} ...")
    
    records = {}  # key=(module_name, model_family)
    
    # 1. Scrape all per-model summary.csv files
    csv_files = list(RESULTS.glob("unified_*/*/summary.csv"))
    print(f"  Found {len(csv_files)} model-level summary CSVs")
    
    for csv_path in csv_files:
        parts = str(csv_path).replace("\\", "/").split("/")
        model_family = parts[-2]   # e.g. distilbert, gpt2, roberta, qwen3_5
        
        try:
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            print(f"  WARNING: Could not read {csv_path}: {e}")
            continue
        
        for row in rows:
            mod = row.get("module_name", "").strip()
            if not mod or mod in ("full", "nan", ""):
                continue
            key = (mod, model_family)
            # Keep highest accuracy if duplicates
            if key in records:
                existing_acc = float(records[key].get("accuracy_pct") or 0)
                new_acc = float(row.get("accuracy_pct") or 0)
                if new_acc <= existing_acc:
                    continue
            r = {f: row.get(f, "") for f in FIELDS}
            r["model_family"] = model_family
            r["source_csv"] = str(csv_path)
            records[key] = r
    
    # 2. Scrape summary_record.json (compact records for non-PASS)
    srec_files = list(RESULTS.glob("unified_*/*/checkpoints/*/summary_record.json"))
    print(f"  Found {len(srec_files)} summary_record.json files")
    for srec_path in srec_files:
        try:
            with open(srec_path, encoding="utf-8") as f:
                rec = json.load(f)
            mod = rec.get("module_name", "").strip()
            family = rec.get("model_family", "").strip()
            if not mod or not family:
                continue
            key = (mod, family)
            if key not in records:
                r = {f: rec.get(f, "") for f in FIELDS}
                r["source_csv"] = str(srec_path)
                records[key] = r
        except Exception:
            pass
    
    all_records = list(records.values())
    print(f"\n  Total unique (module, model_family) combinations: {len(all_records)}")
    
    # 3. Build per-module cross-model summary
    by_module = defaultdict(list)
    for r in all_records:
        by_module[r["module_name"]].append(r)
    
    module_summary = {}
    for mod, rows in by_module.items():
        statuses = {r["model_family"]: r["status"] for r in rows}
        accs = {r["model_family"]: float(r.get("accuracy_pct") or 0) for r in rows}
        pass_count = sum(1 for s in statuses.values() if s == "PASS")
        pass_low_count = sum(1 for s in statuses.values() if s == "PASS_LOW")
        fail_count = sum(1 for s in statuses.values() if s in ("FAIL", ""))
        
        module_summary[mod] = {
            "module_name": mod,
            "models_run": len(rows),
            "pass_count": pass_count,
            "pass_low_count": pass_low_count,
            "fail_count": fail_count,
            "statuses": statuses,
            "accuracies": accs,
            "avg_accuracy": sum(accs.values()) / len(accs) if accs else 0,
            "max_accuracy": max(accs.values()) if accs else 0,
        }
    
    # Statistics
    total_mods = len(module_summary)
    complete_4 = sum(1 for v in module_summary.values() if v["models_run"] >= 4)
    has_any_pass = sum(1 for v in module_summary.values() if v["pass_count"] > 0)
    all_4_pass = sum(1 for v in module_summary.values() if v["pass_count"] >= 4)
    
    print(f"\n{'='*60}")
    print(f"  TOTAL UNIQUE MODULES: {total_mods}")
    print(f"  Modules with 4-model results: {complete_4}")
    print(f"  Modules with ≥1 PASS: {has_any_pass}")
    print(f"  Modules with 4× PASS: {all_4_pass}")
    print(f"  Models breakdown:")
    by_family = defaultdict(lambda: defaultdict(int))
    for r in all_records:
        by_family[r["model_family"]][r.get("status","?")] += 1
    for fam, counts in sorted(by_family.items()):
        print(f"    {fam}: {dict(counts)}")
    print(f"{'='*60}\n")
    
    # 4. Save master JSON
    master = {
        "generated_at": datetime.now().isoformat(),
        "total_module_model_pairs": len(all_records),
        "total_unique_modules": total_mods,
        "complete_4model": complete_4,
        "any_pass": has_any_pass,
        "all4_pass": all_4_pass,
        "module_summary": module_summary,
        "all_records": all_records,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, default=str)
    print(f"Saved master JSON: {OUT_JSON}")
    
    # 5. Save master CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(all_records, key=lambda x: (x.get("module_name",""), x.get("model_family",""))):
            w.writerow(r)
    print(f"Saved master CSV:  {OUT_CSV}")
    
    return master

if __name__ == "__main__":
    main()
