#!/usr/bin/env python3
"""
Unified Orchestrator Backend for NeuralOps.
"""

import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Paths
HERE = Path(__file__).resolve().parent
# HERE is neuralops/runners
NEURALOPS_DIR = HERE.parent
PROJECT_ROOT = NEURALOPS_DIR.parent
TRAINER_SCRIPT = NEURALOPS_DIR / "trainer" / "train_lra.py"

MODELS = [
    {"name": "GPT-2 Base", "slug": "gpt2", "family": "gpt2"},
    {"name": "RoBERTa Base", "slug": "roberta", "family": "roberta"},
    {"name": "DistilBERT Base", "slug": "distilbert", "family": "distilbert"},
    {"name": "GPT-Neo 125M", "slug": "gpt_neo", "family": "gpt_neo"},
]
PASS_THRESHOLD = 75.0

logger = logging.getLogger("UnifiedOrchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_orchestrator(module_key: str, default_args: list = None):
    # Merge default arguments with any user-provided sys.argv
    extra_args = list(default_args) if default_args else []

    # If the user passed arguments to the wrapper (e.g. --batch_size 32), append them!
    if len(sys.argv) > 1:
        extra_args.extend(sys.argv[1:])

    RESULTS_BASE = PROJECT_ROOT / "results" / "01_unified_runs" / f"unified_{module_key}"
    print("=" * 90, flush=True)
    print(f"UNIFIED 4-MODEL BENCHMARK  —  {module_key.upper()}", flush=True)
    print(f"Output : {RESULTS_BASE}", flush=True)
    print("=" * 90, flush=True)

    def run_model(m: dict) -> dict:
        out_dir = RESULTS_BASE / m["slug"]
        cmd = [
            sys.executable,
            str(TRAINER_SCRIPT),
            "--model",
            m["slug"],
            "--module",
            module_key,
            "--output_dir",
            str(out_dir),
        ] + extra_args
        logger.info("Executing: %s", " ".join(cmd))
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            logger.error(
                "Failed %s (exit %d):\n%s", m["name"], proc.returncode, proc.stderr[-1000:]
            )
        else:
            logger.info("Finished %s in %.1fs", m["name"], elapsed)

        row_data = {}
        csv_path = out_dir / "summary.csv"
        if csv_path.exists():
            try:
                with csv_path.open(encoding="utf-8") as cf:
                    for row in csv.DictReader(cf):
                        if row.get("module_name") == module_key:
                            row_data = row
                            break
            except Exception as exc:
                logger.warning("Could not parse %s: %s", csv_path, exc)

        status = row_data.get("status", "FAIL" if proc.returncode != 0 else "UNKNOWN")
        try:
            acc = float(row_data.get("accuracy_pct") or 0.0)
        except (ValueError, TypeError):
            acc = 0.0

        return {
            "name": m["name"],
            "slug": m["slug"],
            "family": m["family"],
            "out_dir": out_dir,
            "exit_code": proc.returncode,
            "status": status,
            "accuracy_pct": acc,
            "elapsed_s": elapsed,
            "row_data": row_data,
        }

    results = []
    for m in MODELS:
        try:
            results.append(run_model(m))
        except Exception as exc:
            logger.error("Unhandled error for %s: %s", m["name"], exc)
            results.append(
                {
                    "name": m["name"],
                    "slug": m["slug"],
                    "family": m["family"],
                    "out_dir": RESULTS_BASE / m["slug"],
                    "exit_code": -1,
                    "status": "CRASH",
                    "accuracy_pct": 0.0,
                    "elapsed_s": 0.0,
                    "row_data": {},
                }
            )

    W = 90
    print("\n" + "=" * W, flush=True)
    print(f"4-MODEL BENCHMARK — {module_key.upper()}", flush=True)
    print("=" * W, flush=True)
    print(
        f"{'Model':<14} | {'Status':<10} | {'Accuracy':>8} | {'Elapsed':>7} | {'Comp Rate':<10} | {'Layers':<6}",
        flush=True,
    )
    print("-" * W, flush=True)
    for r in results:
        row = r["row_data"]
        comp = row.get("compression_rate", "N/A")
        svdl = row.get("num_compressed_layers", "N/A")
        print(
            f"{r['name']:<14} | {r['status']:<10} | {r['accuracy_pct']:>6.2f}%   | {r['elapsed_s']:>5.1f}s  | {str(comp):<10} | {str(svdl):<6}",
            flush=True,
        )
    print("=" * W, flush=True)

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        row = dict(r["row_data"])
        row.update(
            {
                "model_name": r["name"],
                "model_slug": r["slug"],
                "status": r["status"],
                "accuracy_pct": r["accuracy_pct"],
                "elapsed_s": r["elapsed_s"],
            }
        )
        rows.append(row)
    csv_path = RESULTS_BASE / "4models_summary.csv"
    json_path = RESULTS_BASE / "4models_summary.json"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "row_data"} for r in results], f, indent=2)
    logger.info("Consolidated results saved → %s", RESULTS_BASE)

    try:
        from neuralops.utils.hf_git_exporter import (
            find_documentation_file,
            update_documentation_table,
            upload_to_huggingface,
        )

        hf_org = "NeuralOps"
        doc_path = find_documentation_file(module_key)
        pass_count = 0
        hf_urls = []
        for r in results:
            if r["status"] != "PASS" or r["accuracy_pct"] < PASS_THRESHOLD:
                weights_dir = r["out_dir"] / "checkpoints" / module_key
                if weights_dir.exists():
                    shutil.rmtree(weights_dir, ignore_errors=True)
                continue
            pass_count += 1
            slug, out_dir, row = r["slug"], r["out_dir"], r["row_data"]
            repo_id = f"{hf_org}/svd-linear-attention-{slug}-{module_key}"
            hf_dir = out_dir / "checkpoints" / module_key / "hf_export"
            if not hf_dir.exists():
                continue
            for big in hf_dir.glob("pytorch_model.bin"):
                big.unlink(missing_ok=True)
            perf = {
                "accuracy_pct": r["accuracy_pct"],
                "f1_score": row.get("f1_score", "N/A"),
                "val_loss": row.get("val_loss", "N/A"),
                "peak_vram_mb": row.get("peak_vram_mb", "N/A"),
                "tokens_per_sec": row.get("tokens_per_sec", "N/A"),
                "status": r["status"],
            }
            comp = {
                "compression_rate": row.get("compression_rate", "1.0"),
                "compression_ratio_pct": row.get("compression_ratio_pct", "0.0"),
                "num_compressed_layers": row.get("num_compressed_layers", "N/A"),
            }
            url = upload_to_huggingface(
                export_dir=hf_dir,
                repo_id=repo_id,
                model_name=slug,
                module_name=module_key,
                dataset_name="sst2",
                perf_metrics=perf,
                comp_metrics=comp,
            )
            if url:
                hf_urls.append(url)
            if doc_path and doc_path.exists():
                update_documentation_table(
                    doc_path=doc_path,
                    model_name=slug,
                    module_name=module_key,
                    dataset_name="sst2",
                    status=r["status"],
                    baseline_acc=None,
                    acc=r["accuracy_pct"],
                    f1=row.get("f1_score", "N/A"),
                    comp_rate=row.get("compression_rate", "1.00"),
                    num_compressed=row.get("num_compressed_layers", "N/A"),
                    hf_repo_url=url or f"https://huggingface.co/{repo_id}",
                )
    except ImportError:
        pass
    print("\nAll tasks completed.", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Error: Missing module_key. Provide it as the first argument.")
        sys.exit(1)
    run_orchestrator(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
