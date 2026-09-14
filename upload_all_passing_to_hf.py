#!/usr/bin/env python3
"""
Automated Batch Hugging Face Uploader for SVD Linear Attention Project.

Scans the `results` directory, identifies all module checkpoints that meet
the strict Quality Gate criteria:
  - status == "PASS"
  - accuracy_pct >= 56.0%

For each passing checkpoint, prepares a lean package (weights.pt, config, metrics, model card)
and uploads to the private organization repository:
  Hooshaai/svd-linear-attention-{model_slug}-{module_name}
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import pandas as pd

# Hugging Face Hub settings
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

try:
    from huggingface_hub import HfApi
except ImportError:
    print("[ERROR] huggingface_hub is required. Install with: pip install huggingface_hub")
    sys.exit(1)


def generate_model_card(module_name: str, model_slug: str, metrics: dict) -> str:
    acc = metrics.get("accuracy_pct", metrics.get("mean_accuracy", "N/A"))
    f1 = metrics.get("f1_score", "N/A")
    vram = metrics.get("peak_vram_mb", "N/A")
    comp_rate = metrics.get("compression_rate", "N/A")
    num_comp = metrics.get("num_compressed_layers", "N/A")

    return f"""---
tags:
- pytorch
- attention
- linear-attention
- svd
- efficient-transformers
- sst2
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention: `{module_name}` on `{model_slug}`

This repository contains the weights, architecture configuration, and evaluation telemetry for **`{module_name}`** evaluated on **`{model_slug}`** for the GLUE SST-2 benchmark.

## Evaluation Results & Telemetry
- **Validation Accuracy**: `{acc}%`
- **F1 Score**: `{f1}`
- **Peak VRAM**: `{vram} MB`
- **Compression Rate**: `{comp_rate}x`
- **Compressed Layers**: `{num_comp}`
- **Quality Gate Status**: `PASS` (Criteria: Status PASS & Validation Accuracy >= 56.00%)

## Architecture Overview
The module `{module_name}` applies low-rank factorisation / structured sketch projection to replace standard quadratic multi-head attention with efficient linear or hybrid complexity while preserving pretrained knowledge.

## Files
- `weights.pt`: Optimized adapter / projection weights.
- `metrics.json`: Detailed evaluation and profiling telemetry.
- `README.md`: This model card.

*Automated quality-gated release by the SVD Linear Attention framework.*
"""


def main():
    parser = argparse.ArgumentParser(description="Upload all passing checkpoints to Hugging Face")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--hf_org", type=str, default="Hooshaai", help="Hugging Face organization")
    parser.add_argument("--dry_run", action="store_true", help="Scan and report without uploading")
    parser.add_argument("--min_accuracy", type=float, default=56.0, help="Minimum accuracy percentage")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"[ERROR] Results directory '{results_path}' does not exist.")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)

    print("=" * 80)
    print(f"HUGGING FACE UPLOAD SCANNER: {results_path} (Quality Gate >= {args.min_accuracy}%)")
    print("=" * 80)

    summary_files = list(results_path.glob("**/summary.csv"))
    print(f"Found {len(summary_files)} summary.csv files across results.")

    passed_runs = []
    skipped_runs = []

    for s_path in summary_files:
        try:
            df = pd.read_csv(s_path)
        except Exception as e:
            continue

        model_dir = s_path.parent
        model_slug = model_dir.name  # distilbert, roberta, gpt2, qwen3_5
        unified_dir = model_dir.parent
        unified_name = unified_dir.name  # e.g. unified_res_svd, unified_rpc

        for _, row in df.iterrows():
            mod_name = str(row.get("module_name", "")).strip()
            if not mod_name or mod_name.lower() in ("full", "baseline", "nan"):
                continue

            status = str(row.get("status", "FAIL")).strip()
            try:
                acc = float(row.get("accuracy_pct", row.get("mean_accuracy", 0.0)))
            except Exception:
                acc = 0.0

            is_pass = (status == "PASS") and (acc >= args.min_accuracy)
            
            # Find checkpoint
            checkpoints_dir = model_dir / "checkpoints" / mod_name
            weights_file = None
            for cand in [
                checkpoints_dir / "weights.pt",
                checkpoints_dir / "hf_export" / "weights.pt",
                model_dir / "checkpoints" / "weights.pt",
                model_dir / "finetuned_model.pt",
            ]:
                if cand.exists():
                    weights_file = cand
                    break

            info = {
                "unified_name": unified_name,
                "module_name": mod_name,
                "model_slug": model_slug,
                "accuracy_pct": acc,
                "status": status,
                "f1_score": row.get("f1_score", "N/A"),
                "peak_vram_mb": row.get("peak_vram_mb", "N/A"),
                "compression_rate": row.get("compression_rate", "N/A"),
                "num_compressed_layers": row.get("num_compressed_layers", "N/A"),
                "weights_file": weights_file,
                "model_dir": model_dir,
                "checkpoints_dir": checkpoints_dir,
            }

            if is_pass and weights_file:
                passed_runs.append(info)
            else:
                skipped_runs.append(info)

    print(f"\nDiscovered {len(passed_runs)} passing runs eligible for export.")
    print(f"Discovered {len(skipped_runs)} runs not meeting criteria or missing weights.\n")

    uploaded = []
    for item in passed_runs:
        mod = item["module_name"]
        model = item["model_slug"]
        acc = item["accuracy_pct"]
        weights = item["weights_file"]

        repo_id = f"{args.hf_org}/svd-linear-attention-{model}-{mod}"
        print(f"--> [{model.upper()} | {mod}] Acc: {acc:.2f}% | Checkpoint: {weights}")

        if args.dry_run:
            print(f"    [DRY RUN] Would upload to https://huggingface.co/{repo_id}")
            continue

        export_dir = item["model_dir"] / "checkpoints" / mod / "hf_export"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Copy weights
        target_weights = export_dir / "weights.pt"
        if weights != target_weights:
            shutil.copy2(weights, target_weights)

        # Write model card
        readme_path = export_dir / "README.md"
        readme_path.write_text(generate_model_card(mod, model, item), encoding="utf-8")

        # Write metrics.json
        metrics_clean = {
            "module_name": mod,
            "model_slug": model,
            "accuracy_pct": acc,
            "status": item["status"],
            "f1_score": item["f1_score"],
            "peak_vram_mb": item["peak_vram_mb"],
            "compression_rate": item["compression_rate"],
            "num_compressed_layers": item["num_compressed_layers"],
        }
        (export_dir / "metrics.json").write_text(json.dumps(metrics_clean, indent=2), encoding="utf-8")

        # Upload
        try:
            api.create_repo(repo_id, exist_ok=True, private=True)
            for f in sorted(export_dir.rglob("*")):
                if f.is_file():
                    rel_p = f.relative_to(export_dir).as_posix()
                    api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=rel_p,
                        repo_id=repo_id,
                        repo_type="model",
                    )
            url = f"https://huggingface.co/{repo_id}"
            print(f"    [SUCCESS] Uploaded to {url}")
            uploaded.append(url)
        except Exception as e:
            print(f"    [ERROR] Upload failed for {repo_id}: {e}")

    print("\n" + "=" * 80)
    print(f"UPLOAD SUMMARY: {len(uploaded)} repositories uploaded to {args.hf_org}.")
    print("=" * 80)
    for u in uploaded:
        print(f" - {u}")


if __name__ == "__main__":
    main()
