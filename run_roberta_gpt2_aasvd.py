"""Orchestrator: Re-run RoBERTa and GPT-2 with Correct Baselines and Proven AASVD Compression Settings.

Models Evaluated:
1. RoBERTa (models\\roberta-base) with verified ~75% fine-tuned baseline
2. GPT-2 (models\\gpt2) with 81.31% fine-tuned baseline

AASVD Proven Settings:
- rank = 432 (c_attn: 432, c_proj: clamped to 288 for GPT-2)
- max_error = 0.70
- recovery_steps = 200
- attention_only = True
- learning_rate = 2e-4
- batch_size = 8
- precision = fp16

Quality-Gated Publishing Pipeline:
- Success criteria: status PASS and accuracy >= 56.0%
- On success:
  * Lean factorized payload (~90 MB, no pytorch_model.bin)
  * Upload to private Hugging Face repo: Hooshaai/svd-linear-attention-{model}-aasvd
  * Update Docs/content/compressions/aasvd.md
  * Commit results to branch 'benchmark-results-aasvd'
- On failure:
  * Delete checkpoints, do not upload/commit.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AASVDOrchestrator")

if os.path.exists(r"D:\svd_project"):
    os.chdir(r"D:\svd_project")
project_root = Path.cwd()

sys.path[:0] = [str(project_root), str(project_root / "Codes")]

models = [
    {
        "name": "roberta",
        "slug": "roberta",
        "model_path": r"models\roberta-base",
        "family": "roberta",
        "parent_checkpoint": r"results\roberta_sst2_lora_baseline\finetuned_model.pt",
        "out_dir": r"results\unified_aasvd\roberta",
        "rank": "432",
    },
    {
        "name": "gpt2",
        "slug": "gpt2",
        "model_path": r"models\gpt2",
        "family": "gpt",
        "parent_checkpoint": r"results\gpt2_sst2_lora_baseline\finetuned_model.pt",
        "out_dir": r"results\unified_aasvd\gpt2",
        "rank": "432",
    },
]

print("=" * 85, flush=True)
print("STARTING AASVD BENCHMARK RUN FOR ROBERTA & GPT-2 (PROVEN CONFIGURATION)", flush=True)
print("Config: modules=aasvd, dataset=sst2, recovery_steps=200, lr=2e-4, rank=432, max_error=0.70, batch_size=8, precision=fp16", flush=True)
print("Policy: Evaluate both models first, then batch upload successful models to Hugging Face (Hooshaai)", flush=True)
print("=" * 85, flush=True)

eval_results = []

# Step 1: Sequentially execute AASVD on RoBERTa and GPT-2
for m in models:
    name = m["name"]
    print(f"\n{'='*30} RUNNING MODEL: {name.upper()} (AASVD) {'='*30}", flush=True)
    out_dir = Path(m["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
        "--model", m["model_path"],
        "--model_family", m["family"],
        "--dataset", "sst2",
        "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
        "--output_dir", str(out_dir),
        "--modules", "aasvd",
        "--batch_size", "8",
        "--recovery_steps", "200",
        "--learning_rate", "2e-4",
        "--lora_rank", "32",
        "--rank", m["rank"],
        "--aasvd_max_error", "0.70",
        "--aasvd_target_attention_only", "true",
        "--timeout", "1200",
        "--precision", "fp16",
        "--defer_export",
        "--fresh",
    ]
    if m["parent_checkpoint"] and os.path.exists(m["parent_checkpoint"]):
        cmd.extend(["--parent_checkpoint", m["parent_checkpoint"]])
        print(f"Using verified baseline checkpoint: {m['parent_checkpoint']}", flush=True)
    else:
        print(f"Parent checkpoint not specified or not found, will fine-tune head for 200 steps", flush=True)

    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    # Collect result metrics
    csv_path = out_dir / "summary.csv"
    row_data = {}
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as cf:
                reader = list(csv.DictReader(cf))
                for r in reader:
                    if r.get("module_name") == "aasvd":
                        row_data = r
                        break
        except Exception as err:
            logger.warning("Error reading %s: %s", csv_path, err)

    status = row_data.get("status", "FAIL" if res.returncode != 0 else "UNKNOWN")
    try:
        acc = float(row_data.get("accuracy_pct", 0.0))
    except (ValueError, TypeError):
        acc = 0.0

    # Read full metrics JSON for detailed reporting
    full_metrics = {}
    json_path = out_dir / f"{Path(m['model_path']).name}_aasvd_full_metrics.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                full_metrics = json.load(jf)
        except Exception:
            pass

    eval_results.append({
        "name": name,
        "slug": m["slug"],
        "model_path": m["model_path"],
        "family": m["family"],
        "out_dir": out_dir,
        "exit_code": res.returncode,
        "status": status,
        "accuracy_pct": acc,
        "elapsed_s": elapsed,
        "row_data": row_data,
        "full_metrics": full_metrics,
    })
    print(f">>> COMPLETED {name.upper()}: Status={status}, Acc={acc:.2f}%, Elapsed={elapsed:.1f}s", flush=True)

# Step 2: Display Summary Table
print("\n" + "=" * 105, flush=True)
print("AASVD EVALUATION RESULTS SUMMARY: RoBERTa & GPT-2 on SST-2", flush=True)
print("=" * 105, flush=True)
print(f"{'Model':<12} | {'Status':<10} | {'Acc (%)':<9} | {'F1':<8} | {'Comp Rate':<10} | {'Compressed':<11} | {'Fallback':<9} | {'Avg Err':<8} | {'VRAM (MB)':<10}")
print("-" * 105, flush=True)

for r in eval_results:
    row = r["row_data"]
    f1 = row.get("f1_score", "N/A")
    comp_rate = row.get("compression_rate", "N/A")
    num_comp = row.get("num_compressed_layers", "N/A")
    num_fall = row.get("num_fallback_layers", "0")
    avg_err = row.get("avg_reconstruction_error", "N/A")
    vram = row.get("peak_vram_mb", "N/A")
    print(f"{r['name']:<12} | {r['status']:<10} | {r['accuracy_pct']:>7.2f}% | {str(f1)[:6]:<8} | {str(comp_rate):<10} | {str(num_comp):<11} | {str(num_fall):<9} | {str(avg_err)[:6]:<8} | {str(vram):<10}")
print("=" * 105, flush=True)

# Step 3: Post-run Quality Gate & Batch Export
print("\n" + "=" * 85, flush=True)
print("STAGE 2: QUALITY-GATED EXPORT TO HUGGING FACE & GITHUB", flush=True)
print("=" * 85, flush=True)

from Codes.framework.utils.hf_git_exporter import (
    find_documentation_file,
    upload_to_huggingface,
    update_documentation_table,
    commit_and_push_git,
)

hf_org = "Hooshaai"
doc_path = find_documentation_file("aasvd")
pass_count = 0
uploaded_repos = []

for r in eval_results:
    name = r["name"]
    slug = r["slug"]
    status = r["status"]
    acc = r["accuracy_pct"]
    out_dir = r["out_dir"]
    row = r["row_data"]
    weights_dir = out_dir / "checkpoints" / "aasvd"
    hf_export_dir = weights_dir / "hf_export"

    is_successful = (status == "PASS") and (acc >= 56.0)

    if not is_successful:
        print(f"[Quality Gate: REJECTED] {name} did not meet criteria (status={status}, acc={acc:.2f}% < 56.0%).", flush=True)
        if weights_dir.exists():
            print(f"  Pruning local checkpoint artifacts in {weights_dir}", flush=True)
            shutil.rmtree(weights_dir, ignore_errors=True)
        continue

    pass_count += 1
    print(f"[Quality Gate: PASSED] {name} passed criteria (status={status}, acc={acc:.2f}% >= 56.0%). Packaging for export...", flush=True)

    if not hf_export_dir.exists():
        print(f"  Warning: hf_export directory not found at {hf_export_dir}. Skipping HF upload.", flush=True)
        continue

    # Ensure lean payload (~90 MB)
    bin_file = hf_export_dir / "pytorch_model.bin"
    if bin_file.exists():
        try:
            bin_file.unlink()
            print(f"  Pruned uncompressed {bin_file.name} for lean ~90 MB payload", flush=True)
        except Exception:
            pass

    repo_id = f"{hf_org}/svd-linear-attention-{slug}-aasvd"
    perf_metrics = {
        "accuracy_pct": acc,
        "f1_score": row.get("f1_score", "N/A"),
        "precision": row.get("precision", "N/A"),
        "recall": row.get("recall", "N/A"),
        "val_loss": row.get("val_loss", "N/A"),
        "peak_vram_mb": row.get("peak_vram_mb", "N/A"),
        "tokens_per_sec": row.get("tokens_per_sec", "N/A"),
        "status": status,
    }
    comp_metrics = {
        "compression_rate": row.get("compression_rate", "1.0"),
        "compression_ratio_pct": row.get("compression_ratio_pct", "0.0"),
        "flops_reduction_pct": row.get("flops_reduction_pct", "0.0"),
        "num_compressed_layers": row.get("num_compressed_layers", "N/A"),
        "num_fallback_layers": row.get("num_fallback_layers", "0"),
        "avg_reconstruction_error": row.get("avg_reconstruction_error", "N/A"),
    }

    print(f"  Uploading lean package to Hugging Face: https://huggingface.co/{repo_id} (private)", flush=True)
    hf_url = upload_to_huggingface(
        export_dir=hf_export_dir,
        repo_id=repo_id,
        model_name=slug,
        module_name="aasvd",
        dataset_name="sst2",
        perf_metrics=perf_metrics,
        comp_metrics=comp_metrics,
    )
    if hf_url:
        print(f"  Successfully uploaded to {hf_url}", flush=True)
        uploaded_repos.append(hf_url)
    else:
        print(f"  Hugging Face upload failed or skipped for {repo_id}", flush=True)

    # Update documentation table in Docs/content/compressions/aasvd.md
    if doc_path and doc_path.exists():
        f1_val = row.get("f1_score", "N/A")
        comp_val = row.get("compression_rate", "1.00")
        svd_layers = row.get("num_compressed_layers", "N/A")
        update_documentation_table(
            doc_path=doc_path,
            model_name=slug,
            module_name="aasvd",
            dataset_name="sst2",
            status=status,
            baseline_acc=None,
            acc=acc,
            f1=f1_val,
            comp_rate=comp_val,
            num_compressed=svd_layers,
            hf_repo_url=hf_url or f"https://huggingface.co/{repo_id}",
        )
        print(f"  Updated documentation table in {doc_path.name}", flush=True)

# Step 4: Clean Git Commit and Push to Dedicated Branch
target_branch = "benchmark-results-aasvd"
print(f"\nCommitting results and documentation updates to dedicated branch '{target_branch}'...", flush=True)
commit_msg = f"feat(benchmark): AASVD evaluation for RoBERTa & GPT-2 ({pass_count}/{len(models)} passed on SST-2)"
push_res = commit_and_push_git(commit_msg, branch=target_branch)

if push_res:
    print(f"Git commit and push completed successfully on branch '{target_branch}': {push_res}", flush=True)
else:
    print(f"Git commit or push had issues. Working tree status preserved.", flush=True)

print("\n" + "=" * 85, flush=True)
print("BENCHMARK & PUBLISHING RUN COMPLETE!", flush=True)
print(f"Passed Models: {pass_count} / {len(models)}")
print(f"Uploaded Repositories: {len(uploaded_repos)}")
for u in uploaded_repos:
    print(f"  - {u}")
print("=" * 85, flush=True)
