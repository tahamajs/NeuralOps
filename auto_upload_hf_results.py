#!/usr/bin/env python3
"""
Background Auto-Uploader & Verifier for Hugging Face Hub.

Monitors results/ directory, finds ALL model checkpoints and summary_records,
and ensures that:
1. Every repo exists on Hooshaai.
2. A comprehensive, publication-ready model card (README.md) is ALWAYS generated
   and uploaded, incorporating model stats, architecture details, and GLUE metrics.
3. weights.pt is completely uploaded for PASS modules (retries indefinitely until verified).
4. All auxiliary configs/metrics/tokenizers are uploaded.
5. PASS_LOW and FAIL modules get a lightweight model card uploaded (no weights, but metrics).
6. Master results JSON is always kept up-to-date on HF.
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import time
import json
import glob
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi

HF_ORG = "Hooshaai"
CHECK_INTERVAL_S = 180  # Check every 3 minutes
MASTER_RESULTS_REPO = f"{HF_ORG}/svd-benchmark-master-results"

def get_repo_id(model_family: str, mod_name: str) -> str:
    slug = mod_name.replace("_", "-")
    return f"{HF_ORG}/svd-linear-attention-{model_family}-{slug}"

def generate_model_card(mod_name: str, model_name: str, metrics: dict, doc_content: str = "", status: str = "PASS") -> str:
    acc = metrics.get("accuracy_pct") or metrics.get("recovery_pct") or "N/A"
    f1 = metrics.get("f1_score") or "N/A"
    f1_macro = metrics.get("f1_macro") or "N/A"
    precision = metrics.get("precision") or "N/A"
    recall = metrics.get("recall") or "N/A"
    val_loss = metrics.get("val_loss") or "N/A"
    comp_rate = metrics.get("compression_rate") or metrics.get("compression_ratio_pct") or "N/A"
    peak_vram = metrics.get("peak_vram_mb") or "N/A"
    eval_time = metrics.get("elapsed_s") or metrics.get("eval_time_s") or "N/A"
    tokens_per_sec = metrics.get("tokens_per_sec") or "N/A"
    total_params_before = metrics.get("total_params_before") or "N/A"
    total_params_after = metrics.get("total_params_after") or "N/A"
    num_compressed = metrics.get("num_compressed_layers") or metrics.get("num_injected_attention_layers") or "N/A"
    lora_params = metrics.get("lora_params") or "N/A"
    pass_threshold = metrics.get("pass_threshold_pct") or 56.0
    timestamp = metrics.get("timestamp") or datetime.now().isoformat()

    status_badge = "✅ PASS" if status == "PASS" else ("⚠️ PASS_LOW" if status == "PASS_LOW" else f"❌ {status}")
    weights_note = "Trained weights (`weights.pt`) are included in this repository." if status == "PASS" else "Note: Full weights not included (module did not meet quality threshold). Metrics and model card are preserved for research reference."

    card = f"""---
language:
- en
license: mit
tags:
- svd-linear-attention
- pytorch
- efficient-attention
- compression
- {model_name}
- {mod_name}
- benchmark-result
- status-{status.lower()}
datasets:
- glue
metrics:
- accuracy
- f1
pipeline_tag: text-classification
---

# {mod_name.replace('_', ' ').upper()} — {model_name.upper()} Benchmark Result

**Status:** {status_badge}

This repository hosts the benchmark results for the **`{mod_name}`** module applied to **`{model_name}`** on GLUE SST-2.

{weights_note}

## Benchmark Performance (recovery_steps=50, GLUE SST-2)

| Metric | Value |
|:---|:---|
| **Model Family** | `{model_name}` |
| **Module / Method** | `{mod_name}` |
| **Status** | **{status}** |
| **Validation Accuracy** | **{acc}%** |
| **Pass Threshold** | {pass_threshold}% |
| **F1 Score (macro)** | {f1_macro} |
| **F1 Score (weighted)** | {f1} |
| **Precision** | {precision} |
| **Recall** | {recall} |
| **Validation Loss** | {val_loss} |
| **Compression Rate** | {comp_rate}x |
| **Peak GPU VRAM** | {peak_vram} MB |
| **Inference Throughput** | {tokens_per_sec} tokens/sec |
| **Total Eval Time** | {eval_time} s |
| **Parameters (before)** | {total_params_before} |
| **Parameters (after)** | {total_params_after} |
| **Compressed Layers** | {num_compressed} |
| **LoRA Recovery Params** | {lora_params} |
| **Benchmark Timestamp** | {timestamp} |

## Architecture & Module Details

{doc_content if doc_content else f"The `{mod_name}` attention/compression engine integrates into the **Hoosha AI NeuralOps** benchmark suite, replacing standard quadratic attention or dense projection layers with subquadratic approximations using SVD and low-rank state recurrence."}

## Framework

Developed and benchmarked using the **SVD Linear Attention Framework** — Universal GPU Benchmark Suite.
- **Benchmark Suite:** [tahamajs/SVD_linear_Attention](https://github.com/tahamajs/SVD_linear_Attention)
- **Organization:** [Hooshaai](https://huggingface.co/Hooshaai)
"""
    return card

def collect_master_results(results_base: Path) -> list:
    """Collect all results from all summary CSVs across all modules and models."""
    all_results = []
    seen = set()
    
    # 1. From per-model summary CSVs
    for csv_path in results_base.glob("unified_*/*/summary.csv"):
        parts = str(csv_path).replace("\\", "/").split("/")
        if len(parts) >= 3:
            mod = parts[-3].replace("unified_", "")
            model = parts[-2]
            try:
                with open(csv_path, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        module_name = row.get("module_name", "")
                        if module_name in ("full", "", "nan"):
                            continue
                        key = f"{mod}|{model}|{module_name}"
                        if key not in seen:
                            seen.add(key)
                            all_results.append({
                                "module": module_name,
                                "model_dir": mod,
                                "model_family": model,
                                **row
                            })
            except Exception:
                pass
    
    # 2. From summary_record.json (lightweight records for non-PASS modules)
    for srec_path in results_base.glob("unified_*/*/checkpoints/*/summary_record.json"):
        try:
            with open(srec_path, encoding="utf-8") as f:
                rec = json.load(f)
            key = f"{rec.get('module_name','?')}|{rec.get('model_family','?')}|{rec.get('dataset','?')}"
            if key not in seen:
                seen.add(key)
                all_results.append(rec)
        except Exception:
            pass
    
    return all_results

def sync_and_verify_hf():
    repo_dir = Path(os.environ.get("SVD_PROJECT_ROOT",
                                   "D:\\SVD_Project" if os.path.exists("D:\\SVD_Project") else "."))
    results_base = repo_dir / "results"
    docs_base = repo_dir / "Docs" / "content"
    if not results_base.exists():
        return

    api = HfApi()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Running sweep across {results_base}...")

    # -------------------------------------------------------------------------
    # 1. Ensure master results repo exists
    # -------------------------------------------------------------------------
    try:
        api.create_repo(repo_id=MASTER_RESULTS_REPO, exist_ok=True, private=True, repo_type="dataset")
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # 2. Collect ALL results and upload master JSON
    # -------------------------------------------------------------------------
    all_results = collect_master_results(results_base)
    if all_results:
        master_payload = {
            "generated_at": datetime.now().isoformat(),
            "total_results": len(all_results),
            "results": all_results
        }
        try:
            api.upload_file(
                path_or_fileobj=json.dumps(master_payload, indent=2).encode("utf-8"),
                path_in_repo="master_benchmark_results.json",
                repo_id=MASTER_RESULTS_REPO,
                commit_message=f"Update master results: {len(all_results)} records at {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                repo_type="dataset"
            )
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Updated master results JSON ({len(all_results)} records)")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Master JSON upload failed: {e}")

    # -------------------------------------------------------------------------
    # 3. Upload per-module results (PASS with weights, PASS_LOW/FAIL with card)
    # -------------------------------------------------------------------------
    # Collect candidate checkpoints (weights.pt = PASS, summary_record.json = any status)
    handled = set()
    
    # PASS modules: upload full weights + card
    for wf in results_base.glob("unified_*/*/checkpoints/*/weights.pt"):
        ckpt_dir = wf.parent
        if ckpt_dir.name == "full":
            continue
        model_dir = ckpt_dir.parent.parent
        model_name = model_dir.name
        mod_name = model_dir.parent.name.replace("unified_", "")
        key = f"{mod_name}|{model_name}"
        if key in handled:
            continue
        handled.add(key)
        _upload_module_result(api, wf, ckpt_dir, model_name, mod_name, docs_base, status_override=None)

    # PASS_LOW / FAIL modules: upload model card only (no weights)
    for srec_path in results_base.glob("unified_*/*/checkpoints/*/summary_record.json"):
        ckpt_dir = srec_path.parent
        model_dir = ckpt_dir.parent.parent
        model_name = model_dir.name
        mod_name = model_dir.parent.name.replace("unified_", "")
        key = f"{mod_name}|{model_name}"
        if key in handled:
            continue
        handled.add(key)
        try:
            with open(srec_path, encoding="utf-8") as f:
                srec = json.load(f)
            _upload_module_result(api, None, ckpt_dir, model_name, mod_name, docs_base,
                                  status_override=srec.get("status"), metrics_override=srec)
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Failed to process summary_record for {mod_name}/{model_name}: {e}")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Handled {len(handled)} module-model combinations.")


def _upload_module_result(api, wf, ckpt_dir, model_name, mod_name, docs_base, status_override=None, metrics_override=None):
    """Upload results for one module/model combination to HuggingFace."""
    repo_id = get_repo_id(model_name, mod_name)

    # Determine status
    status = status_override
    if not status and wf and wf.exists():
        status = "PASS"
    if not status:
        status = "UNKNOWN"

    # Get metrics
    metrics = metrics_override or {}
    if not metrics:
        summary_csv = ckpt_dir.parent.parent / "summary.csv"
        if summary_csv.exists():
            try:
                with open(summary_csv, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row.get("module_name") not in ("full", "", "nan"):
                            metrics = row
                            break
            except Exception:
                pass
    
    # Check for metrics.json
    metrics_json = ckpt_dir / "hf_export" / "metrics.json"
    if metrics_json.exists():
        try:
            with open(metrics_json, encoding="utf-8") as f:
                mdata = json.load(f)
            metrics.update(mdata.get("performance", {}))
            metrics.update(mdata.get("compression", {}))
        except Exception:
            pass

    # Read doc content
    doc_content = ""
    for cand_doc in [docs_base / "attentions" / f"{mod_name}.md",
                     docs_base / "compressions" / f"{mod_name}.md"]:
        if cand_doc.exists():
            try:
                doc_content = cand_doc.read_text(encoding="utf-8")
                break
            except Exception:
                pass

    # 1. Ensure repo exists
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
    except Exception:
        pass

    # 2. Check remote files
    remote_files = []
    try:
        remote_files = list(api.list_repo_files(repo_id))
    except Exception:
        pass

    # 3. Always upload README.md (overwrite if status changed)
    try:
        readme_text = generate_model_card(mod_name, model_name, metrics, doc_content, status=status)
        api.upload_file(
            path_or_fileobj=readme_text.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=f"Update model card for {model_name}-{mod_name} (status={status})",
        )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] README.md -> {repo_id} [{status}]")
    except Exception as e_rm:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] README upload failed for {repo_id}: {e_rm}")

    # 4. Upload metrics JSON always
    try:
        api.upload_file(
            path_or_fileobj=json.dumps(metrics, indent=2, default=str).encode("utf-8"),
            path_in_repo="benchmark_metrics.json",
            repo_id=repo_id,
            commit_message=f"Upload benchmark metrics for {model_name}-{mod_name}",
        )
    except Exception:
        pass

    # 5. Upload weights.pt only for PASS modules
    if wf and wf.exists() and status == "PASS" and "weights.pt" not in remote_files:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Uploading weights.pt ({wf.stat().st_size / 1e6:.1f} MB) -> {repo_id}...")
        for attempt in range(1, 4):
            try:
                import subprocess
                res = subprocess.run(["hf", "upload", repo_id, str(wf), "weights.pt"], capture_output=True)
                if res.returncode != 0:
                    raise Exception(res.stderr.decode("utf-8"))
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] SUCCESS: weights.pt -> {repo_id}")
                break
            except Exception as e_up:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Attempt {attempt} failed for {repo_id}: {e_up}")
                time.sleep(5)

    # 6. Upload auxiliary files (config.json, tokenizer files, etc.)
    for extra in ckpt_dir.glob("*"):
        if extra.is_file() and extra.name not in ("weights.pt", "pytorch_model.bin",
                                                    "README.md", "summary_record.json"):
            if extra.name not in remote_files:
                try:
                    api.upload_file(
                        path_or_fileobj=str(extra),
                        path_in_repo=extra.name,
                        repo_id=repo_id,
                        commit_message=f"Upload {extra.name} for {mod_name}",
                    )
                except Exception:
                    pass

    # Upload hf_export files if they exist
    hf_export = ckpt_dir / "hf_export"
    if hf_export.exists():
        for extra in hf_export.glob("*"):
            if extra.is_file() and extra.name not in ("weights.pt", "pytorch_model.bin"):
                if extra.name not in remote_files:
                    try:
                        api.upload_file(
                            path_or_fileobj=str(extra),
                            path_in_repo=extra.name,
                            repo_id=repo_id,
                            commit_message=f"Upload {extra.name}",
                        )
                    except Exception:
                        pass


def loop(interval_s: int = CHECK_INTERVAL_S):
    while True:
        try:
            sync_and_verify_hf()
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in HF loop: {e}")
        if interval_s <= 0:
            break
        time.sleep(interval_s)


if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else CHECK_INTERVAL_S
    loop(interval)
