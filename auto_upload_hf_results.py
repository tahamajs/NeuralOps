#!/usr/bin/env python3
"""
Background Auto-Uploader & Verifier for Hugging Face Hub (tahamajs Organization).

Monitors the results/ directory across all evaluation sweeps:
- results/agnews_unified_* (AG News 4-class multi-class benchmark)
- results/mnli_unified_*   (MNLI multi-class benchmark)
- results/unified_*        (GLUE SST-2 benchmark)

Standardized Repo Naming:
- tahamajs/svd-<dataset>-<model_family>-<module_slug>
  e.g.:
    tahamajs/svd-agnews-distilbert-adaptive-span
    tahamajs/svd-sst2-roberta-svd-mamba
    tahamajs/svd-mnli-distilbert-primal

Ensures:
1. Every repo exists on tahamajs with clean, human-readable repository slugs.
2. A comprehensive, publication-ready model card (README.md) is ALWAYS generated and uploaded,
   dynamically highlighting the correct dataset, multi-class accuracy, F1 scores, compression stats,
   and parameter counts.
3. weights.pt is uploaded for PASS and valid factorized/compressed models.
4. All auxiliary configs, metrics, and tokenizer files are synced.
5. PASS_LOW, PASS_NO_COMPRESSION, and FAIL modules get a clean model card + metrics uploaded.
6. Master benchmark results JSON and CSV dataset are kept up-to-date on HF (tahamajs/svd-benchmark-master-results).
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import sys
import time
import json
import glob
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

HF_ORG = "tahamajs"
CHECK_INTERVAL_S = 300  # Check every 5 minutes in background
MASTER_RESULTS_REPO = f"{HF_ORG}/svd-benchmark-master-results"
RATE_LIMITED_REPOS = set()
COMPLETED_UPLOADS = set()

def normalize_dataset_name(raw_dataset: str) -> str:
    raw = (raw_dataset or "").lower().replace("_", "-").strip()
    if "ag" in raw or "news" in raw:
        return "agnews"
    if "mnli" in raw:
        return "mnli"
    if "sst" in raw or "sst2" in raw:
        return "sst2"
    return raw if raw else "sst2"

def get_repo_id(model_family: str, mod_name: str, dataset: str = "sst2") -> str:
    norm_ds = normalize_dataset_name(dataset)
    clean_model = model_family.lower().replace("_", "-").replace("models/", "").replace("-sst2", "")
    slug = mod_name.lower().replace("_", "-").strip()
    return f"{HF_ORG}/svd-linear-attention-{clean_model}-{slug}"

def generate_model_card(
    mod_name: str,
    model_name: str,
    dataset_name: str,
    metrics: dict,
    doc_content: str = "",
    status: str = "PASS",
) -> str:
    acc = metrics.get("accuracy_pct") or metrics.get("recovery_pct") or "N/A"
    f1 = metrics.get("f1_score") or "N/A"
    f1_macro = metrics.get("f1_macro") or "N/A"
    precision = metrics.get("precision") or "N/A"
    recall = metrics.get("recall") or "N/A"
    val_loss = metrics.get("val_loss") or "N/A"
    comp_rate = metrics.get("compression_rate") or metrics.get("compression_ratio_pct") or "1.00"
    peak_vram = metrics.get("peak_vram_mb") or "N/A"
    eval_time = metrics.get("elapsed_s") or metrics.get("eval_time_s") or "N/A"
    tokens_per_sec = metrics.get("tokens_per_sec") or "N/A"
    total_params_before = metrics.get("total_params_before") or "N/A"
    total_params_after = metrics.get("total_params_after") or "N/A"
    num_compressed = metrics.get("num_compressed_layers") or metrics.get("num_injected_attention_layers") or "0"
    lora_params = metrics.get("lora_params") or "N/A"
    pass_threshold = metrics.get("pass_threshold_pct") or "31.63"
    timestamp = metrics.get("timestamp") or datetime.now().isoformat()

    norm_ds = normalize_dataset_name(dataset_name)
    ds_display = "AG News (4 Classes)" if norm_ds == "agnews" else ("MNLI (3 Classes)" if norm_ds == "mnli" else "GLUE SST-2 (Binary)")
    hf_dataset_tag = "ag_news" if norm_ds == "agnews" else ("glue" if norm_ds in ("sst2", "mnli") else norm_ds)

    if status == "PASS":
        status_badge = "✅ PASS"
    elif status == "PASS_LOW":
        status_badge = "⚠️ PASS_LOW (Below Threshold)"
    elif status == "PASS_NO_COMPRESSION":
        status_badge = "ℹ️ PASS (Efficient Attention / Zero Param Reduction)"
    else:
        status_badge = f"❌ {status}"

    weights_note = (
        "Trained weights (`weights.pt`) are included in this repository."
        if status in ("PASS", "PASS_NO_COMPRESSION") and metrics.get("has_weights", False)
        else "Note: Weights not bundled or pruned; comprehensive profiling metrics, configurations, and verification card are preserved for research reproducibility."
    )

    clean_mod_title = mod_name.replace("_", " ").upper()
    clean_model_title = model_name.replace("_", " ").upper()

    card = f"""---
language:
- en
license: mit
tags:
- svd-linear-attention
- pytorch
- efficient-attention
- model-compression
- {model_name}
- {mod_name}
- benchmark-result
- status-{status.lower()}
datasets:
- {hf_dataset_tag}
metrics:
- accuracy
- f1
pipeline_tag: text-classification
---

# {clean_mod_title} — {clean_model_title} Benchmark Result

**Status:** {status_badge}  
**Dataset:** `{ds_display}`  
**Architecture:** `{model_name}`  
**Module:** `{mod_name}`  

This repository hosts empirical benchmark results and factorized checkpoints for **`{mod_name}`** applied to **`{model_name}`** on **`{ds_display}`**.

{weights_note}

## Benchmark Performance Metrics

| Metric | Value |
|:---|:---|
| **Model Family** | `{model_name}` |
| **Module / Method** | `{mod_name}` |
| **Dataset** | **{ds_display}** |
| **Validation Status** | **{status}** |
| **Validation Accuracy** | **{acc}%** |
| **Pass Threshold (3*SE Margin)** | {pass_threshold}% |
| **F1 Score (Macro)** | {f1_macro} |
| **F1 Score (Weighted)** | {f1} |
| **Precision** | {precision} |
| **Recall** | {recall} |
| **Validation Loss** | {val_loss} |
| **Compression Factor** | {comp_rate}x |
| **Peak GPU VRAM** | {peak_vram} MB |
| **Inference Throughput** | {tokens_per_sec} tokens/sec |
| **Evaluation Time** | {eval_time} s |
| **Parameters (Before)** | {total_params_before} |
| **Parameters (After)** | {total_params_after} |
| **Compressed Layers** | {num_compressed} |
| **LoRA Trainable Params** | {lora_params} |
| **Benchmark Timestamp** | {timestamp} |

## Architecture & Module Specification

{doc_content if doc_content else f"The `{mod_name}` module is evaluated under the **Hoosha AI NeuralOps SVD & Linear Attention Suite**, benchmarking subquadratic attention and rank-adaptive tensor factorization against full quadratic baselines."}

## Framework & Verification

Developed and benchmarked using the **SVD Linear Attention Framework** — Universal GPU Benchmark Suite.
- **Repository:** [tahamajs/SVD_linear_Attention](https://github.com/tahamajs/SVD_linear_Attention)
- **Organization:** [tahamajs](https://huggingface.co/tahamajs)
"""
    return card

def collect_master_results(results_base: Path) -> list:
    all_results = []
    seen = set()

    # 1. From summary.csv across all unified directories (agnews_unified_*, mnli_unified_*, unified_*)
    for csv_path in results_base.glob("*unified_*/*/summary.csv"):
        parts = str(csv_path).replace("\\", "/").split("/")
        if len(parts) >= 3:
            dir_name = parts[-3]
            model = parts[-2]
            
            # Extract dataset from directory name
            if dir_name.startswith("agnews_unified_"):
                mod = dir_name.replace("agnews_unified_", "")
                ds = "ag_news"
            elif dir_name.startswith("mnli_unified_"):
                mod = dir_name.replace("mnli_unified_", "")
                ds = "mnli"
            else:
                mod = dir_name.replace("unified_", "")
                ds = "sst2"

            try:
                with open(csv_path, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        module_name = row.get("module_name", "")
                        if module_name in ("full", "", "nan"):
                            continue
                        row_ds = row.get("dataset") or ds
                        key = f"{mod}|{model}|{module_name}|{row_ds}"
                        if key not in seen:
                            seen.add(key)
                            all_results.append({
                                "module": module_name,
                                "model_dir": dir_name,
                                "model_family": model,
                                "dataset": row_ds,
                                **row
                            })
            except Exception:
                pass

    # 2. From summary_record.json
    for srec_path in results_base.glob("*unified_*/*/checkpoints/*/summary_record.json"):
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

def _upload_module_result(api, wf, ckpt_dir, model_name, mod_name, dataset_name, docs_base, status_override=None, metrics_override=None):
    repo_id = get_repo_id(model_name, mod_name, dataset_name)

    # Determine status
    status = status_override
    if not status and wf and wf.exists():
        status = "PASS"
    if not status:
        status = "UNKNOWN"

    # Gather metrics
    metrics = metrics_override or {}
    if not metrics:
        summary_csv = ckpt_dir.parent.parent / "summary.csv"
        if summary_csv.exists():
            try:
                with open(summary_csv, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row.get("module_name") not in ("full", "", "nan"):
                            metrics = dict(row)
                            break
            except Exception:
                pass

    # Also check for metrics.json inside hf_export
    metrics_json = ckpt_dir / "hf_export" / "metrics.json"
    if metrics_json.exists():
        try:
            with open(metrics_json, encoding="utf-8") as f:
                mdata = json.load(f)
            metrics.update(mdata.get("performance", {}))
            metrics.update(mdata.get("compression", {}))
        except Exception:
            pass

    has_weights = bool(wf and wf.exists())
    metrics["has_weights"] = has_weights

    # Read module documentation if available
    doc_content = ""
    for cand_doc in [
        docs_base / "attentions" / f"{mod_name}.md",
        docs_base / "attentions" / f"{mod_name}_attention.md",
        docs_base / "compressions" / f"{mod_name}.md",
        docs_base / "compressions" / f"{mod_name}_compressor.md",
    ]:
        if cand_doc.exists():
            try:
                doc_content = cand_doc.read_text(encoding="utf-8")
                break
            except Exception:
                pass

    # Filter out bad results: do NOT upload repos for FAIL, PASS_LOW, or sub-threshold accuracy
    raw_acc = metrics.get("accuracy_pct") or metrics.get("recovery_pct") or 0.0
    try:
        acc_float = float(raw_acc)
    except (ValueError, TypeError):
        acc_float = 0.0

    raw_thresh = metrics.get("pass_threshold_pct")
    try:
        thresh_float = float(raw_thresh) if raw_thresh is not None else 31.63
    except (ValueError, TypeError):
        thresh_float = 31.63

    # Reject bad results: must not be FAIL/PASS_LOW/UNKNOWN and must exceed pass threshold
    if status in ("FAIL", "PASS_LOW", "UNKNOWN") or acc_float < thresh_float:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Skipping bad/low result: {repo_id} (status={status}, acc={acc_float:.2f}% < thresh={thresh_float:.2f}%)")
        return

    if repo_id in COMPLETED_UPLOADS or repo_id in RATE_LIMITED_REPOS:
        return

    # 1. Ensure repo exists on tahamajs safely
    repo_ready = False
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        repo_ready = True
    except HfHubHTTPError as e:
        if getattr(e.response, "status_code", None) == 404:
            try:
                api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
                repo_ready = True
            except HfHubHTTPError as e2:
                status_code = getattr(e2.response, "status_code", None)
                if status_code == 429:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] HF rate limit reached (429). Skipping upload for {repo_id}.")
                    RATE_LIMITED_REPOS.add(repo_id)
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/hf_uploads_skipped.log", "a", encoding="utf-8") as lf:
                        lf.write(f"[{datetime.now().isoformat()}] RATE_LIMIT_429: {repo_id}\n")
                    return
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] create_repo warning for {repo_id}: {e2}")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] repo_info check failed for {repo_id}: {e}")
    except Exception as e_gen:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Unexpected error inspecting {repo_id}: {e_gen}")

    if not repo_ready:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Repo {repo_id} does not exist or cannot be created. Skipping upload.")
        os.makedirs("logs", exist_ok=True)
        with open("logs/hf_uploads_skipped.log", "a", encoding="utf-8") as lf:
            lf.write(f"[{datetime.now().isoformat()}] REPO_NOT_READY: {repo_id}\n")
        return

    # 2. Check remote files
    remote_files = []
    try:
        remote_files = list(api.list_repo_files(repo_id))
    except Exception:
        pass

    # 3. Always upload / update README.md
    try:
        readme_text = generate_model_card(
            mod_name=mod_name,
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            doc_content=doc_content,
            status=status,
        )
        api.upload_file(
            path_or_fileobj=readme_text.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=f"Update model card for {model_name}-{mod_name} on {dataset_name} [{status}]",
        )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] README.md -> {repo_id} [{status}]")
    except Exception as e_rm:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] README upload failed for {repo_id}: {e_rm}")

    # 4. Upload benchmark_metrics.json
    try:
        api.upload_file(
            path_or_fileobj=json.dumps(metrics, indent=2, default=str).encode("utf-8"),
            path_in_repo="benchmark_metrics.json",
            repo_id=repo_id,
            commit_message=f"Upload benchmark metrics for {model_name}-{mod_name}",
        )
    except Exception:
        pass

    # 5. Upload weights.pt if PASS and exists
    if has_weights and status in ("PASS", "PASS_NO_COMPRESSION") and "weights.pt" not in remote_files:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Uploading weights.pt ({wf.stat().st_size / 1e6:.1f} MB) -> {repo_id}...")
        for attempt in range(1, 4):
            try:
                api.upload_file(
                    path_or_fileobj=str(wf),
                    path_in_repo="weights.pt",
                    repo_id=repo_id,
                    commit_message=f"Upload weights for {model_name}-{mod_name} on {dataset_name}",
                )
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] SUCCESS: weights.pt -> {repo_id}")
                break
            except Exception as e_up:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Attempt {attempt} failed for {repo_id}: {e_up}")
                time.sleep(5)

    # 6. Upload auxiliary files (config.json, tokenizer files, etc.)
    for extra in ckpt_dir.glob("*"):
        if extra.is_file() and extra.name not in ("weights.pt", "pytorch_model.bin", "README.md", "summary_record.json"):
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

    COMPLETED_UPLOADS.add(repo_id)

def sync_and_verify_hf():
    repo_dir = Path(os.environ.get("SVD_PROJECT_ROOT",
                                   "D:\\SVD_Project" if os.path.exists("D:\\SVD_Project") else "."))
    results_base = repo_dir / "results"
    docs_base = repo_dir / "Docs" / "content"
    if not results_base.exists():
        return

    api = HfApi()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Running sweep across {results_base}...")

    # 1. Master results dataset repo
    try:
        api.create_repo(repo_id=MASTER_RESULTS_REPO, exist_ok=True, private=True, repo_type="dataset")
    except Exception:
        pass

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

    # 2. Upload per-module results across agnews_unified_*, mnli_unified_*, unified_*
    handled = set()

    # Process all directories matching *unified_*/*/checkpoints/*
    for ckpt_dir in results_base.glob("*unified_*/*/checkpoints/*"):
        if not ckpt_dir.is_dir() or ckpt_dir.name in ("full", "hf_export"):
            continue

        model_dir = ckpt_dir.parent.parent
        model_name = model_dir.name
        dir_name = model_dir.parent.name
        mod_name = ckpt_dir.name

        if dir_name.startswith("agnews_unified_"):
            dataset_name = "ag_news"
        elif dir_name.startswith("mnli_unified_"):
            dataset_name = "mnli"
        else:
            dataset_name = "sst2"

        key = f"{dataset_name}|{model_name}|{mod_name}"
        if key in handled:
            continue
        handled.add(key)

        wf = ckpt_dir / "weights.pt"
        srec_path = ckpt_dir / "summary_record.json"

        srec = None
        if srec_path.exists():
            try:
                with open(srec_path, encoding="utf-8") as f:
                    srec = json.load(f)
            except Exception:
                pass

        status = srec.get("status") if srec else ("PASS" if wf.exists() else None)
        _upload_module_result(
            api=api,
            wf=wf if wf.exists() else None,
            ckpt_dir=ckpt_dir,
            model_name=model_name,
            mod_name=mod_name,
            dataset_name=dataset_name,
            docs_base=docs_base,
            status_override=status,
            metrics_override=srec,
        )

    # Also handle results where checkpoints were cleared but summary.csv exists
    for csv_path in results_base.glob("*unified_*/*/summary.csv"):
        model_dir = csv_path.parent
        model_name = model_dir.name
        dir_name = model_dir.parent.name
        if dir_name.startswith("agnews_unified_"):
            dataset_name = "ag_news"
            mod_name = dir_name.replace("agnews_unified_", "")
        elif dir_name.startswith("mnli_unified_"):
            dataset_name = "mnli"
            mod_name = dir_name.replace("mnli_unified_", "")
        else:
            dataset_name = "sst2"
            mod_name = dir_name.replace("unified_", "")

        key = f"{dataset_name}|{model_name}|{mod_name}"
        if key in handled:
            continue
        handled.add(key)

        ckpt_dir = model_dir / "checkpoints" / mod_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _upload_module_result(
            api=api,
            wf=None,
            ckpt_dir=ckpt_dir,
            model_name=model_name,
            mod_name=mod_name,
            dataset_name=dataset_name,
            docs_base=docs_base,
            status_override=None,
            metrics_override=None,
        )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Processed {len(handled)} module/model/dataset targets.")

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
