import os
import sys
import time
import socket
import csv
from pathlib import Path
from huggingface_hub import HfApi

# Set global socket timeout to prevent indefinite hangs
socket.setdefaulttimeout(60)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

HF_ORG = "Hooshaai"
api = HfApi()

print("Fetching existing Hugging Face repositories in Hooshaai...", flush=True)
hf_models = set(m.id for m in api.list_models(author=HF_ORG))
print(f"Current repository count: {len(hf_models)}", flush=True)

results_dir = Path("results")
tasks = []

for unified_dir in sorted(results_dir.glob("unified_*")):
    if not unified_dir.is_dir():
        continue
    mod = unified_dir.name.replace("unified_", "")
    for model_dir in unified_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        csv_path = model_dir / "summary.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                best_acc = 0.0
                best_status = "FAIL"
                best_row = {}
                for row in reader:
                    m_name = (row.get("module_name") or "").strip()
                    if m_name == "full":
                        continue
                    try:
                        acc = float(row.get("accuracy_pct", row.get("mean_accuracy", 0)) or 0)
                        status = row.get("status", "FAIL")
                        if acc > best_acc:
                            best_acc, best_status = acc, status
                            best_row = row
                    except Exception:
                        pass
                if best_status == "PASS":
                    repo_id = f"{HF_ORG}/svd-linear-attention-{model_name}-{mod}"
                    if repo_id not in hf_models:
                        tasks.append({
                            "module": mod,
                            "model": model_name,
                            "acc": best_acc,
                            "status": best_status,
                            "repo_id": repo_id,
                            "model_dir": model_dir,
                            "row": best_row
                        })
        except Exception as e:
            print(f"Error reading {csv_path}: {e}", flush=True)

print(f"Found {len(tasks)} models to upload.", flush=True)

uploaded = 0
failed = 0

for i, t in enumerate(tasks, 1):
    repo_id = t["repo_id"]
    mod = t["module"]
    model_name = t["model"]
    acc = t["acc"]
    status = t["status"]
    model_dir = t["model_dir"]

    print(f"[{i}/{len(tasks)}] Uploading {repo_id} (Acc: {acc:.2f}%)...", flush=True)

    try:
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        
        # Build comprehensive README
        f1 = t["row"].get("f1_score", "N/A")
        comp = t["row"].get("compression_rate", "N/A")
        vram = t["row"].get("peak_vram_mb", "N/A")

        readme_content = f"""---
tags:
- svd-linear-attention
- {mod}
- {model_name}
- pytorch
datasets:
- sst2
license: apache-2.0
---

# SVD Linear Attention: {mod.upper()} / {model_name}

This repository contains the benchmark artifacts and evaluation metrics for **{mod}** evaluated on **{model_name}** for the GLUE SST-2 sentiment classification task.

## Benchmark Performance (GLUE SST-2)
- **Model Family:** {model_name}
- **Module / Method:** {mod}
- **Validation Accuracy:** {acc:.2f}%
- **Status:** {status}
- **F1 Score:** {f1}
- **Compression Rate:** {comp}
- **Peak VRAM:** {vram}
"""
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=f"Upload model card (acc={acc:.2f}%)"
        )

        # Upload summary.csv
        csv_file = model_dir / "summary.csv"
        if csv_file.exists():
            api.upload_file(
                path_or_fileobj=str(csv_file),
                path_in_repo="summary.csv",
                repo_id=repo_id,
                commit_message="Upload summary metrics"
            )

        # Look for weights.pt
        candidate_weights = [
            model_dir / "checkpoints" / mod / "weights.pt",
            model_dir / "checkpoints" / mod / "hf_export" / "weights.pt",
            model_dir / "hf_export" / "weights.pt",
            model_dir / "weights.pt",
            model_dir / "finetuned_model.pt",
        ]
        weights_file = next((w for w in candidate_weights if w.exists()), None)
        if not weights_file:
            all_w = [w for w in model_dir.rglob("weights.pt") if "\\full\\" not in str(w) and "/full/" not in str(w)]
            if all_w:
                weights_file = all_w[0]

        if weights_file and weights_file.exists():
            print(f"  Uploading weights: {weights_file.name} ({weights_file.stat().st_size / 1024 / 1024:.2f} MB)...", flush=True)
            api.upload_file(
                path_or_fileobj=str(weights_file),
                path_in_repo="weights.pt",
                repo_id=repo_id,
                commit_message="Upload model weights"
            )

        # Look for extra json files (e.g. metrics, config, tokenizer)
        for jf in model_dir.glob("*.json"):
            if "finetune" not in jf.name:
                try:
                    api.upload_file(
                        path_or_fileobj=str(jf),
                        path_in_repo=jf.name,
                        repo_id=repo_id,
                        commit_message=f"Upload {jf.name}"
                    )
                except Exception:
                    pass

        print(f"  ✓ Successfully uploaded {repo_id}", flush=True)
        uploaded += 1
    except Exception as e:
        print(f"  ✗ Failed to upload {repo_id}: {e}", flush=True)
        failed += 1

print("=" * 60, flush=True)
print(f"FINISHED BACKFILL UPLOAD: {uploaded} succeeded, {failed} failed.", flush=True)
