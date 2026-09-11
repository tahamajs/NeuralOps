import os
import sys
import time
import csv
from pathlib import Path
from huggingface_hub import HfApi

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

HF_ORG = "Hooshaai"
api = HfApi()

print("Scanning results directory for PASS models...", flush=True)
results_dir = Path("results")
uploaded = 0
failed = 0

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
                for row in reader:
                    m_name = (row.get("module_name") or "").strip()
                    if m_name == "full":
                        continue
                    try:
                        acc = float(row.get("accuracy_pct", row.get("mean_accuracy", 0)) or 0)
                        status = row.get("status", "FAIL")
                        if acc > best_acc:
                            best_acc, best_status = acc, status
                    except Exception:
                        pass

                if best_status == "PASS" and best_acc >= 56.0:
                    repo_id = f"{HF_ORG}/svd-linear-attention-{model_name}-{mod}"
                    # Check if weights.pt exists locally
                    candidate_weights = [
                        model_dir / "checkpoints" / mod / "hf_export" / "weights.pt",
                        model_dir / "checkpoints" / mod / "weights.pt",
                        model_dir / "hf_export" / "weights.pt",
                        model_dir / "weights.pt",
                        model_dir / "finetuned_model.pt",
                    ]
                    weights_file = next((w for w in candidate_weights if w.exists()), None)
                    if not weights_file:
                        all_w = [w for w in model_dir.rglob("weights.pt") if "\\full\\" not in str(w) and "/full/" not in str(w)]
                        if all_w:
                            weights_file = all_w[0]

                    # Check if already has weights in repo
                    try:
                        files = set(api.list_repo_files(repo_id))
                        if "weights.pt" in files and "README.md" in files:
                            continue
                    except Exception:
                        # Repo doesn't exist yet, create it
                        try:
                            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
                            time.sleep(1)
                        except Exception as e:
                            print(f"Failed to create {repo_id}: {e}", flush=True)
                            continue

                    print(f"Uploading {repo_id} (Acc: {best_acc:.2f}%)...", flush=True)
                    # Upload README
                    readme = f"---\ntags: [svd-linear-attention, {mod}, pytorch]\ndatasets: [sst2]\n---\n# {mod} / {model_name}\nAccuracy: {best_acc:.2f}% | Status: PASS\nEvaluated on GLUE SST-2 benchmark.\n"
                    api.upload_file(
                        path_or_fileobj=readme.encode("utf-8"),
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        commit_message=f"Model card acc={best_acc:.2f}%"
                    )

                    # Upload summary.csv
                    if csv_path.exists():
                        try:
                            api.upload_file(
                                path_or_fileobj=str(csv_path),
                                path_in_repo="summary.csv",
                                repo_id=repo_id,
                                commit_message="Upload summary metrics"
                            )
                        except Exception:
                            pass

                    # Upload weights
                    if weights_file and weights_file.exists():
                        print(f"  -> Uploading weights ({weights_file.stat().st_size / 1024 / 1024:.1f} MB)...", flush=True)
                        api.upload_file(
                            path_or_fileobj=str(weights_file),
                            path_in_repo="weights.pt",
                            repo_id=repo_id,
                            commit_message="Upload model weights"
                        )

                    # Upload json configs
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

                    print(f"  [DONE] Successfully uploaded {repo_id}", flush=True)
                    uploaded += 1
                    time.sleep(2)
        except Exception as e:
            print(f"Error processing {model_dir}: {e}", flush=True)
            failed += 1

print(f"DIRECT UPLOAD COMPLETE: {uploaded} newly uploaded, {failed} failed.", flush=True)
