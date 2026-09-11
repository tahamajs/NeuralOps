import sys
import os
import time
import csv
from pathlib import Path
from huggingface_hub import HfApi

HF_ORG = "Hooshaai"
RESULTS_DIR = Path("D:/SVD_Project/results")
PASS_THRESHOLD = 56.0
MODELS = ["distilbert", "roberta", "gpt2"]

def upload_to_hf(module, model_slug, status, accuracy):
    try:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        api = HfApi()
        repo_id = f"{HF_ORG}/svd-linear-attention-{model_slug}-{module}"
        try:
            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        except Exception:
            pass
        readme = f"---\ntags: [svd-linear-attention, {module}, pytorch]\ndatasets: [sst2]\n---\n# {module} / {model_slug}\nAccuracy: {accuracy:.2f}% | Status: {status}\n"
        api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id, commit_message=f"Update card acc={accuracy:.2f}%")
        unified = RESULTS_DIR / f"unified_{module}"
        if status == "PASS":
            candidate_weights = [
                unified / model_slug / "checkpoints" / module / "weights.pt",
                unified / model_slug / "checkpoints" / module / "hf_export" / "weights.pt",
                unified / model_slug / "hf_export" / "weights.pt",
                unified / model_slug / "weights.pt",
            ]
            weights_file = next((w for w in candidate_weights if w.exists()), None)
            if not weights_file:
                all_w = [w for w in (unified / model_slug).rglob("weights.pt") if "\\full\\" not in str(w) and "/full/" not in str(w)]
                if all_w:
                    weights_file = all_w[0]
            if weights_file:
                print(f"[HF-SYNC] Uploading weights -> {repo_id} ({weights_file.stat().st_size / 1e6:.1f} MB)")
                api.upload_file(path_or_fileobj=str(weights_file), path_in_repo="weights.pt", repo_id=repo_id, commit_message="Upload weights")
        csv_path = unified / model_slug / "summary.csv"
        if csv_path.exists():
            api.upload_file(path_or_fileobj=str(csv_path), path_in_repo="summary.csv", repo_id=repo_id, commit_message="Upload metrics")
        print(f"[HF-SYNC] Upload done: {repo_id}")
        return True
    except Exception as e:
        print(f"[HF-SYNC] Upload failed {module}/{model_slug}: {e}")
        return False

def sync_all():
    api = HfApi()
    try:
        remote_models = {m.id for m in api.list_models(author=HF_ORG)}
    except Exception as e:
        print(f"Error fetching remote repos: {e}")
        remote_models = set()
    
    print(f"Remote models in org: {len(remote_models)}")
    count = 0
    for udir in RESULTS_DIR.glob("unified_*"):
        module = udir.name.replace("unified_", "")
        for model in MODELS:
            csv_path = udir / model / "summary.csv"
            if not csv_path.exists():
                continue
            try:
                best_status, best_acc = "FAIL", 0.0
                with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
                    for row in csv.DictReader(f):
                        if (row.get("module_name") or "").strip() == "full":
                            continue
                        acc = float(row.get("accuracy_pct", row.get("mean_accuracy", 0)) or 0)
                        status = row.get("status", "FAIL")
                        if acc > best_acc:
                            best_acc, best_status = acc, status
                
                if best_acc >= PASS_THRESHOLD or best_status in ("PASS", "PASS_LOW"):
                    repo_id = f"{HF_ORG}/svd-linear-attention-{model}-{module}"
                    # Check if repo exists or has files
                    # Upload
                    print(f"Syncing {module}/{model} -> acc={best_acc:.2f}%, status={best_status}")
                    upload_to_hf(module, model, best_status, best_acc)
                    count += 1
            except Exception as e:
                print(f"Error on {module}/{model}: {e}")
    print(f"Total synced: {count}")

if __name__ == "__main__":
    sync_all()
