import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
results_dir = Path(r"D:\svd_project\results")

print("Fetching existing repositories for author 'Hooshaai'...", flush=True)
try:
    existing_repos = set(m.id for m in api.list_models(author="Hooshaai"))
    print(f"Total existing repos found: {len(existing_repos)}", flush=True)
except Exception as e:
    print(f"Warning fetching repos list: {e}", flush=True)
    existing_repos = set()

# Collect candidate directories with weights
export_dirs = []
for pattern in [
    "unified_*/distilbert/checkpoints/*/hf_export",
    "unified_*/roberta/checkpoints/*/hf_export",
    "unified_*/gpt2/checkpoints/*/hf_export",
    "unified_sweep/*/checkpoints/*/hf_export",
]:
    for p in results_dir.glob(pattern):
        weights = list(p.glob("*.bin")) + list(p.glob("*.safetensors")) + list(p.glob("*.pt"))
        if weights:
            export_dirs.append(p)

print(f"Discovered {len(export_dirs)} checkpoint directories with model weights.", flush=True)

to_upload = []
for p in export_dirs:
    parts = p.parts
    if "unified_sweep" in parts:
        model_name = parts[4]
        mod_name = parts[6]
    else:
        mod_name = parts[3].replace("unified_", "")
        model_name = parts[4]
    repo_id = f"Hooshaai/svd-linear-attention-{model_name}-{mod_name}"
    
    # Only upload if repo is not already created / uploaded
    if repo_id not in existing_repos:
        to_upload.append((p, repo_id, model_name, mod_name))

print(f"Found {len(to_upload)} unuploaded models out of {len(export_dirs)} total.", flush=True)

success_count = 0
fail_count = 0

for p, repo_id, model_name, mod_name in to_upload:
    try:
        print(f"[{success_count + fail_count + 1}/{len(to_upload)}] Uploading {repo_id} from {p}...", flush=True)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(p),
            repo_id=repo_id,
            commit_message=f"Upload model weights for {model_name} {mod_name}",
        )
        print(f"[OK] Successfully uploaded {repo_id}", flush=True)
        existing_repos.add(repo_id)
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Error uploading {repo_id}: {e}", flush=True)
        fail_count += 1

print(f"\nUpload run complete: {success_count} succeeded, {fail_count} failed.", flush=True)
