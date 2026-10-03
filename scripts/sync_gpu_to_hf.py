import os
import glob
from huggingface_hub import HfApi

api = HfApi()
hf_org = "Hooshaai"

remote_models = {m.id for m in api.list_models(author=hf_org)}
print(f"HF Server currently has {len(remote_models)} models.")

from pathlib import Path

root = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(__file__).resolve().parent.parent
results_dir = root / "results"

candidates = (
    glob.glob(str(results_dir / "**" / "weights.pt"), recursive=True) +
    glob.glob(str(results_dir / "**" / "finetuned_model.pt"), recursive=True)
)

print(f"Total localized weights on GPU server: {len(candidates)}")

uploaded = 0
for w in candidates:
    parts = w.replace("\\", "/").split("/")
    arch = None
    for a in ["distilbert", "roberta", "gpt2", "qwen3_5", "qwen"]:
        if a in parts:
            arch = a
            break
    if not arch:
        continue
    arch_idx = [i for i, p in enumerate(parts) if arch in p][-1]
    mod_raw = parts[arch_idx - 1]
    mod = mod_raw.replace("unified_", "").replace("_attention", "").replace("_compressor", "").replace("_", "-").lower()
    if mod in ["results", "test-hgrn2", "roberta-sst2-lora-baseline"]:
        continue
    repo_id = f"{hf_org}/svd-linear-attention-{arch}-{mod}"
    
    try:
        if repo_id in remote_models:
            files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            if "weights.pt" in files or "pytorch_model.bin" in files or "model.safetensors" in files:
                continue
    except Exception:
        pass
    
    try:
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=w,
            path_in_repo="weights.pt",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"feat: upload validated checkpoint weights for {arch}-{mod}"
        )
        uploaded += 1
        print(f"Uploaded weights to {repo_id}")
    except Exception as e:
        print(f"Upload failed for {repo_id}: {e}")

print(f"Direct GPU-to-HF sync complete: {uploaded} checkpoints uploaded.")
