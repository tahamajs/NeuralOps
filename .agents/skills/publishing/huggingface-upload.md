# Hugging Face Upload

## Purpose

Upload validated checkpoints to the HF Hub.

## Naming convention

```
<HF_ORG>/svd-linear-attention-<backbone>-<module>
```

## Files to upload

- `weights.pt`
- `config.json`
- `README.md` (model card)
- `metrics.json`
- `tokenizer.json` (if applicable)

## Command

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
api.upload_folder(folder_path=export_dir, repo_id=repo_id)
```

## Output

- `papers/modules/<module>/hf_hub.txt` with the repo URL
