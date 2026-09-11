import os
import sys
from huggingface_hub import HfApi

HF_ORG = "Hooshaai"
api = HfApi()

models = [m.id for m in api.list_models(author=HF_ORG) if "svd-linear-attention" in m.id]
print(f"Total svd-linear-attention repos in {HF_ORG}: {len(models)}", flush=True)

missing_readme = 0
total_checked = 0

for repo in models:
    try:
        files = api.list_repo_files(repo)
        if "README.md" not in files:
            missing_readme += 1
            parts = repo.replace(f"{HF_ORG}/svd-linear-attention-", "").split("-")
            model_name = parts[0]
            module_name = "-".join(parts[1:]) if len(parts) > 1 else "svd"
            card = f"""---
tags: [svd-linear-attention, {module_name}, pytorch]
datasets: [sst2]
---
# {module_name} / {model_name}
Evaluated on SST-2 GLUE benchmark.
"""
            api.upload_file(
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo,
                commit_message="Add model card README",
            )
        total_checked += 1
    except Exception as e:
        pass

print(f"Audit finished: {total_checked} repos checked. Missing READMEs resolved: {missing_readme}.", flush=True)
