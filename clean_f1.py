from huggingface_hub import HfApi, hf_hub_download
import re
import traceback

api = HfApi()
repos = list(api.list_models(author='Hooshaai'))
deleted = []

for r in repos:
    if 'gpt2' not in r.modelId.lower():
        continue
    try:
        # fetch readme
        readme_path = hf_hub_download(repo_id=r.modelId, filename="README.md", repo_type="model")
        with open(readme_path, "r", encoding="utf-8") as f:
            readme = f.read()
            
        # check if it contains F1: 0.0 or F1: 0.00
        # e.g. **F1:** 0.0 or F1: 0.0000
        if re.search(r'\*\*F1:\*\* 0\.0', readme) or re.search(r'\*\*F1 Score:\*\* 0\.0', readme) or re.search(r'F1.*0\.0\b', readme):
            print(f"Deleting {r.modelId} because F1 == 0.0")
            api.delete_repo(r.modelId, repo_type='model')
            deleted.append(r.modelId)
    except Exception as e:
        print(f"Skipping {r.modelId}: {e}")

print(f"\nDeleted F1=0.0 repos: {deleted}")
