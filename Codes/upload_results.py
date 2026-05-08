import os
from huggingface_hub import HfApi, create_repo

# Set your token directly (or use environment variable)
HF_TOKEN = "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"  # replace with your actual token

repo_id = "tahamajs/attention-test-results"  # replace YOUR_USERNAME

# Create repo if it doesn't exist
create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

api = HfApi()
api.upload_file(
    # path_or_fileobj="test_results.csv",
    path_in_repo="./experiments_full_separate",
    repo_id=repo_id,
    repo_type="model",
    token=HF_TOKEN,
)
print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")