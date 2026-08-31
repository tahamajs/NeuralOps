from huggingface_hub import snapshot_download
from pathlib import Path
p=Path("D:/SVD_PROJECT/offline_hf/mirror_full")
p.mkdir(parents=True, exist_ok=True)
print("MODEL", flush=True)
snapshot_download("google/flan-t5-small", local_dir=str(p/"t5-small"))
print("DONE_MODEL", flush=True)
