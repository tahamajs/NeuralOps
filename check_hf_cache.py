import os
from pathlib import Path

hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub/models--textattack--roberta-base-SST-2/snapshots"))
if hf_cache.exists():
    snapshots = list(hf_cache.iterdir())
    print("Found snapshots:", snapshots)
    if snapshots:
        latest = snapshots[0]
        print("Snapshot contents:")
        for f in latest.iterdir():
            print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
else:
    print("HF cache path not found:", hf_cache)
