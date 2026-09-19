import os, shutil
from pathlib import Path

src = Path(os.path.expanduser("~/.cache/huggingface/hub/models--textattack--roberta-base-SST-2/snapshots/84ee248b91053ef5d0c748bbac4edfba1cf89584"))
dst = Path(r"D:\svd_project\models\roberta_sst2")
dst.mkdir(parents=True, exist_ok=True)

print(f"Copying files from {src} to {dst}...")
for f in src.iterdir():
    target = dst / f.name
    print(f"Copying {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)...")
    shutil.copy2(f, target)

print("Done copying! Model directory contents:")
for f in dst.iterdir():
    print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
