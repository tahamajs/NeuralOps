#!/usr/bin/env python3
"""
patch_runners_skip_upload.py
Inserts an early exit check before huggingface_hub imports in all run_all_4_*.py runners:
if os.environ.get("SKIP_HF_UPLOAD") or os.environ.get("SKIP_STAGE2"):
    print("[RUNNER] Skipping runner upload (handled asynchronously by orchestrator).", flush=True)
    sys.exit(0)
"""

import glob
from pathlib import Path

SKIP_SNIPPET = '''if os.environ.get("SKIP_HF_UPLOAD") or os.environ.get("SKIP_STAGE2"):
    print("[RUNNER] Skipping runner-level upload (handled asynchronously by background orchestrator).", flush=True)
    sys.exit(0)

'''

runners = sorted(glob.glob("run_all_4_*.py"))
patched_count = 0

for r in runners:
    path = Path(r)
    content = path.read_text(encoding="utf-8")
    if 'os.environ.get("SKIP_HF_UPLOAD")' in content:
        continue
    
    # Target 1: "from huggingface_hub import HfApi"
    target1 = "from huggingface_hub import HfApi"
    target2 = "import huggingface_hub"
    
    if target1 in content:
        new_content = content.replace(target1, SKIP_SNIPPET + target1, 1)
        path.write_text(new_content, encoding="utf-8")
        patched_count += 1
    elif target2 in content:
        new_content = content.replace(target2, SKIP_SNIPPET + target2, 1)
        path.write_text(new_content, encoding="utf-8")
        patched_count += 1

print(f"Patched {patched_count} runner files to exit early on SKIP_HF_UPLOAD.")
