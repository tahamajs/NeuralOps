import sys, re
from pathlib import Path

content = Path("Codes/framework/run_quality_sweep.py").read_text(encoding="utf-8")

# 1. We need to import ModelStateSnapshot
if "ModelStateSnapshot" not in content:
    content = content.replace("from framework.core.registry import", "from framework.core.snapshot_restore import ModelStateSnapshot\nfrom framework.core.registry import")

# 2. Modify `run_single_module_main` to use ModelStateSnapshot instead of copy.deepcopy.
# Note: we need to handle timing logs too!
# Let's write a targeted script to do this refactor, or we can just replace the whole loop.
