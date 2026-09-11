#!/usr/bin/env python3
"""
fix_runner_indentation.py
Properly formats the skip check in all run_all_4_*.py runners to match exact indentation.
"""

import glob
from pathlib import Path
import ast

bad_block = '''if os.environ.get("SKIP_HF_UPLOAD") or os.environ.get("SKIP_STAGE2"):
    print("[RUNNER] Skipping runner-level upload (handled asynchronously by background orchestrator).", flush=True)
    sys.exit(0)

'''

runners = sorted(glob.glob("run_all_4_*.py"))
fixed = 0

for r in runners:
    path = Path(r)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    
    # First, strip out any badly inserted skip blocks
    new_lines = []
    skip_mode = False
    for line in lines:
        if 'if os.environ.get("SKIP_HF_UPLOAD") or os.environ.get("SKIP_STAGE2"):' in line:
            skip_mode = True
            continue
        if skip_mode:
            if 'sys.exit(0)' in line:
                skip_mode = False
            continue
        new_lines.append(line)
    
    content = "".join(new_lines)
    
    # Now find where "from huggingface_hub import" or "import huggingface_hub" is
    final_lines = []
    for line in content.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("from huggingface_hub import") or stripped.startswith("import huggingface_hub"):
            indent = line[:len(line) - len(stripped)]
            skip_code = (
                f"{indent}if os.environ.get('SKIP_HF_UPLOAD') or os.environ.get('SKIP_STAGE2'):\n"
                f"{indent}    print('[RUNNER] Skipping runner upload (handled in background).', flush=True)\n"
                f"{indent}    sys.exit(0)\n"
            )
            final_lines.append(skip_code)
        final_lines.append(line)
    
    result = "".join(final_lines)
    try:
        ast.parse(result, filename=r)
        path.write_text(result, encoding="utf-8")
        fixed += 1
    except SyntaxError as se:
        print(f"ERROR fixing {r}: {se}")

print(f"Fixed and validated AST for {fixed} runners.")
