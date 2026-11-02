#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Convenience CLI redirector for paper LaTeX table generation.
Runs `academic/paper/scripts/generate_paper_tables.py` directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    target_script = repo_root / "academic" / "paper" / "scripts" / "generate_paper_tables.py"
    if not target_script.exists():
        print(f"Error: Table generator not found at {target_script}", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(target_script)] + sys.argv[1:]
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main())
