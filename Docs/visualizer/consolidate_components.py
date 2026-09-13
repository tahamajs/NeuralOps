#!/usr/bin/env python3
"""
consolidate_components.py
=========================
Consolidates all 241 individual component JSON files under Docs/visualizer/components/
into the unified Docs/visualizer/modules_components.json and rebuilds standalone.html.
"""

import os
import glob
import json
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE_DIR)

COMP_DIR = os.path.join(BASE_DIR, "Docs", "visualizer", "components")
OUT_FILE = os.path.join(BASE_DIR, "Docs", "visualizer", "modules_components.json")

def consolidate():
    files = sorted(glob.glob(os.path.join(COMP_DIR, "*.json")))
    print(f"Found {len(files)} component files in {COMP_DIR}")

    modules = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                mod = json.load(f)
                modules.append(mod)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    payload = {
        "total": len(modules),
        "modules": modules
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] Successfully consolidated {len(modules)} modules into {OUT_FILE}")

    # Rebuild standalone.html
    build_script = os.path.join(BASE_DIR, "Docs", "visualizer", "build_standalone.py")
    subprocess.run(["python3", build_script], check=True)
    print("[OK] Rebuilt standalone.html")

if __name__ == "__main__":
    consolidate()
