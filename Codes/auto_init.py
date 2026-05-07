#!/usr/bin/env python3
"""
Replace all occurrences of AUX_REGISTRY with ATTENTION_REGISTRY
in all Python files under framework/.
Also updates imports.
"""

import re
from pathlib import Path

def process_file(file_path, dry_run=False):
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"❌ Cannot read {file_path}: {e}")
        return

    # Replace AUX_REGISTRY -> ATTENTION_REGISTRY
    new_content = content.replace("AUX_REGISTRY", "ATTENTION_REGISTRY")
    
    if new_content == content:
        return  # No change

    if dry_run:
        print(f"[DRY RUN] Would update {file_path}")
    else:
        file_path.write_text(new_content, encoding='utf-8')
        print(f"✅ Updated {file_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview changes")
    parser.add_argument("--dir", default="framework", help="Root directory to scan")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"❌ {root} not found")
        return

    py_files = list(root.rglob("*.py"))
    print(f"Found {len(py_files)} Python files")
    for py_file in py_files:
        process_file(py_file, dry_run=args.dry_run)

if __name__ == "__main__":
    main()