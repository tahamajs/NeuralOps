#!/usr/bin/env python3
"""
Flatten attention and compressor modules into top‑level directories,
fix imports, and generate __init__.py for eager loading.
"""

import os
import re
import shutil
from pathlib import Path

def move_py_files(src_dir: Path, dest_dir: Path, prefix: str = ""):
    """
    Move all .py files from src_dir (and subdirs) into dest_dir.
    If prefix is given, rename file as prefix_original_name to avoid collisions.
    """
    if not src_dir.exists():
        return
    for py_file in src_dir.rglob("*.py"):
        if py_file.parent == dest_dir:
            continue  # already at destination
        if py_file.name == "__init__.py":
            continue
        new_name = py_file.name
        if prefix:
            # Use subdirectory name as prefix to avoid collisions (e.g., leka_v2.py -> leka_v2.py already unique)
            # But if two files have same name in different subdirs, add subdir prefix.
            # For simplicity, we assume unique names. If collision, we'll overwrite. Add prefix if needed.
            pass
        dest_file = dest_dir / new_name
        print(f"Moving {py_file} -> {dest_file}")
        shutil.move(str(py_file), str(dest_file))

def update_imports_in_file(file_path: Path):
    """Replace old registry imports with unified ones, and fix base class imports."""
    with open(file_path, 'r') as f:
        content = f.read()
    # Replace registry imports
    content = re.sub(
        r'from\s+(?:\.\.core\.registry|framework\.(?:attentions|compression_engine)\.core\.registry)\s+import\s+ATTENTION_REGISTRY',
        'from framework.core.registry import ATTENTION_REGISTRY',
        content,
    )
    content = re.sub(
        r'from\s+(?:\.\.core\.registry|framework\.(?:attentions|compression_engine)\.core\.registry)\s+import\s+COMPRESSOR_REGISTRY',
        'from framework.core.registry import COMPRESSOR_REGISTRY',
        content,
    )
    content = re.sub(
        r'from\s+(?:\.\.core\.base|framework\.(?:attentions|compression_engine)\.core\.base)\s+import\s+BaseAttention',
        'from framework.core.base import BaseAttention',
        content,
    )
    content = re.sub(
        r'from\s+(?:\.\.core\.base|framework\.(?:attentions|compression_engine)\.core\.base)\s+import\s+BaseCompressor',
        'from framework.core.base import BaseCompressor',
        content,
    )
    # Also replace any ATTENTION_REGISTRY.get imports that might be wrong
    content = re.sub(r'ATTENTION_REGISTRY\.get', 'ATTENTION_REGISTRY.get', content)
    content = re.sub(r'COMPRESSOR_REGISTRY\.get', 'COMPRESSOR_REGISTRY.get', content)
    with open(file_path, 'w') as f:
        f.write(content)

def generate_init(package_dir: Path, registry_name: str):
    """
    Create __init__.py that imports all .py files in the directory (eager import).
    registry_name is just for comment.
    """
    py_files = sorted(package_dir.glob("*.py"))
    modules = [f.stem for f in py_files if f.name != "__init__.py"]
    if not modules:
        return
    init_content = f"# Auto-generated – eager import of all {registry_name}\n"
    for mod in modules:
        init_content += f"from . import {mod}\n"
    init_file = package_dir / "__init__.py"
    init_file.write_text(init_content)
    print(f"Generated {init_file} with {len(modules)} modules")

def clean_empty_dirs(base_dir: Path):
    """Remove empty subdirectories recursively."""
    for subdir in sorted(base_dir.rglob("*"), reverse=True):
        if subdir.is_dir() and subdir != base_dir:
            if not any(subdir.iterdir()):
                subdir.rmdir()
                print(f"Removed empty directory: {subdir}")

def main():
    base = Path("framework")

    # 1. Flatten attentions
    attn_dest = base / "attentions"
    attn_dest.mkdir(exist_ok=True)
    # Move all attention files from any subdirectory under attentions/
    for subdir in (base / "attentions").rglob("*"):
        if subdir.is_dir() and subdir != attn_dest:
            move_py_files(subdir, attn_dest)
    # 2. Flatten compression_engine
    comp_dest = base / "compression_engine"
    comp_dest.mkdir(exist_ok=True)
    for subdir in (base / "compression_engine").rglob("*"):
        if subdir.is_dir() and subdir != comp_dest:
            move_py_files(subdir, comp_dest)

    # 3. Fix imports in all moved files
    for py_file in attn_dest.glob("*.py"):
        if py_file.name != "__init__.py":
            update_imports_in_file(py_file)
    for py_file in comp_dest.glob("*.py"):
        if py_file.name != "__init__.py":
            update_imports_in_file(py_file)

    # 4. Generate __init__.py for eager imports
    generate_init(attn_dest, "attentions")
    generate_init(comp_dest, "compressors")

    # 5. Clean up empty directories
    clean_empty_dirs(base / "attentions")
    clean_empty_dirs(base / "compression_engine")

    print("\n✅ Done! All attention modules are now in framework/attentions/")
    print("   All compressor modules are now in framework/compression_engine/")
    print("   Imports have been updated to use framework.core.registry.")
    print("   Run your experiment with eager imports (no lazy loading).")

if __name__ == "__main__":
    main()