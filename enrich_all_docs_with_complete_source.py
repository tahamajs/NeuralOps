import os
import re
import shutil
from pathlib import Path

BASE_DIR = Path("/Users/tahamajs/Documents/uni/Project")
DOCS_DIR = BASE_DIR / "Docs/content"
FRONTEND_DOCS_DIR = BASE_DIR / "Codes/frontend/public/Docs/content"

ATTN_CODE_DIR = BASE_DIR / "Codes/framework/attentions"
COMP_CODE_DIR = BASE_DIR / "Codes/framework/compression_engine"

# Map of all python files in attentions
attn_py_files = {p.stem.lower(): p for p in ATTN_CODE_DIR.glob("*.py")}
# Map of all python files in compressions
comp_py_files = {p.stem.lower(): p for p in COMP_CODE_DIR.glob("*.py")}

def find_matching_py(md_stem, py_map, fallback_dir):
    # 1. Exact match
    if md_stem in py_map:
        return py_map[md_stem]
    
    # 2. Try variations (removing _attention, _compressor, etc.)
    clean_stem = md_stem.replace("_attention", "").replace("_compressor", "").replace("_pruner", "").replace("_quantizer", "")
    for k, v in py_map.items():
        clean_k = k.replace("_attention", "").replace("_compressor", "").replace("_pruner", "").replace("_quantizer", "")
        if clean_stem == clean_k or clean_stem in k or k in clean_stem:
            return v
            
    # 3. Search directory for close match
    for p in fallback_dir.glob("*.py"):
        if clean_stem in p.stem.lower():
            return p
            
    return None

def update_md_file(md_path, py_path):
    if not py_path or not py_path.exists():
        return False
        
    md_text = md_path.read_text()
    py_code = py_path.read_text()
    
    py_filename = py_path.name
    code_section_header = f"\n\n---\n\n## Complete Source Code Implementation (`{py_filename}`)\n\n```python\n"
    
    # Check if a code section already exists
    if "## Complete Source Code Implementation" in md_text:
        # Replace existing code section
        parts = md_text.split("## Complete Source Code Implementation")
        new_md_text = parts[0].rstrip() + code_section_header + py_code.rstrip() + "\n```\n"
    elif "```python" in md_text and "from framework" in md_text:
        # Already has code block, replace the last code block or append if short
        new_md_text = md_text.rstrip() + code_section_header + py_code.rstrip() + "\n```\n"
    else:
        # Append section
        new_md_text = md_text.rstrip() + code_section_header + py_code.rstrip() + "\n```\n"
        
    md_path.write_text(new_md_text)
    return True

updated_attn = 0
for md_file in (DOCS_DIR / "attentions").glob("*.md"):
    py_file = find_matching_py(md_file.stem.lower(), attn_py_files, ATTN_CODE_DIR)
    if py_file and update_md_file(md_file, py_file):
        updated_attn += 1

updated_comp = 0
for md_file in (DOCS_DIR / "compressions").glob("*.md"):
    py_file = find_matching_py(md_file.stem.lower(), comp_py_files, COMP_CODE_DIR)
    if py_file and update_md_file(md_file, py_file):
        updated_comp += 1

print(f"Updated {updated_attn} attention markdown docs and {updated_comp} compression markdown docs.")

# Synchronize all Docs/content to Codes/frontend/public/Docs/content
if FRONTEND_DOCS_DIR.exists():
    shutil.rmtree(FRONTEND_DOCS_DIR)
shutil.copytree(DOCS_DIR, FRONTEND_DOCS_DIR)

print("Synchronized all documentation files to frontend public directory.")
