import os
import glob

def fix_md(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    changed = False
    
    if "## Usage Examples" not in content:
        content += "\n## Usage Examples\n\n```python\n# TODO: Add specific usage example\n```\n"
        changed = True
        
    if "## Benchmark Results" not in content:
        content += "\n## Benchmark Results\n\n"
        content += "| Status | Accuracy | Val Loss | Peak VRAM | Inference Latency |\n"
        content += "|--------|----------|----------|-----------|-------------------|\n"
        content += "| Pending| Pending  | Pending  | Pending   | Pending           |\n\n"
        changed = True
        
    if "## Limitations" not in content and "## Limitations & Future Work" not in content:
        content += "\n## Limitations & Future Work\n\nTODO: Identify specific limitations and edge cases for this module.\n"
        changed = True
        
    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

for md in glob.glob("Docs/content/attentions/*.md") + glob.glob("Docs/content/compressions/*.md"):
    fix_md(md)

print("Fixed existing markdown files.")
