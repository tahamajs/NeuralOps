"""audit_paper_modules.py — Classify PerModulePaper modules."""
import json
import re
from pathlib import Path
import pypdf

PAPER_DIR = Path("paper/PerModulePaper")
VAR_REGEX = re.compile(r"\{\{[A-Z0-9_]{3,}\}\}")

def classify(module_dir: Path) -> str:
    if module_dir.name.startswith("_"):
        return "skip"
    
    md_files = [f for f in module_dir.glob("*.md") if f.is_file()]
    tex_files = [f for f in module_dir.glob("*.tex") if f.is_file()]
    all_files = md_files + tex_files
    if not all_files:
        return "empty"
    
    # Check for unresolved template variables like {{MODULE_NAME}}
    for f in all_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if VAR_REGEX.search(text):
            return "template_var"
            
    pdf = module_dir / "main.pdf"
    if not pdf.exists():
        return "unknown"
        
    try:
        reader = pypdf.PdfReader(str(pdf))
        num_pages = len(reader.pages)
    except Exception:
        return "unknown"
        
    # Papers with <= 5 pages or stub phrase are thin skeleton stubs
    if num_pages <= 5:
        return "stub"
        
    return "complete"

def main():
    results = {"stub": [], "template_var": [], "complete": [], "unknown": [], "empty": []}
    for d in sorted(PAPER_DIR.iterdir()):
        if not d.is_dir():
            continue
        cls = classify(d)
        if cls == "skip":
            continue
        results.setdefault(cls, []).append(d.name)
    
    summary = {k: len(v) for k, v in results.items()}
    print(json.dumps(summary, indent=2))
    with open("audit/paper_audit.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    Path("audit").mkdir(exist_ok=True)
    main()
