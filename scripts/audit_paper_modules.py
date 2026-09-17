"""audit_paper_modules.py — Classify PerModulePaper modules."""
import json
from pathlib import Path

PAPER_DIR = Path("paper/PerModulePaper")
TEMPLATE_DIR = PAPER_DIR / "_TEMPLATE"

def classify(module_dir: Path) -> str:
    if module_dir.name.startswith("_"):
        return "skip"
    
    md_files = list(module_dir.glob("*.md"))
    tex_files = list(module_dir.glob("*.tex"))
    all_files = md_files + tex_files
    if not all_files:
        return "empty"
    
    # Check if module intro is a stub
    intro_tex = module_dir / "01_intro.tex"
    intro_md = module_dir / "01_intro.md"
    is_stub = False
    if intro_tex.exists():
        t = intro_tex.read_text(encoding="utf-8", errors="ignore")
        if "Formal architectural specifications for" in t:
            is_stub = True
    elif intro_md.exists():
        t = intro_md.read_text(encoding="utf-8", errors="ignore")
        if "Formal architectural specifications for" in t:
            is_stub = True
    
    if is_stub:
        return "stub"
    
    # Check for template variables
    has_template_var = False
    for f in all_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "{{" in text and "}}" in text:
            has_template_var = True
            break
            
    if has_template_var:
        return "template_var"
    
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
    
    # Clean up empty categories
    summary = {k: len(v) for k, v in results.items() if len(v) > 0}
    print(json.dumps(summary, indent=2))
    with open("audit/paper_audit.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    Path("audit").mkdir(exist_ok=True)
    main()
