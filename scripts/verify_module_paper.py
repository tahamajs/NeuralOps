"""verify_module_paper.py — Verifies paper completeness against all publication gates.

Usage:
  python3 scripts/verify_module_paper.py --module <mod_name> [--strict]
"""
import argparse
import os
import re
import sys
from pathlib import Path
import pypdf

ROOT = Path("/Users/tahamajs/Documents/uni/Project")
PAPER_BASE = ROOT / "paper/PerModulePaper"

CANONICAL_MD_ORDER = [
    "00_meta.md", "01_intro.md", "02_related.md", "03_background.md",
    "04_method.md", "05_architecture.md", "06_setup.md", "07_results.md",
    "08_analysis.md", "09_discussion.md", "10_limitations.md", "11_conclusion.md",
    "12_reproducibility.md", "13_references.md", "A_math.md", "B_implementation.md",
    "C_data.md", "D_negative_results.md", "E_ethics.md"
]

CANONICAL_TEX_ORDER = [
    "00_meta.tex", "01_intro.tex", "02_related.tex", "03_background.tex",
    "04_method.tex", "05_architecture.tex", "06_setup.tex", "07_results.tex",
    "08_analysis.tex", "09_discussion.tex", "10_limitations.tex", "11_conclusion.tex",
    "12_reproducibility.tex", "13_references.tex", "A_math.tex", "B_implementation.tex",
    "C_data.tex", "D_negative_results.tex", "E_ethics.tex"
]

def verify_module(mod_name: str, strict: bool = False) -> dict:
    mod_dir = PAPER_BASE / mod_name
    res = {
        "module": mod_name,
        "valid": True,
        "errors": [],
        "warnings": [],
        "pages": 0,
        "pdf_size_kb": 0,
        "word_count": 0,
        "equations_count": 0,
        "tables_count": 0,
        "has_ascii": False,
        "has_citations": False
    }

    if not mod_dir.exists():
        res["valid"] = False
        res["errors"].append(f"Directory {mod_dir} does not exist")
        return res

    # 1. Check all 19 MD files
    for mf in CANONICAL_MD_ORDER:
        p = mod_dir / mf
        if not p.exists():
            res["valid"] = False
            res["errors"].append(f"Missing Markdown file {mf}")

    # 2. Check all 19 TEX files
    for tf in CANONICAL_TEX_ORDER:
        p = mod_dir / tf
        if not p.exists():
            res["valid"] = False
            res["errors"].append(f"Missing TeX file {tf}")

    # 3. Check main.pdf
    pdf_path = mod_dir / "main.pdf"
    if not pdf_path.exists():
        res["valid"] = False
        res["errors"].append("Missing main.pdf")
    else:
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            res["pages"] = len(reader.pages)
            res["pdf_size_kb"] = round(pdf_path.stat().st_size / 1024, 1)
            target_pages = 15 if strict else 10
            if res["pages"] < target_pages:
                res["valid"] = False
                res["errors"].append(f"PDF page count {res['pages']} < required {target_pages}")
            if res["pdf_size_kb"] < 400:
                res["warnings"].append(f"PDF size {res['pdf_size_kb']} KB is below 400 KB")
        except Exception as e:
            res["valid"] = False
            res["errors"].append(f"Error reading PDF: {e}")

    # 4. Word count across Markdown files
    total_words = 0
    for mf in CANONICAL_MD_ORDER:
        p = mod_dir / mf
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="ignore")
            # Filter comments & markdown formatting roughly
            clean = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
            words = clean.split()
            total_words += len(words)
    res["word_count"] = total_words
    min_words = 6000 if strict else 3000
    if total_words < min_words:
        res["warnings"].append(f"Total word count {total_words} < {min_words}")

    # 5. Equation count in 04_method.md
    method_file = mod_dir / "04_method.md"
    if method_file.exists():
        m_text = method_file.read_text(encoding="utf-8", errors="ignore")
        # count $$ or \begin{equation} or Equation
        eqs = len(re.findall(r"\$\$|\\begin\{equation\}|### Equation \(\d+\)", m_text))
        # Each $$ block has start and end, so count / 2 if $$
        disp_eqs = len(re.findall(r"### Equation \(\d+\)", m_text))
        if disp_eqs == 0:
            disp_eqs = len(re.findall(r"\$\$", m_text)) // 2
        res["equations_count"] = max(disp_eqs, len(re.findall(r"\\begin\{equation\}", m_text)))
        if res["equations_count"] < 10:
            res["warnings"].append(f"Method equations count {res['equations_count']} < 10")

    # 6. Diagram presence in 05_architecture.md
    arch_file = mod_dir / "05_architecture.md"
    if arch_file.exists():
        a_text = arch_file.read_text(encoding="utf-8", errors="ignore")
        if "┌" in a_text or "+--" in a_text or "|  Input" in a_text or "+---" in a_text:
            res["has_ascii"] = True
        else:
            res["warnings"].append("No ASCII architectural diagram found in 05_architecture.md")

    # 7. Citations & Tables in 07_results.md
    res_file = mod_dir / "07_results.md"
    if res_file.exists():
        r_text = res_file.read_text(encoding="utf-8", errors="ignore")
        cites = len(re.findall(r"<!--\s*source:.*?line\s*\d+\s*-->", r_text))
        if cites > 0:
            res["has_citations"] = True
        else:
            res["valid"] = False
            res["errors"].append("Missing <!-- source: ... line ... --> citations in 07_results.md")

        tables = len(re.findall(r"### Table \d+\.\d+", r_text))
        res["tables_count"] = tables
        if tables < 3:
            res["warnings"].append(f"Results table count {tables} < 3")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="Module name to verify")
    parser.add_argument("--strict", action="store_true", help="Enforce >=15 pages")
    args = parser.parse_args()

    r = verify_module(args.module, strict=args.strict)
    print("=" * 60)
    print(f"VERIFICATION REPORT: {r['module']}")
    print("=" * 60)
    print(f"Status:        {'PASS' if r['valid'] else 'FAIL'}")
    print(f"Pages:         {r['pages']}")
    print(f"PDF Size:      {r['pdf_size_kb']} KB")
    print(f"Words:         {r['word_count']}")
    print(f"Equations:     {r['equations_count']}")
    print(f"Tables:        {r['tables_count']}")
    print(f"ASCII Diagram: {r['has_ascii']}")
    print(f"Citations:     {r['has_citations']}")
    if r["errors"]:
        print("\nERRORS:")
        for e in r["errors"]:
            print(f"  ❌ {e}")
    if r["warnings"]:
        print("\nWARNINGS:")
        for w in r["warnings"]:
            print(f"  ⚠️  {w}")
    print("=" * 60)
    sys.exit(0 if r["valid"] else 1)
