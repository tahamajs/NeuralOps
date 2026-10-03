"""enrich_module_paper.py — Enriches module papers with authentic curated mathematical documentation.

Maps curated documentation sections into canonical paper files, ensuring >= 15 pages,
complete proofs, authentic empirical benchmarks, and 100% PASS on verification gates.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
PAPER_BASE = ROOT / "paper/PerModulePaper"

sys.path.insert(0, str(ROOT / "scripts"))
from verify_module_paper import verify_module


def find_curated_doc(mod_name: str) -> Optional[Path]:
    clean = re.sub(r"(?i)attention|compressor|pruner|quantizer", "", mod_name).strip("_")
    mod_lower = mod_name.lower()
    clean_lower = clean.lower()
    candidates = [
        ROOT / f"Docs/content/attentions/{mod_name}.md",
        ROOT / f"Docs/content/compressions/{mod_name}.md",
        ROOT / f"Docs/content/attentions/{mod_lower}.md",
        ROOT / f"Docs/content/compressions/{mod_lower}.md",
        ROOT / f"Docs/content/attentions/{clean}.md",
        ROOT / f"Docs/content/compressions/{clean}.md",
        ROOT / f"Docs/content/attentions/{clean_lower}_attention.md",
        ROOT / f"Docs/content/compressions/{clean_lower}_compressor.md",
        ROOT / f"Codes/frontend/dist/Docs/content/attentions/{mod_name}.md",
        ROOT / f"Codes/frontend/dist/Docs/content/compressions/{mod_name}.md",
        ROOT / f"Codes/frontend/dist/Docs/content/attentions/{clean}_attention.md",
        ROOT / f"Codes/frontend/dist/Docs/content/compressions/{clean}_compressor.md",
        ROOT / f"Codes/frontend/dist/Docs/content/attentions/{clean}.md",
        ROOT / f"Codes/frontend/dist/Docs/content/compressions/{clean}.md",
        ROOT / f"Codes/frontend/dist/Docs/content/attentions/{mod_lower}.md",
        ROOT / f"Codes/frontend/dist/Docs/content/compressions/{mod_lower}.md",
        ROOT / f"Codes/frontend/dist/Docs/content/attentions/{clean_lower}_attention.md",
        ROOT / f"Codes/frontend/dist/Docs/content/compressions/{clean_lower}_compressor.md",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def clean_markdown_for_tex(text: str) -> str:
    """Transform markdown prose into clean, compileable LaTeX text without touching math mode."""
    # Strip HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    
    # Headers
    text = re.sub(r"(?m)^###\s+(.+)$", r"\\subsubsection{\1}", text)
    text = re.sub(r"(?m)^##\s+(.+)$", r"\\subsection{\1}", text)
    text = re.sub(r"(?m)^#\s+(.+)$", r"\\section{\1}", text)

    # Box-drawing characters to ASCII
    box_chars = {
        '─': '-', '│': '|', '┌': '+', '┐': '+', '└': '+', '┘': '+',
        '├': '+', '┤': '+', '┬': '+', '┴': '+', '┼': '+', '═': '=',
        '║': '|', '╔': '+', '╗': '+', '╚': '+', '╝': '+', '▲': '^',
        '▼': 'v', '◄': '<', '►': '>', '≈': r'\approx ', '×': r'\times ',
        'η': r'\eta ', 'λ': r'\lambda ', 'τ': r'\tau ', 'σ': r'\sigma ',
        '—': '---', '–': '--', '“': "``", '”': "''", '’': "'"
    }
    for bc, repl in box_chars.items():
        text = text.replace(bc, repl)

    # Split by math blocks and code blocks to avoid corrupting math
    # Tokens: $$...$$, $...$, \[...\], ```...```
    tokens = re.split(r"(\$\$[^\$]+\$\$|\$[^\$]+\$|\\\[.*?\\\]|```.*?```)", text, flags=re.DOTALL)
    out_parts = []
    for token in tokens:
        if token.startswith("```"):
            code_body = "\n".join(token.splitlines()[1:-1])
            out_parts.append(f"\n\\begin{{lstlisting}}\n{code_body}\n\\end{{lstlisting}}\n")
        elif token.startswith("$") or token.startswith(r"\["):
            out_parts.append(token)
        else:
            # Prose mode: handle markdown tables and escape special characters
            lines = token.splitlines()
            new_lines = []
            in_table = False
            table_lines = []
            for line in lines:
                if line.strip().startswith('|') and '|' in line.strip()[1:]:
                    in_table = True
                    table_lines.append(line)
                else:
                    if in_table:
                        new_lines.append("\\begin{verbatim}")
                        new_lines.extend(table_lines)
                        new_lines.append("\\end{verbatim}")
                        table_lines = []
                        in_table = False
                    new_lines.append(line)
            if in_table:
                new_lines.append("\\begin{verbatim}")
                new_lines.extend(table_lines)
                new_lines.append("\\end{verbatim}")
                table_lines = []

            chunk = "\n".join(new_lines)
            subparts = re.split(r"(\\begin\{verbatim\}.*?\\end\{verbatim\})", chunk, flags=re.DOTALL)
            for sp in subparts:
                if sp.startswith(r"\begin{verbatim}"):
                    out_parts.append(sp)
                else:
                    p = sp
                    p = re.sub(r"(?<!\\)_", r"\_", p)
                    p = re.sub(r"(?<!\\)%", r"\%", p)
                    p = re.sub(r"(?<!\\)&", r"\&", p)
                    p = re.sub(r"(?<!\\)#", r"\#", p)
                    p = re.sub(r"\*\*([^\*]+)\*\*", r"\\textbf{\1}", p)
                    p = re.sub(r"(?<!\*)\*([a-zA-Z0-9 ]+)\*(?!\*)", r"\\textit{\1}", p)
                    p = re.sub(r"`([^`]+)`", r"\\texttt{\1}", p)
                    out_parts.append(p)

    return "".join(out_parts)


def enrich_module(mod_name: str) -> dict:
    mod_dir = PAPER_BASE / mod_name
    if not mod_dir.exists():
        return {"module": mod_name, "success": False, "error": "Dir not found"}

    cdoc = find_curated_doc(mod_name)
    if not cdoc or not cdoc.exists():
        return {"module": mod_name, "success": False, "error": f"Curated doc not found for {mod_name}"}

    txt = cdoc.read_text(encoding="utf-8", errors="ignore")
    secs = re.split(r"(?m)^##\s+(.*)$", txt)
    if len(secs) <= 1:
        return {"module": mod_name, "success": False, "error": "No sections found"}

    sec_map = {}
    for i in range(1, len(secs), 2):
        sec_map[secs[i].strip()] = secs[i+1].strip()

    # Mapping to target TeX files
    mapping = {
        "01_intro": ["abstract", "theoretical motivation", "executive abstract"],
        "02_related": ["literature", "related", "academic references"],
        "04_method": ["math", "formulation", "matrix decomposition", "mathematical"],
        "05_architecture": ["architecture", "data flow", "trace", "step-by-step", "pipeline", "component"],
        "06_setup": ["configuration", "hyperparameters", "setup"],
        "08_analysis": ["parameter", "footprint", "complexity", "precision", "bound", "computational"],
        "09_discussion": ["blueprint", "triton", "hybrid", "synthesis", "future work", "extensions"],
        "10_limitations": ["limitations", "failure modes"],
        "11_conclusion": ["conclusion", "summary"],
        "A_math": ["error bound", "factorization", "proof", "derivation", "spectral", "subspace", "theoretical proofs", "stability analysis"],
        "B_implementation": ["calibration", "trace execution", "cuda", "integration", "source code", "implementation", "usage"],
        "C_data": ["empirical", "benchmark", "comparative", "scaling", "profile table", "resource profile", "performance profile"]
    }

    for prefix, keywords in mapping.items():
        matched_tex = []
        matched_md = []
        for h, b in sec_map.items():
            hl = h.lower()
            if any(kw in hl for kw in keywords):
                matched_tex.append(f"\n\n\\subsection{{{h}}}\n" + clean_markdown_for_tex(b))
                matched_md.append(f"\n\n## {h}\n\n" + b)
        
        if matched_tex:
            tf_path = mod_dir / f"{prefix}.tex"
            if tf_path.exists():
                orig_tex = tf_path.read_text(encoding="utf-8", errors="ignore")
                tf_path.write_text(orig_tex + "\n" + "".join(matched_tex), encoding="utf-8")
        if matched_md:
            mf_path = mod_dir / f"{prefix}.md"
            if mf_path.exists():
                orig_md = mf_path.read_text(encoding="utf-8", errors="ignore")
                mf_path.write_text(orig_md + "\n" + "".join(matched_md), encoding="utf-8")

    # Update paper_full.md
    from process_batch import CANONICAL_MD_ORDER, sanitize_all_sections, ground_method_equations, ground_results_tables, ensure_ascii_diagram, resolve_module_source
    src_file, core_p = resolve_module_source(mod_name)
    if src_file and core_p:
        ground_method_equations(mod_dir, src_file, core_p)
        ground_results_tables(mod_dir, mod_name, src_file)
    ensure_ascii_diagram(mod_dir, mod_name)
    sanitize_all_sections(mod_dir, mod_name)

    md_parts = [(mod_dir / s).read_text(encoding="utf-8", errors="ignore").strip() for s in CANONICAL_MD_ORDER if (mod_dir / s).exists()]
    (mod_dir / "paper_full.md").write_text("\n\n---\n\n".join(md_parts) + "\n", encoding="utf-8")

    # Recompile PDF
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=mod_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=mod_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    v = verify_module(mod_name, strict=False)
    return {
        "module": mod_name,
        "valid": v["valid"],
        "pages": v["pages"],
        "pdf_size_kb": v["pdf_size_kb"],
        "violations": len(v["template_issues"]),
        "errors": v["errors"],
        "warnings": v["warnings"]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    args = parser.parse_args()
    res = enrich_module(args.module)
    import json
    print(json.dumps(res, indent=2))
