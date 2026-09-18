#!/usr/bin/env python3
"""
Exhaustive audit to detect template artifacts across all PerModulePaper modules.
Run this BEFORE any regeneration to quantify the damage.
"""
import re
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("paper/PerModulePaper")
OUT  = Path("audit/TEMPLATE_ARTIFACT_AUDIT_2026_09_18")
OUT.mkdir(parents=True, exist_ok=True)

# The 10 categories of artifacts found in lora_attention
ARTIFACT_PATTERNS = {
    "double_percent":       r'%%',
    "double_cross":         r'××',
    "duplicate_MB":         r'\bMB\s+MB\b',
    "duplicate_KB":         r'\bKB\s+KB\b',
    "nonsense_units":       r'\d+\s*x\s+s\b',       # "1.42x s"
    "placeholder_xx":       r'\d+\.?\d*xx',
    "unsubstituted_var":    r'\b[A-Z_]{4,}(?=[×x\*]|\b)',  # TRAINREDUCTION
    "double_curly":         r'\{\{',                # unrendered LaTeX
    "double_close_curly":   r'\}\}',
    "todo_marker":          r'<TODO>|FILL_IN|PLACEHOLDER',
    "raw_star":             r'\*[a-z_]+\b',          # "*-lora attention"
    "double_period":        r'\.\.\b',
}

# Numbers that are suspiciously repeated (placeholder values)
SUSPICIOUS_NUMBERS = ["0.85", "0.32", "0.85%", "86.4", "32.5"]

def analyze_module(module_dir):
    """Analyze one module and return a dict of findings."""
    findings = {
        "module": module_dir.name,
        "files_checked": 0,
        "artifact_counts": defaultdict(int),
        "artifact_examples": defaultdict(list),
        "suspicious_number_counts": defaultdict(int),
        "total_words": 0,
        "total_equations": 0,
        "total_source_citations": 0,
        "equation_citation_ratio": 0.0,
        "python_syntax_errors": [],
        "section_duplications": [],
    }

    # Check every markdown file
    for md in sorted(module_dir.glob("*.md")):
        findings["files_checked"] += 1
        text = md.read_text(errors="ignore")
        findings["total_words"] += len(text.split())
        findings["total_equations"] += len(re.findall(r'\$\$', text)) // 2
        findings["total_source_citations"] += len(re.findall(r'<!--\s*source:', text))

        # Check each artifact pattern
        for name, pat in ARTIFACT_PATTERNS.items():
            matches = re.findall(pat, text)
            if matches:
                findings["artifact_counts"][name] += len(matches)
                # Save first 3 examples
                for m in re.finditer(pat, text):
                    line_no = text[:m.start()].count('\n') + 1
                    if len(findings["artifact_examples"][name]) < 3:
                        findings["artifact_examples"][name].append(
                            f"{md.name}:{line_no}: {m.group()[:60]}"
                        )

        # Count suspicious repeated numbers
        for num in SUSPICIOUS_NUMBERS:
            cnt = len(re.findall(re.escape(num), text))
            if cnt > 0:
                findings["suspicious_number_counts"][num] += cnt

    # Check Python code blocks for syntax errors
    for md in sorted(module_dir.glob("*.md")):
        text = md.read_text(errors="ignore")
        for match in re.finditer(r'```python\n(.*?)```', text, re.DOTALL):
            code = match.group(1)
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                findings["python_syntax_errors"].append({
                    "file": md.name,
                    "line": e.lineno,
                    "error": str(e),
                })

    # Equation-citation ratio
    if findings["total_equations"] > 0:
        findings["equation_citation_ratio"] = (
            findings["total_source_citations"] / findings["total_equations"]
        )

    return findings

def cross_module_diff(module_a, module_b, section="04_method.md"):
    """Compare two modules' method sections."""
    fa = module_a / section
    fb = module_b / section
    if not fa.exists() or not fb.exists():
        return None
    lines_a = fa.read_text(errors="ignore").splitlines()
    lines_b = fb.read_text(errors="ignore").splitlines()
    # Simple line-based overlap
    set_a = set(lines_a)
    set_b = set(lines_b)
    common = len(set_a & set_b)
    total = max(len(set_a), len(set_b))
    return {
        "module_a": module_a.name,
        "module_b": module_b.name,
        "section": section,
        "common_lines": common,
        "total_lines": total,
        "overlap_pct": 100.0 * common / total if total else 0,
    }

# ============ MAIN ============
modules = sorted([d for d in ROOT.iterdir() if d.is_dir() and d.name != "_TEMPLATE"])

print(f"Found {len(modules)} modules to audit")

all_findings = []
for m in modules:
    try:
        f = analyze_module(m)
        all_findings.append(f)
        print(f"  {m.name}: {sum(f['artifact_counts'].values())} artifacts, "
              f"{f['total_source_citations']} citations, "
              f"{f['total_equations']} equations")
    except Exception as e:
        print(f"  {m.name}: ERROR {e}")
        all_findings.append({"module": m.name, "error": str(e)})

# Save per-module JSON
with open(OUT / "per_module_findings.json", "w") as fp:
    json.dump(all_findings, fp, indent=2, default=str)

# Save CSV summary
with open(OUT / "per_module_summary.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow([
        "module", "total_words", "total_equations",
        "total_source_citations", "equation_citation_ratio",
        "total_artifact_count", "python_syntax_errors"
    ])
    for f in all_findings:
        if "error" in f:
            w.writerow([f["module"], "ERROR", "", "", "", "", f["error"]])
            continue
        w.writerow([
            f["module"],
            f["total_words"],
            f["total_equations"],
            f["total_source_citations"],
            f"{f['equation_citation_ratio']:.2f}",
            sum(f["artifact_counts"].values()),
            len(f["python_syntax_errors"]),
        ])

# Cross-module overlap analysis (sample 10 pairs)
print("\n=== Cross-module overlap (04_method.md) ===")
pairs = []
sample = modules[:20]  # take 20 for pairwise
for i, a in enumerate(sample):
    for b in sample[i+1:]:
        d = cross_module_diff(a, b)
        if d:
            pairs.append(d)

with open(OUT / "cross_module_overlap.csv", "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=["module_a", "module_b", "section",
                                       "common_lines", "total_lines", "overlap_pct"])
    w.writeheader()
    for p in pairs:
        w.writerow(p)

avg_overlap = sum(p["overlap_pct"] for p in pairs) / len(pairs) if pairs else 0
print(f"Average overlap across sampled pairs: {avg_overlap:.1f}%")

# Summary statistics
print("\n=== Global Summary ===")
total_artifacts = sum(sum(f.get("artifact_counts", {}).values()) for f in all_findings)
print(f"Total artifacts detected: {total_artifacts}")
print(f"Modules with 0 source citations: {sum(1 for f in all_findings if f.get('total_source_citations', 0) == 0)}")
print(f"Modules with syntax errors in python blocks: {sum(1 for f in all_findings if f.get('python_syntax_errors'))}")
print(f"Average equation/citation ratio: {sum(f.get('equation_citation_ratio', 0) for f in all_findings) / len(all_findings):.2f}")
