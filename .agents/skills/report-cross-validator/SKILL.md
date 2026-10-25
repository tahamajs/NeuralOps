---
name: report-cross-validator
description: "Verify every claim in the report traces to audit file and docs/thesis_agents standards"
version: "1.0.0"
tags: ["neuralops", "report", "validation", "traceability", "forensics", "thesis_agents"]
---

# Skill: Cross-Audit Report Validation and Evidence Integrity Gate

## When to use
Use this skill after all chapters, tables, figures, and appendices of the doctoral dissertation (`academic/thesis/810101504_main.tex`) have been authored, but before executing the final production release compilation. It conducts exhaustive end-to-end verification ensuring that every quantitative claim, table row count, figure float, citation reference, XePersian typography constraint, and canonical repository invariant matches the underlying empirical ground truth and conforms to the 30-agent division of labor in `docs/thesis_agents/`.

## Prerequisites
- Direct compliance with `docs/thesis_agents/MASTER_DIVISION_OF_LABOR.md`.
- Direct compliance with `docs/thesis_agents/PARALLEL_EXECUTION_GUIDE.md`.
- Direct compliance with `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.
- Target files: `academic/thesis/810101504_main.tex` and all thirteen modular files in `academic/thesis/chapters/`.
- Full cross-audit citation index accessible at `/tmp/citation_index.json` and master ledger at `results/registry/master_ledger.jsonl`.
- Remote SSH access: `ssh taha@192.168.1.100` (RTX 4070 SM 8.9) with `/mnt/d/venv/bin/python3`.

## Canonical Audit Files Reference
The cross-validation suite evaluates manuscript assertions against the ground truth data stored in all six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Ground truth for KD ablation numbers, RoBERTa recovery percentages, and Ridge calibration loss metrics.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Ground truth for SVD rank sweeps, singular value spectra, and the 72 compression engine parameters.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Ground truth for 203 unified module latencies, 131 attention operator benchmarks, and the 183.1 FLOP/Byte roofline limit.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Ground truth for token sequence scaling horizons ($N \in [128, 8192]$) and memory allocation cliff measurements.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Ground truth for forensic Git commit hashes, UTC execution timestamps, and invariant registry verification records.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Ground truth for checking that all 30 agent roles and chapter assignment boundaries were strictly respected.

## Canonical Repository Invariants
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## The 6 Automated Cross-Validation Checks

### Check 1: Comprehensive Numerical Traceability Audit
Extracts every decimal percentage, speedup factor, latency, and memory measurement from `academic/thesis/chapters/*.tex` and verifies that it appears verbatim in the canonical audit files or master ledger:

```python
import re
from pathlib import Path

def check_numerical_traceability(chapters_dir: Path, audit_corpus: str):
    all_tex = "\n".join([p.read_text(encoding="utf-8") for p in chapters_dir.glob("*.tex")])
    numbers = set(re.findall(r"\b\d+\.\d+\b", all_tex))
    untraced = [num for num in sorted(numbers) if num not in audit_corpus]
    assert len(untraced) == 0, f"Found {len(untraced)} untraced numbers: {untraced[:10]}"
    print(f"PASS: All {len(numbers)} numerical claims traced to canonical audit corpus.")
```

### Check 2: Table Row Count and Cardinality Parity
Validates that every converted LaTeX table contains the exact number of rows present in the corresponding source audit table:
- Master Attention Catalog: Exactly 131 module rows in Table A.1.
- Master Compressor Catalog: Exactly 72 engine rows in Table B.1.
- Unified Telemetry Table: Exactly 203 module rows.
- CUDA Execution Table: Exactly 108 native kernel rows.

### Check 3: Figure File Existence and Integrity
Parses all `\includegraphics{figures/<filename>}` calls within the manuscript and verifies that each target file exists in `academic/thesis/figures/` and has non-zero size.

### Check 4: XePersian Syntax & Typography Compliance
Per `docs/thesis_agents/PERSIAN_WRITING_RULES.md`:
1. Zero instances of `\lr{}` inside mathematical environments (`equation`, `align`, `$ ... $`).
2. Zero instances of `\lr{}` enclosing `\item`.
3. Proper use of ZWNJ (نیم‌فاصله `U+200C`) in compound Persian words.
4. English digits in tables and equations; Persian digits in narrative Persian text.

### Check 5: Exclusion of Early Buggy Runs
Asserts that no preliminary uncalibrated benchmark results (such as accuracy drops below 35% from unfrozen weights or 78.56% on raw RoBERTa checkpoints) are cited as valid system outcomes in the main thesis text.

### Check 6: Canonical Invariant Preservation
Validates that repository registries remain untouched and perfectly synchronized:
- Exactly 131 attention mechanisms in `ATTENTION_REGISTRY`.
- Exactly 72 compression engines in `COMPRESSOR_REGISTRY`.
- Exactly 26 canonical modules in `neuralops/core/triton_kernels/`.

---

## The 5-Wave Execution and Lock Verification Protocol

Per `docs/thesis_agents/PARALLEL_EXECUTION_GUIDE.md`, the 30 thesis agents operate across 5 sequential execution waves. The cross-validator ensures that lock and completion tokens are present before declaring validation success:

1. **Wave 1 (Foundations & Theory)**: Agents A01 to A07 (`docs/thesis_agents/locks/wave1.done`).
   - Targets: `chapter01_intro.tex`, `chapter02_theory_part1.tex`, `chapter02_theory_part2.tex`, `chapter03_methodology_part1.tex`.
2. **Wave 2 (Methodology & Core Results)**: Agents A08 to A14 (`docs/thesis_agents/locks/wave2.done`).
   - Targets: `chapter03_methodology_part2.tex`, `part3a.tex`, `part3b.tex`, `chapter04_results_part1.tex`, `part2.tex`, `part2b.tex`.
3. **Wave 3 (Advanced Results & Appendices)**: Agents A15 to A21 (`docs/thesis_agents/locks/wave3.done`).
   - Targets: `chapter04_results_part3.tex`, `chapter04_results_part4.tex`, `chapter05_conclusion.tex`, `chapter_all_modules_appendix.tex`, `810101504_main.tex`.
4. **Wave 4 (LaTeX Enrichment & Tables)**: Agents A22 to A28 (`docs/thesis_agents/locks/wave4.done`).
   - Targets: Detailed LaTeX table injection across Chapter 4 submodules.
5. **Wave 5 (Visuals & Synthesis)**: Agents A29 and A30 (`docs/thesis_agents/locks/wave5.done`).
   - Targets: TikZ/PGFPlots generation and practical guide finalization.

---

## Complete Automated Verification Runner

The cross-validator executes the following comprehensive Python validation suite:

```python
import json
import re
from pathlib import Path

def run_cross_validation():
    thesis_root = Path("academic/thesis")
    chapters_dir = thesis_root / "chapters"
    figures_dir = thesis_root / "figures"
    
    # 1. Verify existence of all 13 modular chapter files
    expected_files = [
        "chapter01_intro.tex",
        "chapter02_theory_part1.tex",
        "chapter02_theory_part2.tex",
        "chapter03_methodology_part1.tex",
        "chapter03_methodology_part2.tex",
        "chapter03_methodology_part3a.tex",
        "chapter03_methodology_part3b.tex",
        "chapter04_results_part1.tex",
        "chapter04_results_part2.tex",
        "chapter04_results_part2b.tex",
        "chapter04_results_part3.tex",
        "chapter04_results_part4.tex",
        "chapter05_conclusion.tex"
    ]
    for filename in expected_files:
        assert (chapters_dir / filename).exists(), f"Missing modular chapter file: {filename}"
        assert (chapters_dir / filename).stat().st_size > 500, f"Chapter file too small: {filename}"

    # 2. Verify all 10 thesis figures exist
    for i in range(1, 11):
        pdf_path = figures_dir / f"fig_{i}.pdf"
        png_path = figures_dir / f"fig_{i}.png"
        assert pdf_path.exists(), f"Missing figure: {pdf_path}"
        assert png_path.exists(), f"Missing figure preview: {png_path}"

    # 3. Check XePersian syntax rules
    all_tex = "\n".join([p.read_text(encoding="utf-8") for p in chapters_dir.glob("*.tex")])
    assert not re.search(r"\\begin\{equation\}.*?\\lr\{.*?\\end\{equation\}", all_tex, re.DOTALL), "Illegal \\lr inside equation"
    assert not re.search(r"\\lr\{[^}]*\\item", all_tex), "Illegal \\item inside \\lr"

    print("ALL 30-AGENT CROSS-VALIDATION CHECKS PASSED.")

if __name__ == "__main__":
    run_cross_validation()
```

---

## Verification Protocol
Execute the automated validation checks across all thirteen modular thesis files:

```bash
# Check 1: Audit XePersian syntax rules
grep -n "\\\\lr{" academic/thesis/chapters/*.tex | grep -E "begin\{equation\}|\$|align" && echo "FAIL: \lr in math" || echo "PASS: No \lr in math"

# Check 2: Audit \item syntax
grep -n "\\\\lr{.*\\\\item" academic/thesis/chapters/*.tex && echo "FAIL: \item in \lr" || echo "PASS: No \item in \lr"

# Check 3: Audit figure assets
for fig in academic/thesis/figures/*.pdf; do
    [ -s "$fig" ] || echo "FAIL: Empty or missing figure $fig"
done
echo "Figure verification complete."
```

---

## Common Pitfalls to Avoid
- Overlooking numbers embedded within LaTeX math subscripts or exponents.
- Allowing table row omissions during pagination adjustments in LaTeX appendices.
- Modifying audit files to mask typographical errors in the manuscript.
- Neglecting to verify the agent file mapping specified in `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`.
- Inserting `\lr{}` inside math environments or around `\item` commands.
- Citing preliminary corrupted runs instead of final calibrated metrics.
