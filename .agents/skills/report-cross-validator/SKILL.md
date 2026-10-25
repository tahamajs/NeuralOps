---
name: report-cross-validator
description: "Verify every claim in the report traces to audit file AND results/registry"
version: "1.0.0"
tags: ["neuralops", "report", "validation", "traceability", "forensics"]
---

# Skill: Cross-Audit Report Validation and Evidence Integrity Gate

## When to use
Use this skill after all chapters, tables, figures, and appendices of the technical report or doctoral thesis have been authored, but before executing the final production release compilation. It conducts exhaustive end-to-end verification ensuring that every quantitative claim, table row count, figure float, citation reference, and canonical invariant matches the underlying empirical ground truth.

## Prerequisites
- Completed draft of the master manuscript located at `thesis_final/810101504_main.tex`.
- Python 3.10+ execution environment with regular expressions and JSON parsing tools.
- Read access to all audit files in `docs/` and raw telemetry records in `results/registry/master_ledger.jsonl`.
- Remote SSH connectivity to `taha@192.168.1.100` (or local static AST auditor) to verify canonical repository counts.

## Canonical Audit Files Reference
The cross-validation suite evaluates manuscript assertions against the ground truth data stored in all six canonical audit files:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Ground truth for KD ablation numbers, RoBERTa recovery percentages, and Ridge calibration loss metrics.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Ground truth for SVD rank sweeps, singular value spectra, and the 72 compression engine parameters.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Ground truth for 203 unified module latencies, 131 attention operator benchmarks, and the 183.1 FLOP/Byte roofline limit.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Ground truth for token sequence scaling horizons ($N \in [128, 8192]$) and memory allocation cliff measurements.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Ground truth for forensic Git commit hashes, UTC execution timestamps, and invariant registry verification records.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Ground truth for checking that all 20 agent roles and chapter assignment boundaries were strictly respected.

---

## The 5 Automated Cross-Validation Checks

### Check 1: Comprehensive Numerical Traceability Audit
Extracts every decimal percentage, speedup factor, latency, and memory measurement from the LaTeX source and ensures that it appears verbatim in at least one canonical audit file:

```python
import re
from pathlib import Path

def check_numerical_traceability(thesis_path: Path, audit_files: list[Path]):
    thesis_text = thesis_path.read_text()
    numbers = set(re.findall(r"\b\d+\.\d+\b", thesis_text))
    
    audit_corpus = "\n".join([p.read_text() for p in audit_files])
    untraced = []
    
    for num in sorted(numbers):
        if num not in audit_corpus:
            untraced.append(num)
            
    assert len(untraced) == 0, f"Found {len(untraced)} untraced numbers in thesis: {untraced[:10]}"
    print(f"PASS: All {len(numbers)} numerical claims traced to canonical audit corpus.")
```

### Check 2: Table Row Count and Cardinality Parity
Validates that every converted LaTeX table contains the exact number of rows present in the corresponding source audit table:
- Master Attention Table: Exactly 131 module rows.
- Master Compressor Table: Exactly 72 engine rows.
- Master Telemetry Registry Table: Exactly 203 unified operator rows.
- CUDA Execution Table: Exactly 108 native kernel rows.

```python
def check_table_cardinality(latex_content: str, label: str, expected_rows: int):
    # Locate table block by label
    table_pattern = rf"\\label\{{{label}\}}.*?(?:\\end\{{tabular\}}|\\end\{{longtable\}})"
    match = re.search(table_pattern, latex_content, re.DOTALL)
    assert match, f"Table with label {label} not found in manuscript"
    rows = [r for r in match.group(0).split(r"\\") if r.strip() and not r.strip().startswith(r"\toprule")]
    assert len(rows) >= expected_rows, f"Table {label} has {len(rows)} rows, expected {expected_rows}"
    print(f"PASS: Table {label} verified with {len(rows)} rows.")
```

### Check 3: Figure File Existence and Integrity
Parses all `\includegraphics{figures/<filename>}` calls within the manuscript and verifies that each target file exists on disk and has non-zero size:

```python
def check_figure_assets(latex_text: str, figures_dir: Path):
    figures = re.findall(r"\\includegraphics(?:\[.*?\])?\{figures/(.+?)\}", latex_text)
    for fig_name in set(figures):
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing figure file: {fig_path}"
        assert fig_path.stat().st_size > 0, f"Empty figure file: {fig_path}"
    print(f"PASS: All {len(figures)} figure assets exist and are verified non-empty.")
```

### Check 4: Bibliography Resolution and Citation Integrity
Extracts all `\cite{...}` keys across the entire manuscript and verifies that every key corresponds to an existing entry in `thesis_final/references.bib`:

```bash
grep -E "\\cite\{[^}]+\}" thesis_final/810101504_main.tex | \
    sed 's/.*\\cite{\(.*\)}.*/\1/' | tr ',' '\n' | sed 's/^[ \t]*//;s/[ \t]*$//' | sort -u | \
    while read key; do
        grep -q "@.*{$key," thesis_final/references.bib || echo "MISSING CITATION: $key"
    done
```

### Check 5: Canonical Invariant Preservation
Validates that repository registries remain untouched and perfectly synchronized:
- Exactly 131 attention mechanisms in `ATTENTION_REGISTRY`.
- Exactly 72 compression engines in `COMPRESSOR_REGISTRY`.
- Exactly 26 canonical modules in `neuralops/core/triton_kernels/`.

```bash
python3 scripts/utils/audit_dead_code.py
python3 scripts/audit_readme_consistency.py
```

---

## Final Validation Report Generation (`docs/reports/REPORT_VALIDATION_REPORT.md`)

Upon passing all 5 checks, generate the final markdown validation artifact summarizing the audit results:

```markdown
# NeuralOps Thesis & Report Evidence Validation Certification

| Quality Gate Check | Status | Empirical Validation Details |
|:---|:---:|:---|
| All Numerical Claims Traced | ✅ PASS | 847/847 numbers traced to AUDIT_01 through AUDIT_05 |
| Table Row Cardinality Parity | ✅ PASS | 15/15 tables match exact audit row counts (203/131/72/108) |
| Figure File Integrity | ✅ PASS | 8/8 figures present in thesis_final/figures/ (>0 bytes) |
| Bibliography Resolution | ✅ PASS | 127/127 citations resolved with 0 dangling keys |
| Canonical Invariants Preserved | ✅ PASS | Exactly 131 attentions / 72 compressors / 26 Triton modules |
```

---

## Strict Validation Invariants

1. **Zero Tolerance for Untraced Claims**: If even a single floating-point number in the thesis fails to match the audit files, the validation must fail immediately.
2. **Deterministic Certification**: The validation report must include the Git commit SHA, system timestamp, and machine hostname.
3. **Immutable Audits**: The validator must never alter the source audit files to "fit" the thesis text. If discrepancies exist, the thesis text must be corrected.

---

## Verification
Execute the master validator runner:

```bash
python3 scripts/utils/verify_report_cross_validation.py \
    --thesis thesis_final/810101504_main.tex \
    --output docs/reports/REPORT_VALIDATION_REPORT.md
```

Confirm that the output report displays green checkmarks (`✅ PASS`) across all 5 verification rows.

---

## Common Pitfalls
- Overlooking numbers embedded within LaTeX math subscripts or exponents.
- Allowing table row omissions during pagination adjustments in LaTeX appendices.
- Modifying audit files to mask typographical errors in the manuscript.
- Neglecting to verify the agent file mapping specified in `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`.
