---
name: report-table-extractor
description: "Extract Markdown tables from audit files and convert to LaTeX booktabs with full provenance columns"
version: "1.0.0"
tags: ["neuralops", "report", "latex", "tables", "booktabs"]
---

# Skill: LaTeX Table Extractor and Booktabs Formatter

## When to use
Use this skill whenever generating tabular content for thesis chapters, academic conference manuscripts, or technical reports. It provides a rigorous conversion methodology that ingests verified Markdown tables from NeuralOps audit files and outputs pristine LaTeX `booktabs` environments, guaranteeing zero data truncation, full provenance attribution, and exact numeric fidelity.

## Prerequisites
- Completed execution of `report-audit-file-reader` producing `/tmp/citation_index.json`.
- Access to the LaTeX document root directory (e.g., `thesis_final/` or `paper/`).
- Familiarity with the standard LaTeX packages: `booktabs`, `caption`, `float`, and `siunitx`.
- Python 3.10+ execution environment for automated table generation scripts.

## Canonical Audit Files Reference
Every table extracted and converted by this skill originates strictly from the six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Source for knowledge distillation ablation matrices (Table 1 to Table 6), closed-form Ridge parameter tables, and RoBERTa accuracy recovery numbers.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Source for SVD energy retention tables, rank-sweep matrices (Table 1 to Table 4), and Cholesky whitening parameter benchmarks.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Source for the master 203-module telemetry registry (Table 1), 131 attention benchmarks (Table 4), and 108 CUDA C++ kernel execution timings.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Source for sequence length scaling tables (Table 1 to Table 3), peak VRAM memory allocations, and linear-to-quadratic latency crossover coordinates.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Source for architectural integrity verification tables, Git commit SHA mapping, and hardware testbed environment disclosures.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Source for mapping table generation tasks to designated agent responsibility boundaries and report sections.

---

## Extraction Workflow and Transformation Protocol

### Step 1: Query the Structured Citation Index
Rather than manually copying markdown cells, the extractor reads parsed records directly from the structured citation index (`/tmp/citation_index.json`):

```python
import json
from pathlib import Path

def load_audit_table(audit_id: str, table_idx: int) -> dict:
    index_path = Path("/tmp/citation_index.json")
    if not index_path.exists():
        raise RuntimeError("Citation index missing. Run report-audit-file-reader first.")
    data = json.loads(index_path.read_text())
    return data[audit_id]["tables"][table_idx]
```

### Step 2: Establish the LaTeX Booktabs Template
Every converted table must strictly use the formal `booktabs` design standard without vertical rules (`|`). The table structure must conform to the following schema:

```latex
\begin{table}[H]
\centering
\small
\caption{<Descriptive caption from audit file> \cite{<audit_bib_key>}}
\label{tab:<unique_reference_label>}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Module} & \textbf{Family} & \textbf{Latency (ms)} & \textbf{Speedup} & \textbf{Diff} & \textbf{Source} \\
\midrule
flash_attention_v2 & Exact Softmax & 0.421 & 3.12$\times$ & 0.00000 & \texttt{AUDIT\_03} \\
mamba_ssm & Recurrent SSM & 0.389 & 3.38$\times$ & 0.00008 & \texttt{AUDIT\_03} \\
...
\bottomrule
\end{tabular}
\end{table}
```

### Step 3: Implement the Markdown-to-LaTeX Transformer Engine
Construct a Python translation engine that escapes special LaTeX characters, formats numbers, and appends the mandatory `Source` provenance column:

```python
def escape_latex(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}"
    }
    for char, replacement in replacements:
        text = text.replace(char, replacement)
    return text

def convert_table_to_latex(
    table_data: dict,
    caption: str,
    label: str,
    audit_source_tag: str
) -> str:
    headers = table_data["headers"] + ["Source"]
    num_cols = len(headers)
    col_spec = "l" + "r" * (num_cols - 2) + "r"
    
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption} \\cite{{{audit_source_tag.lower()}}}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(f"\\textbf{{{escape_latex(h)}}}" for h in headers) + r" \\",
        r"\midrule"
    ]
    
    for row in table_data["rows"]:
        row_vals = [escape_latex(str(row.get(h, ""))) for h in table_data["headers"]]
        row_vals.append(f"\\texttt{{{audit_source_tag}}}")
        lines.append(" & ".join(row_vals) + r" \\")
        
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    return "\n".join(lines)
```

---

## Invariant Transformation Rules

1. **Zero Row Truncation**: Every single row present in the source audit table must appear in the output LaTeX table. If an audit table lists 131 or 203 rows, the LaTeX output must contain all 131 or 203 rows (using `longtable` if necessary for multi-page appendices).
2. **Mandatory Provenance Column**: Every generated table MUST include a right-aligned `Source` column explicitly identifying the origin audit document (e.g., `\texttt{AUDIT_01}`, `\texttt{AUDIT_03}`).
3. **Exact Numeric Precision**: Precision must be preserved down to the last decimal digit. Never round `0.421` to `0.42`, nor `73.62%` to `74%`.
4. **Preserve Row Ordering**: The original sort order of the audit table (whether alphabetical, by latency, or by family) must remain identical.
5. **No Vertical Lines**: Under no circumstances should `|` be used in LaTeX `\begin{tabular}` specifications. Use only `\toprule`, `\midrule`, and `\bottomrule`.

---

## Verification
Validate the transformed LaTeX tables using an automated verification script:

```python
def verify_extracted_table(latex_str: str, original_rows: list[dict], source_tag: str):
    # Count rows by scanning for LaTeX line terminators
    table_lines = [
        line.strip() for line in latex_str.splitlines() 
        if line.strip().endswith(r"\\") and not line.strip().startswith(r"\textbf")
    ]
    # Check 1: Row count parity
    assert len(table_lines) == len(original_rows), (
        f"Row count mismatch! Expected {len(original_rows)}, found {len(table_lines)}"
    )
    # Check 2: Provenance column presence
    for line in table_lines:
        assert f"\\texttt{{{source_tag}}}" in line, f"Missing provenance source tag in line: {line}"
    print("Table extraction and formatting verified successfully.")
```

---

## Common Pitfalls
- Failing to escape `%` in percentage values (e.g., `85.4%`), resulting in LaTeX comment truncation.
- Using standard `table` environment for large tables ($>45$ rows) instead of `longtable`, causing page overflows.
- Omitting the `\cite{...}` reference in the table caption.
- Re-sorting or filtering rows based on personal preference, violating the canonical audit snapshot.
- Forgetting to import the `booktabs` package in the LaTeX document preamble.
- Disregarding the cross-reference mapping in `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md` when assigning table numbering.
