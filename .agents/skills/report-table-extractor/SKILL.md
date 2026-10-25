---
name: report-table-extractor
description: "Extract Markdown tables from audit files and convert to LaTeX booktabs per docs/thesis_agents guides"
version: "1.0.0"
tags: ["neuralops", "report", "latex", "tables", "booktabs", "thesis_agents"]
---

# Skill: LaTeX Table Extractor and Booktabs Formatter

## When to use
Use this skill whenever generating tabular content for doctoral thesis chapters, academic conference manuscripts, or technical reports. It provides a rigorous conversion methodology that ingests verified Markdown tables from NeuralOps audit files and outputs pristine LaTeX `booktabs` environments, guaranteeing zero data truncation, full provenance attribution, and exact numeric fidelity according to Agents 26, 27, and 28.

## Prerequisites
- Direct compliance with `docs/thesis_agents/AGENT_26_CHAPTER4_COMPRESSION_LATEX.md`.
- Direct compliance with `docs/thesis_agents/AGENT_27_CHAPTER4_KERNELS_LATEX.md`.
- Direct compliance with `docs/thesis_agents/AGENT_28_CHAPTER4_MACROTELEMETRY.md`.
- Full compliance with `docs/thesis_agents/MASTER_DIVISION_OF_LABOR.md` and `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.
- Target files: `academic/thesis/chapters/chapter04_results_part1.tex`, `chapter04_results_part2b.tex`, and `chapter04_results_part3.tex`.
- Python 3.10+ execution environment for automated table generation scripts.

## Canonical Audit Files Reference
Every table extracted and converted by this skill originates strictly from the six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Source for knowledge distillation ablation matrices (Table 1 to Table 6), closed-form Ridge parameter tables, and RoBERTa accuracy recovery numbers.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Source for SVD energy retention tables, rank-sweep matrices (Table 1 to Table 4), and Cholesky whitening parameter benchmarks.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Source for the master 203-module telemetry registry (Table 1), 131 attention benchmarks (Table 4), and 108 CUDA C++ kernel execution timings.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Source for sequence length scaling tables (Table 1 to Table 3), peak VRAM memory allocations, and linear-to-quadratic latency crossover coordinates.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Source for architectural integrity verification tables, Git commit SHA mapping, and hardware testbed environment disclosures.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Source for mapping table generation tasks to designated agent responsibility boundaries and report sections.

## Canonical Repository Invariants
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Canonical Table Templates from Thesis Agents

### 1. Macro Telemetry Classification Table (Agent 28)
Target: `academic/thesis/chapters/chapter04_results_part1.tex`.
Source: `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`.

```latex
\begin{table}[H]
\centering
\small
\caption{دسته‌بندی پنج‌گانه تله‌متری کلان ۲۰۳ ماژول پلتفرم \cite{audit03}}
\label{tab:macro_classification_5cat}
\begin{tabular}{llrrl}
\toprule
\textbf{دسته‌بندی} & \textbf{تعریف وضعیت} & \textbf{تعداد} & \textbf{درصد} & \textbf{سهم بک‌اند غالب} \\
\midrule
PASS\_FAST & سرعت بالاتر از مرجع پایتورچ ($\ge 1.0\times$) & 42 & 20.7\% & Triton JIT (62\%) \\
PASS\_SLOW & سرعت پایین‌تر به دلیل سربار راه‌اندازی & 161 & 79.3\% & PyTorch Reference (85\%) \\
FAIL\_TIMEOUT & وقفه در اجرای استنتاج بیش از حد مجاز & 0 & 0.0\% & -- \\
FAIL\_OOM & سرریز حافظه گرافیکی در طول توالی & 0 & 0.0\% & -- \\
FAIL\_NAN & ناپایداری عددی و گرادیان نامتناهی & 0 & 0.0\% & -- \\
\bottomrule
\end{tabular}
\end{table}
```

### 2. Four-Way HNC-SVD Ablation Table (Agent 26)
Target: `academic/thesis/chapters/chapter04_results_part2b.tex`.
Source: `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`.

```latex
\begin{table}[H]
\centering
\small
\caption{مقایسه عملکردی چهار روش فشرده‌سازی در نسبت رتبه ۵۰٪ روی مدل \lr{RoBERTa-base} \cite{audit01}}
\label{tab:hnc_ablation_4way}
\begin{tabular}{lrrrrr}
\toprule
\textbf{پیکربندی روش} & \textbf{دقت SST-2} & \textbf{حفظ دقت} & \textbf{خطای بازسازی} & \textbf{زمان (s)} & \textbf{منبع} \\
\midrule
مدل متراکم مبنا & 94.50\% & 100.0\% & 0.0000 & -- & \texttt{AUDIT\_01} \\
Truncated SVD خام & 52.06\% & 55.09\% & 0.1842 & 4.2 & \texttt{AUDIT\_01} \\
Cholesky SVD & 78.40\% & 82.96\% & 0.1105 & 8.7 & \texttt{AUDIT\_01} \\
HNC-SVD بدون هیلینگ & 86.25\% & 91.27\% & 0.0894 & 12.1 & \texttt{AUDIT\_01} \\
HNC-SVD + هیلینگ ۳۰ گام & 93.12\% & 98.54\% & 0.0612 & 34.5 & \texttt{AUDIT\_01} \\
\bottomrule
\end{tabular}
\end{table}
```

### 3. Three-Level Kernel Benchmark Table (Agent 27)
Target: `academic/thesis/chapters/chapter04_results_part3.tex`.
Source: `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`.

```latex
\begin{table}[H]
\centering
\small
\caption{ارزیابی تجربی ۱۸ عملگر سه‌سطحی پلتفرم در ابعاد پرتاب استاندارد \cite{audit03}}
\label{tab:benchmark_3level_18ops}
\begin{tabular}{llrrrr}
\toprule
\textbf{عملگر کِرنل} & \textbf{بک‌اند برتر} & \textbf{زمان CUDA (ms)} & \textbf{زمان Triton (ms)} & \textbf{زمان PyTorch (ms)} & \textbf{شتاب} \\
\midrule
selective\_ssm & Native CUDA C++ & 0.382 & 0.495 & 1.340 & 3.51$\times$ \\
flash\_attn\_tile & Native CUDA C++ & 0.412 & 0.450 & 1.280 & 3.11$\times$ \\
gated\_linear_unit & Triton JIT & 0.185 & 0.142 & 0.520 & 3.66$\times$ \\
moe\_routing_gate & Native CUDA C++ & 0.082 & 0.110 & 1.100 & 13.41$\times$ \\
fused_layernorm & Triton JIT & 0.095 & 0.084 & 0.310 & 3.69$\times$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Invariant Transformation Rules
1. **Zero Row Truncation**: Every row in the canonical audit table must be rendered.
2. **Mandatory Provenance Column**: Every table must cite its origin audit document (`\texttt{AUDIT_01}` through `\texttt{AUDIT_05}`).
3. **Exact Numeric Precision**: Precision must be preserved down to the last decimal digit (never round 93.12% to 93%).
4. **No Vertical Lines**: Strict adherence to the `booktabs` standard (`\toprule`, `\midrule`, `\bottomrule`).
5. **English Digits in Tables**: Numeric metrics in tabular cells must be in English characters (`93.12%`), while Persian captions describe the contents.
6. **No Early Buggy Runs**: Never transcribe flawed preliminary runs (<35% accuracy, uncalibrated checkpoints).

---

## Verification Protocol
Validate the transformed LaTeX tables using an automated verification script:

```python
import re
from pathlib import Path

def verify_table_fidelity(tex_path: Path):
    content = tex_path.read_text(encoding="utf-8")
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", content, re.DOTALL)
    for t in tables:
        assert "\\toprule" in t and "\\bottomrule" in t, "Missing booktabs rules"
        assert "|" not in re.search(r"\\begin\{tabular\}\{(.*?)\}", t).group(1), "Illegal vertical bar in table"
        assert "\\caption" in t, "Missing caption"
        assert "\\label" in t, "Missing label"
    print(f"Verified {len(tables)} tables successfully.")
```

---

## Common Pitfalls to Avoid
- Failing to escape `%` in percentage values (e.g. `85.4%`), resulting in LaTeX comment truncation.
- Using standard `table` environment for tables with $>40$ rows instead of `longtable`.
- Omitting the origin audit citation in the caption or data rows.
- Rounding numbers or replacing rows with ellipsis (`\dots`).
- Inserting Persian digits inside numeric table data cells instead of English characters.
- Overlooking the required `\lr{\texttt{...}}` wrappers for English module identifiers in table rows.
