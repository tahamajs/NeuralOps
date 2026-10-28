---
name: neuralops-paper-table-generator
description: "Generate reproducible LaTeX paper tables directly from master benchmark SQLite ledger."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "paper", "latex", "tables", "benchmarks"]
---

# Skill: Paper Table Generator

## When to use
Trigger before every academic paper or thesis submission to compile publication tables from empirical records.

## Prerequisites
- LaTeX environment with `pdflatex`

## Steps
1. **Query Master Benchmark Database**:
   Verify SQLite records exist for all 131 attention mechanisms, 72 compression engines, 26 Triton kernel modules, and 108 CUDA packages:
   ```bash
   ```
2. **Execute Table Generation Script**:
   Generate LaTeX table fragments:
   ```bash
   ```
3. **Verify LaTeX Table Files**:
   Ensure all 16 generated `.tex` files contain verified numbers and zero `TODO`, `TBD`, or placeholder values:
   ```bash
   grep -rnE "(TODO|TBD|\?\?\?)" paper/tables/*.tex || echo "All tables verified clean"
   ```
   - **Mandatory English Math-Mode Digits**: All digits, percentages, speedups, and dimensions in tables MUST be formatted in English math mode `$ ... $` (e.g. `$94.50\%$`, `$1.74\times$`, `$0.0089\text{ ms}$`, `$34.80\%$`, `$r=32$`). Persian digits in table cells are strictly prohibited to avoid XePersian decimal point and sign flipping.
   - **Zero Margin Overflow**: Tables must never exceed text margins (`Zero Overfull \hbox`). Use `\begin{tabularx}{\textwidth}{@{} ... X @{}}` with `\small` or `\footnotesize`, adjust `\tabcolsep` (e.g. `4pt` or `4.5pt`), and use `\arraystretch{1.2}`. Word squeezing in narrow columns is strictly prohibited.
   - **Accompanying Mechanistic Reasoning**: Every table must be accompanied by at least two substantive paragraphs of deep physical, algebraic, and microarchitectural explanation (hardware bottlenecks, condition numbers, tensor core efficiency, cache behavior).
4. **Compile Paper Manuscript**:
   Run `pdflatex` twice for reference convergence:
   ```bash
   cd paper && pdflatex iclr2027_conference.tex && pdflatex iclr2027_conference.tex
   ```

## Verification
- Inspect compiled PDF and confirm all 16 tables render crisp typography and exact ledger values.

## Common pitfalls
- Manually editing `.tex` table rows instead of fixing generator scripts or ledger data.
- Running `pdflatex` only once, leaving citations or table references unresolved.
