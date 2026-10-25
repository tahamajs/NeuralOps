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
- SSH access to `taha@192.168.1.100`
- `/mnt/d/venv/bin/python3`
- LaTeX environment with `pdflatex`

## Steps
1. **Query Master Benchmark Database**:
   Verify SQLite records exist for all 131 attention mechanisms, 72 compression engines, 26 Triton kernel modules, and 108 CUDA packages:
   ```bash
   ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && /mnt/d/venv/bin/python3 -c 'import sqlite3; conn = sqlite3.connect(\"results/master_ledger.sqlite\"); print(conn.execute(\"SELECT count(*) FROM runs\").fetchone())'"
   ```
2. **Execute Table Generation Script**:
   Generate LaTeX table fragments:
   ```bash
   ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && /mnt/d/venv/bin/python3 scripts/benchmarks/generate_all_paper_tables.py"
   ```
3. **Verify LaTeX Table Files**:
   Ensure all 16 generated `.tex` files contain verified numbers and zero `TODO`, `TBD`, or placeholder values:
   ```bash
   grep -rnE "(TODO|TBD|\?\?\?)" paper/tables/*.tex || echo "All tables verified clean"
   ```
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
