---
name: neuralops-thesis-writer
description: "Author and verify academic thesis chapters aligned with empirical ledger data."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "thesis", "writer", "academic", "latex"]
---

# Skill: Academic Thesis Writer

## When to use
Trigger when drafting, revising, or compiling academic doctoral thesis chapters in `thesis_final/`.

## Prerequisites
- SSH access to `taha@192.168.1.100`
- LaTeX distribution with `xelatex` and modern font packages

## Steps
1. **Read Thesis Agent Specification**:
   Examine target chapter outline and guidelines in `docs/thesis_agents/AGENT_XX_*.md`.
2. **Follow Strict Content Architecture**:
   Structure sections following theoretical foundations, architectural formulations, hardware kernel engineering, and empirical ablation results.
3. **Cite Master Ledger Telemetry**:
   Reference factual experimental runs directly from `results/registry/master_ledger.jsonl`.
4. **Compile Manuscript via XeLaTeX**:
   Verify clean, warning-free compilation:
   ```bash
   cd thesis_final && xelatex -interaction=nonstopmode main.tex && bibtex main && xelatex -interaction=nonstopmode main.tex
   ```

## Verification
- Confirm 0 compilation fatal errors.
- Confirm 0 occurrences of placeholder text (`TODO`, `TBD`, `???`, `0.00`).

## Common pitfalls
- Inserting estimated or speculative numbers rather than quoting the ledger.
- Writing chapter text without first reviewing the designated thesis agent file.
