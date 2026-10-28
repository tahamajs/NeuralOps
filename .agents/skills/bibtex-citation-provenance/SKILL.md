---
name: bibtex-citation-provenance
description: "Generate and audit BibTeX entries with forensic metadata, commit SHAs, and master ledger linkages"
version: "1.0.0"
tags: ["neuralops", "bibtex", "citation", "provenance", "references"]
---

# Skill: BibTeX Citation Management and Forensic Provenance

## When to use
Use this skill when managing the manuscript bibliography (`thesis_final/references.bib` or `paper/references.bib`), adding new paper citations, or formatting audit reference entries with explicit forensic tracking fields (Git SHA, UTC timestamp, and ledger record ID).

## Prerequisites
- Bibliography file: `thesis_final/references.bib`.
- Manuscript file: `thesis_final/810101504_main.tex`.
- Master results ledger: `results/registry/master_ledger.jsonl`.

## Canonical Audit Files Reference
All audit citations in the bibliography must point to the canonical documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Registered under bib key `audit01`.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Registered under bib key `audit02`.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Registered under bib key `audit03`.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Registered under bib key `audit04`.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Registered under bib key `audit05`.

## Canonical Repository Invariants
Ensure bibliography entries maintain canonical counts:
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Provenance-Enriched BibTeX Architecture

Every technical report entry referencing internal NeuralOps evaluations must include the exact commit hash and data path in its `note` field:

```bibtex
@techreport{neuralops_telemetry_2026,
  title = {NeuralOps Comprehensive Macro Telemetry and Architecture Zoo},
  author = {Majlesi, Taha and NeuralOps Contributors},
  institution = {NeuralOps Open Source Initiative},
  year = {2026},
  note = {Data source: \texttt{results/registry/master\_ledger.jsonl}, git\_commit=\texttt{d14d80e9}, timestamp=\texttt{2026-10-21T18:42:00Z}}
}
```

### Automated Citation Cross-Checker
Verify that every `\cite{key}` used in the LaTeX source exists in `references.bib`:
```bash
grep -oE "\\cite\{[^}]+\}" thesis_final/810101504_main.tex | \
    sed 's/\\cite{//;s/}//' | tr ',' '\n' | sort -u | \
    while read -r key; do
        if ! grep -q "@.*{[ \t]*$key[ \t]*," thesis_final/references.bib; then
            echo "ERROR: Unresolved citation key: $key"
        fi
    done
```

---

## Verification
Confirm the bibliography integration passes repository quality gates:
1. **Invariant Counts**:
   ```bash
   python3 scripts/utils/verify_readme_counts.py
   # Must confirm 131 / 72 / 26
   ```
2. **Audit Citations**:
   ```bash
   grep -c "AUDIT_0" thesis_final/810101504_main.tex
   # Must be >= 5
   ```
3. **Compilation & Bibliography Resolution**:
   ```bash
   cd thesis_final && xelatex -interaction=nonstopmode 810101504_main.tex && bibtex 810101504_main && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must compile cleanly with 0 [??] citation warnings
   ```

---

## Common Pitfalls
- Duplicating citation keys with differing metadata across multiple sections.
- Using unescaped underscores in the `note` field (e.g., `git_commit=abc` instead of `git\_commit=abc`).
- Forgetting to execute `bibtex` after modifying `references.bib`.
