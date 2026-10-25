---
name: claim-evidence-auditor
description: "Audit and verify every quantitative claim in LaTeX manuscripts against canonical audit tables and ledger records"
version: "1.0.0"
tags: ["neuralops", "audit", "verification", "provenance", "claims"]
---

# Skill: Claim-Evidence Auditor and Numerical Verification Gate

## When to use
Use this skill before merging PRs or finalizing manuscript builds to audit every numerical assertion, percentage, speedup factor, and latency measurement in `thesis_final/810101504_main.tex` against the canonical audit files and master results ledger.

## Prerequisites
- Manuscript file: `thesis_final/810101504_main.tex`.
- Remote GPU host: `ssh taha@192.168.1.100` (RTX 4070 SM 8.9).
- Master results ledger: `results/registry/master_ledger.jsonl`.
- Python 3.10+ execution environment.

## Canonical Audit Files Reference
Every audited claim must trace to at least one of the six canonical audits:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Ground truth for KD loss ablations and SST-2 accuracy numbers.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Ground truth for SVD rank sweeps and parameter reduction factors.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Ground truth for 203 unified operator metrics and roofline boundaries.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Ground truth for sequence length scaling data and memory cliffs.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Ground truth for Git commit hashes and execution timestamps.

## Canonical Repository Invariants
Audits must enforce canonical counts:
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Numerical Verification Engine

Run the following automated claim auditor across the manuscript:

```python
import re
from pathlib import Path

def audit_manuscript_claims(tex_path: Path, audit_paths: list[Path]):
    content = tex_path.read_text()
    # Extract numbers with decimal points
    numbers = set(re.findall(r"\b\d+\.\d+\b", content))
    
    audit_text = "\n".join([p.read_text() for p in audit_paths])
    untraced = [num for num in numbers if num not in audit_text]
    
    if untraced:
        raise ValueError(f"Found {len(untraced)} untraced numbers in manuscript: {untraced[:5]}")
    print(f"All {len(numbers)} numbers successfully verified against audit corpus.")
```

---

## Verification
Confirm the auditor enforces all repository quality gates:
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
3. **Compilation Check**:
   ```bash
   cd thesis_final && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must compile cleanly with 0 fatal errors
   ```

---

## Common Pitfalls
- Assuming a number is verified without checking that its decimal precision matches the audit file.
- Permitting arbitrary approximation rounding in prose discussions.
- Skipping the check on numbers appearing in tables or figure captions.
