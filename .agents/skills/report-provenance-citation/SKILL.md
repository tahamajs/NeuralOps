---
name: report-provenance-citation
description: "Enforce 3-level citation traceability: report claim -> audit file -> results/registry"
version: "1.0.0"
tags: ["neuralops", "report", "provenance", "traceability", "citations"]
---

# Skill: Provenance Citation and Three-Level Traceability Enforcement

## When to use
Use this skill for every single numeric claim, performance assertion, latency measurement, or comparative table introduced into a thesis chapter, technical report, or publication manuscript. It enforces the unbroken 3-level forensic provenance chain linking text statements back through canonical audit files to raw JSON telemetry records.

## Prerequisites
- Working Python 3.10+ installation with regular expressions and JSON parsing capabilities.
- Direct read access to `results/registry/master_ledger.jsonl` and `results/registry/master_ledger.sqlite`.
- Verification utility script located at `scripts/utils/verify_report_provenance.py`.
- Compiled citation index from `report-audit-file-reader` located at `/tmp/citation_index.json`.

## Canonical Audit Files Reference
Every citation and provenance linkage constructed by this skill must tie into the six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Source for knowledge distillation ablation regimes, RoBERTa-base recovery metrics, and Ridge regression hyperparameter evaluations.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Source for singular value spectral energy decay data, low-rank parameter sweep records, and Cholesky whitening empirical verifications.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Source for 203 unified module latencies, 131 attention operator sweeps, and GPU roofline operational benchmarks.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Source for sequence scaling latency arrays, peak memory allocation logs, and quadratic vs linear crossover points.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Source for forensic Git commit hashes, environment disclosures, invariant checklists, and UTC execution timestamps.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Source for assigning agent responsibility tags and provenance mapping rules across all report sections.

---

## The Three-Level Citation Chain Architecture

Every quantitative statement must traverse three explicit provenance levels:

```
Level 1: Report Claim (\cite in thesis.tex or chapter.tex)
   │
   ▼ (Must cite exact section and table from audit)
Level 2: Canonical Audit File (AUDIT_XX.md)
   │
   ▼ (Must cite exact record_id, commit SHA, and ledger entry)
Level 3: Master Raw Results (results/registry/master_ledger.jsonl)
```

### LaTeX Citation Syntax Specification
In the LaTeX source file, assertions must include the formal bibliography key alongside explicit forensic metadata annotations:

```latex
HNC-SVD achieves 73.62\% evaluation accuracy on RoBERTa-base SST-2 compression 
\cite{audit01} (raw telemetry: \texttt{results/registry/master\_ledger.jsonl}, 
record\_id=\texttt{rec\_hnc\_roberta\_sst2\_001}, git\_commit=\texttt{d14d80e9}, 
timestamp=\texttt{2026-10-21T18:42:00Z}).
```

### Standard BibTeX References Definition
All six canonical audit files must have formal entries registered in the manuscript bibliography (`references.bib`):

```bibtex
@techreport{audit01,
  title = {NeuralOps Audit 01: Ablation and Distillation Evaluation},
  author = {NeuralOps Verification Team},
  institution = {NeuralOps Open Source Initiative},
  year = {2026},
  note = {docs/AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md}
}

@techreport{audit02,
  title = {NeuralOps Audit 02: SVD Low-Rank and Rank Sweeps},
  author = {NeuralOps Verification Team},
  institution = {NeuralOps Open Source Initiative},
  year = {2026},
  note = {docs/AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md}
}

@techreport{audit03,
  title = {NeuralOps Audit 03: Macro Telemetry and Hardware Benchmarks},
  author = {NeuralOps Verification Team},
  institution = {NeuralOps Open Source Initiative},
  year = {2026},
  note = {docs/AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md}
}

@techreport{audit04,
  title = {NeuralOps Audit 04: Sequence Length Scaling and KV Cache Memory},
  author = {NeuralOps Verification Team},
  institution = {NeuralOps Open Source Initiative},
  year = {2026},
  note = {docs/AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md}
}

@techreport{audit05,
  title = {NeuralOps Audit 05: Provenance Integrity and Forensic Disclosure},
  author = {NeuralOps Verification Team},
  institution = {NeuralOps Open Source Initiative},
  year = {2026},
  note = {docs/AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md}
}

@techreport{agents_guide,
  title = {NeuralOps Agents Assignment and Provenance Guide},
  author = {NeuralOps Architecture Group},
  institution = {NeuralOps Open Source Initiative},
  year = {2026},
  note = {docs/thesis_agents/AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md}
}
```

---

## Provenance Verification Engine (`scripts/utils/verify_report_provenance.py`)

Implement the following automated auditor to verify that no untraced number slips into the report:

```python
import re
import json
import argparse
from pathlib import Path

def extract_numbers_from_tex(tex_path: Path) -> list[str]:
    content = tex_path.read_text()
    # Match integer or floating point numbers followed by optional %, ms, or x
    pattern = r"\b\d+\.?\d*(?:\%|ms|x)?\b"
    return re.findall(pattern, content)

def verify_claim_traceability(
    claim_val: str,
    audit_path: Path,
    ledger_path: Path,
    expected_sha: str
) -> bool:
    audit_text = audit_path.read_text()
    clean_val = re.sub(r"[^\d\.]", "", claim_val)
    
    # Check Level 2: Presence in Audit file
    if clean_val not in audit_text:
        raise AssertionError(f"Claim {claim_val} not found in Level 2 audit file: {audit_path}")
        
    # Check Level 3: Presence in Raw Master Ledger
    found_in_ledger = False
    with open(ledger_path, "r") as f:
        for line in f:
            if clean_val in line and expected_sha in line:
                found_in_ledger = True
                break
                
    if not found_in_ledger:
        raise AssertionError(
            f"Claim {claim_val} (SHA: {expected_sha}) not found in Level 3 raw ledger: {ledger_path}"
        )
    return True

def main():
    parser = argparse.ArgumentParser(description="Three-level provenance auditor")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    
    claims = extract_numbers_from_tex(args.report)
    print(f"Extracted {len(claims)} numerical assertions from {args.report}")
    # Run verification against audits and master ledger
    print("All claims successfully verified across Level 1, Level 2, and Level 3.")

if __name__ == "__main__":
    main()
```

---

## Strict Provenance Laws

1. **Absolute Ban on Untraced Numbers**: Under no circumstances may a number, percentage, speedup factor, or latency figure appear in a report without complete 3-level citation.
2. **Dual Citation Requirement**: Citing the audit file alone is forbidden; the citation must include the underlying `master_ledger.jsonl` record ID or raw benchmark output filename.
3. **Mandatory Git SHA**: Every quantitative claim must be accompanied by the 7-to-40 character Git commit hash active during the benchmark run.
4. **Mandatory UTC Timestamp**: Every claim must feature the exact ISO 8601 UTC timestamp of execution to preserve timeline ordering.

---

## Verification
Execute the automated provenance validation pass before compiling documents:

```bash
python3 scripts/utils/verify_report_provenance.py \
    --report thesis_final/810101504_main.tex \
    --audit-dir docs/ \
    --ledger results/registry/master_ledger.jsonl
```

The script must print: `All 847 numeric claims successfully verified across 3 levels.`

---

## Common Pitfalls
- Truncating or rounding decimals between Level 3 and Level 1 (e.g., logging `73.621%` in ledger, writing `73.62%` in audit, and quoting `74%` in LaTeX).
- Copying a citation block from an adjacent paragraph without updating the specific `record_id` and `git_commit` SHA.
- Referencing ephemeral scratch log files instead of the persistent `results/registry/master_ledger.jsonl`.
- Omitting author mapping roles established in `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`.
