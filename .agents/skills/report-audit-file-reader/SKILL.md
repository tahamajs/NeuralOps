---
name: report-audit-file-reader
description: "How to read and parse the 6 NeuralOps audit files to extract structured evidence per docs/thesis_agents"
version: "1.0.0"
tags: ["neuralops", "report", "audit", "parsing", "thesis_agents"]
---

# Skill: Audit File Reader and Evidence Extractor

## When to use
Use this skill prior to writing any thesis chapter, technical report, manuscript section, LaTeX table, or publication figure. It establishes the foundational ingestion pipeline that reads raw Markdown and tabular evidence from the six canonical NeuralOps audit files and transforms them into structured JSON citation dictionaries according to Agents 21 through 25.

## Prerequisites
- Direct compliance with `docs/thesis_agents/AGENT_21_AUDIT01_ABLATION_ENRICHMENT.md`.
- Direct compliance with `docs/thesis_agents/AGENT_22_AUDIT02_SVD_ENRICHMENT.md`.
- Direct compliance with `docs/thesis_agents/AGENT_23_AUDIT03_HARDWARE_ENRICHMENT.md`.
- Direct compliance with `docs/thesis_agents/AGENT_24_AUDIT04_SEQLEN_ENRICHMENT.md`.
- Direct compliance with `docs/thesis_agents/AGENT_25_AUDIT05_PROVENANCE_ENRICHMENT.md`.
- Full compliance with `docs/thesis_agents/MASTER_DIVISION_OF_LABOR.md` and `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.
- Strict enforcement of the Early Buggy Runs Exclusion Charter: ignore preliminary runs with catastrophic forgetting (<35% accuracy) or uncalibrated raw checkpoints.

## Canonical Audit Files Reference
Every report-writing task must reference and source its empirical claims from the following six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md` (Agent 21): 6-pillar knowledge distillation ablation matrix, closed-form Ridge calibration, and 4-way HNC ablation on RoBERTa-base SST-2.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md` (Agent 22): SVD rank sweeps across ranks 16 to 384, knee point identification, Cholesky activation whitening, and singular value energy decay curves.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md` (Agent 23): 203 unified modules (131 attentions + 72 compressors), 18 three-level operators, and counter-intuitive slow downs on RTX 4070 SM 8.9.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md` (Agent 24): Sequence length scaling for 6 representative mechanisms over $N \in [128, 8192]$, empirical power law $b$, and Dense SDPA OOM boundary.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md` (Agent 25): 6 disclosure laws, Git commit SHAs, UTC execution timestamps, architectural invariants, and FAIL_BELOW_CHANCE triage records.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Comprehensive agent assignment and provenance routing across chapters and sections.

## Canonical Repository Invariants
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Detailed Parsing Methodology and Steps

### Step 1: Scan and Ingest All Six Audit Files
Read the files in their entirety without truncating header comments or tabular rows. Traverse the canonical file paths:

```python
from pathlib import Path
import re
import json

CANONICAL_FILES = {
    "AUDIT_01": Path("docs/AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md"),
    "AUDIT_02": Path("docs/AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md"),
    "AUDIT_03": Path("docs/AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md"),
    "AUDIT_04": Path("docs/AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md"),
    "AUDIT_05": Path("docs/AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md"),
    "AGENTS_GUIDE": Path("docs/thesis_agents/AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md")
}

for name, filepath in CANONICAL_FILES.items():
    if not filepath.exists():
        raise FileNotFoundError(f"Missing canonical audit file: {filepath}")
```

### Step 2: Markdown Table Extraction Engine
Parse all GFM (GitHub Flavored Markdown) pipe tables into structured row records:

```python
def extract_markdown_tables(text: str) -> list[dict]:
    tables = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|(\s*:?-+:?\s*\|)+$", lines[i+1].strip()):
            headers = [h.strip() for h in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if len(cells) == len(headers):
                    rows.append(dict(zip(headers, cells)))
                i += 1
            tables.append({"headers": headers, "rows": rows})
        else:
            i += 1
    return tables
```

### Step 3: Numeric Claim and Metric Extraction with Data Filtering
Extract quantitative assertions into structured tuples `(value, unit, context)`. Apply the Early Buggy Runs Exclusion Charter:
- Filter out uncalibrated preliminary sweeps where accuracy fell below 35% due to unfrozen baseline weights.
- Filter out initial raw checkpoint accuracies (e.g. 78.56% on uncalibrated RoBERTa) and retain only final calibrated baselines (94.50% on RoBERTa-base and 90.71% on DistilBERT).
- Parse speedup multipliers (`X.XXx`), latency figures (`X.XX ms`), and memory allocations (`X.X MB`).

### Step 4: Provenance Mapping (Git SHAs and Timestamps)
Scan text for 7-to-40 character hexadecimal Git commit hashes and ISO 8601 UTC timestamps:
- Capture commits associated with benchmark runs (`d14d80e9`, `d6deac03`, etc.).
- Link commit hashes to table identifiers and experimental runs.
- Map agent author roles as defined in `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`.

### Step 5: Serialize Master Citation Index
Compile the parsed entities into a canonical cache file `/tmp/citation_index.json`:

```json
{
  "AUDIT_01": {
    "source_file": "docs/AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md",
    "tables_count": 6,
    "tables": [
      {
        "table_id": "tab_hnc_ablation_4way",
        "rows": [
          {"Config": "HNC-SVD + 30-step healing", "Accuracy": "93.12%", "Retention": "98.54%"}
        ]
      }
    ],
    "numbers": [
      {"value": 93.12, "unit": "%", "context": "HNC-SVD accuracy on RoBERTa-base SST-2"}
    ],
    "provenance": {
      "git_shas": ["d14d80e9"],
      "timestamps": ["2026-10-21T18:42:00Z"]
    }
  }
}
```

---

## Verification Protocol
1. Verify that all six canonical files exist and were parsed without exceptions.
2. Confirm `/tmp/citation_index.json` is generated and contains non-empty dictionaries for `AUDIT_01` through `AUDIT_05` plus `AGENTS_GUIDE`.
3. Check that the extracted table count matches ground truth:
   - `AUDIT_01` $\ge 6$ tables (Agent 21).
   - `AUDIT_02` $\ge 4$ tables (Agent 22).
   - `AUDIT_03` $\ge 6$ tables (Agent 23).
   - `AUDIT_04` $\ge 3$ tables (Agent 24).
   - `AUDIT_05` $\ge 2$ tables (Agent 25).
4. Assert that no corrupted preliminary runs (<35%) appear in the verified numbers index.
5. Verify JSON integrity with automated schema check:
   ```python
   data = json.loads(Path("/tmp/citation_index.json").read_text())
   assert all(k in data for k in ["AUDIT_01", "AUDIT_02", "AUDIT_03", "AUDIT_04", "AUDIT_05", "AGENTS_GUIDE"])
   print("Citation index validation passed.")
   ```

---

## Common Pitfalls to Avoid
- Retaining raw uncalibrated test numbers (<35%) instead of calibrated recovery metrics.
- Skipping table rows that span multiple lines or contain embedded backticks and math notation.
- Rounding trailing decimal places (e.g. parsing `93.12%` as `93%`), which breaks downstream cross-validation.
- Attempting to hardcode audit table structures instead of dynamically parsing GFM markdown tokens.
- Ignoring agent assignment mappings in `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`.
