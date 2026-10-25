---
name: report-audit-file-reader
description: "How to read and parse the 6 NeuralOps audit files to extract structured evidence for report writing"
version: "1.0.0"
tags: ["neuralops", "report", "audit", "parsing"]
---

# Skill: Audit File Reader and Evidence Extractor

## When to use
Use this skill prior to writing any thesis chapter, technical report, manuscript section, LaTeX table, or publication figure. It establishes the foundational ingestion pipeline that reads raw Markdown and tabular evidence from the six canonical NeuralOps audit files and transforms them into structured JSON citation dictionaries.

## Prerequisites
- Local workspace access to the NeuralOps repository root.
- Python 3.10+ standard library (`json`, `re`, `pathlib`, `typing`).
- Availability of the six canonical audit documents located in `docs/` and `docs/thesis_agents/`.

## Canonical Audit Files Reference
Every report-writing task must reference and source its empirical claims from the following six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Contains 5 core sections and 6 empirical tables detailing the 6-pillar knowledge distillation ablation matrix, closed-form Ridge calibration results, and RoBERTa-base SST-2 compression benchmarks.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Contains 4 core sections and 4 exhaustive tables covering SVD singular value spectral decay sweeps, Cholesky whitening formulations, and HNC-SVD versus Naive SVD energy preservation benchmarks.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Contains 8 core sections and 6 macro tables documenting 203 unified modules (131 attention mechanisms + 72 compression engines), isolated kernel latencies on NVIDIA RTX 4070 (SM 8.9), and roofline operational bounds.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Contains 3 core sections and 3 scaling tables documenting sequence length scaling across $N \in [128, 256, 512, 1024, 2048, 4096, 8192]$, KV cache memory allocations, and linear versus quadratic crossover thresholds.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Contains 4 core sections and 2 validation tables detailing forensic Git commit SHAs, UTC execution timestamps, architectural invariants, and hardware telemetry disclosures.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Contains 20 granular agent mapping specifications connecting chapter authors, audit sections, and raw telemetry ledgers across the repository.

---

## Detailed Parsing Methodology and Steps

### Step 1: Scan and Ingest All Six Audit Files
Read the files in their entirety without truncating header comments or tabular rows. Instantiate a master parser script that traverses the designated file paths:

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
1. Detect table headers by scanning for markdown delimiter lines (`|:---|:---:|...|`).
2. Parse header column tokens and normalize whitespace.
3. Iterate through subsequent table lines until encountering an empty line or non-table marker.
4. Convert each row into a typed dictionary key-value mapping.

```python
def extract_markdown_tables(text: str) -> list[dict]:
    tables = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|(\s*:?-+:?\s*\|)+$", lines[i+1].strip()):
            headers = [h.strip() for h in line.strip("|").split("|")]
            i += 2  # Skip header and delimiter
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

### Step 3: Numeric Claim and Metric Extraction
Extract every quantitative performance assertion into a structured tuple `(value, unit, context)`:
- Percentage figures (`XX.XX%`) indicating downstream accuracy, retention, or sparsity.
- Speedup multipliers (`X.XXx` or `X.Xx`) indicating kernel acceleration over PyTorch baselines.
- Latency measurements (`X.XX ms`, `X.XXX ms`, or `X.XX us`) representing forward execution duration.
- Memory consumption figures (`X.X MB` or `X.XX GB`) capturing peak allocation.
- FLOP counts and arithmetic intensity metrics (`FLOPs/byte`).

```python
def extract_numeric_claims(text: str) -> list[dict]:
    claims = []
    patterns = [
        (r"(\d+\.?\d*)\s*%", "%", "percentage"),
        (r"(\d+\.?\d*)\s*x", "x", "speedup"),
        (r"(\d+\.?\d*)\s*ms", "ms", "latency"),
        (r"(\d+\.?\d*)\s*(?:GB|MB)", "memory", "memory_usage")
    ]
    for pattern, unit, category in patterns:
        for match in re.finditer(pattern, text):
            claims.append({
                "value": float(match.group(1)),
                "unit": unit,
                "category": category,
                "span": match.span(),
                "snippet": text[max(0, match.start() - 30): min(len(text), match.end() + 30)].strip()
            })
    return claims
```

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
        "table_id": "tab_kd_regimes",
        "rows": [
          {"Regime": "Regime 0", "Accuracy": "52.06%", "PPL": "24.8"}
        ]
      }
    ],
    "numbers": [
      {"value": 73.62, "unit": "%", "context": "HNC-SVD accuracy on RoBERTa-base SST-2"}
    ],
    "provenance": {
      "git_shas": ["d14d80e9"],
      "timestamps": ["2026-10-21T18:42:00Z"]
    }
  }
}
```

---

## Verification
To confirm that this skill was executed correctly:
1. Verify that all six canonical files exist and were parsed without exceptions.
2. Confirm `/tmp/citation_index.json` is generated and contains non-empty dictionaries for `AUDIT_01` through `AUDIT_05` plus `AGENTS_GUIDE`.
3. Check that the extracted table count matches the known audit ground truth:
   - `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md` $\ge 6$ tables.
   - `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md` $\ge 4$ tables.
   - `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md` $\ge 6$ tables.
   - `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md` $\ge 3$ tables.
   - `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md` $\ge 2$ tables.
   - `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md` $\ge 20$ agent routing specifications.
4. Execute validation assertion:
   ```python
   data = json.loads(Path("/tmp/citation_index.json").read_text())
   assert all(k in data for k in ["AUDIT_01", "AUDIT_02", "AUDIT_03", "AUDIT_04", "AUDIT_05", "AGENTS_GUIDE"])
   print("Citation index verification passed.")
   ```

---

## Common Pitfalls
- Skipping table rows that span multiple lines or contain embedded backticks and math notation ($\LaTeX$).
- Confusing relative speedup factors (`2.4x`) with dimension multipliers or rank ratios.
- Discarding trailing decimal places (e.g., parsing `73.62%` as `73.6%` or `74%`), which breaks numeric cross-validation.
- Attempting to hardcode audit table structures instead of dynamically parsing the GFM markdown tokens.
- Failing to reference `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md` when assigning provenance roles to extracted claims.
