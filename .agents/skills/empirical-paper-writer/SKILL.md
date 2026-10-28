---
name: empirical-paper-writer
description: "Author empirical machine learning manuscript sections grounded in verified audit tables and hardware telemetry"
version: "1.0.0"
tags: ["neuralops", "academic", "latex", "empirical", "writing"]
---

# Skill: Empirical Paper Writer for NeuralOps

## When to use
Use this skill when drafting empirical machine learning sections (Experimental Setup, Results, Ablations, and Discussions) for research manuscripts, technical reports, and doctoral dissertations. It translates raw telemetry and audit files into structured LaTeX prose conforming to publication standards.

## Prerequisites
- Manuscript template: `thesis_final/810101504_main.tex` (or conference format in `paper/`).
- Read access to `results/registry/master_ledger.jsonl` and master SQLite database `results/master_ledger.sqlite`.

## Canonical Audit Files Reference
All empirical assertions, benchmark parameters, and evaluation baselines must cite the canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced for knowledge distillation regimes (Table 3) and RoBERTa-base SST-2 recovery benchmarks ($52.06\% \rightarrow 73.62\%$).
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced for the 72 compression engines, low-rank parameter sweeps, and singular value spectral energy decay.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced for the 131 attention zoo evaluation, 203 unified operators, and hardware roofline benchmarks (183.1 FLOP/Byte).
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced for sequence scaling horizons $N \in [128, 8192]$ and linear crossover thresholds ($N \approx 2048$).
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced for forensic Git commit SHAs, testbed execution timestamps, and canonical invariant verifications.

## Canonical Repository Invariants
Every paper section must maintain strict alignment with the canonical repository counts:
- Exactly **131** attention mechanisms in `ATTENTION_REGISTRY`.
- Exactly **72** compression engines in `COMPRESSOR_REGISTRY`.
- Exactly **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Detailed Drafting Methodology and Steps

### Step 1: Ingest Empirical Telemetry from Master Ledger
Query experimental trials directly from `results/registry/master_ledger.jsonl` or the SQLite mirror:
```python
import json
from pathlib import Path

def query_ledger(module_name: str) -> dict:
    with open("results/registry/master_ledger.jsonl", "r") as f:
        for line in f:
            record = json.loads(line)
            if record.get("module_name") == module_name:
                return record
    raise KeyError(f"Module {module_name} not found in master ledger.")
```

### Step 2: Structure the Empirical Section Architecture
Structure empirical sections using the standard ML conference format:
1. **Experimental Protocol & Testbed**: Detail the hardware platform (NVIDIA RTX 4070 SM 8.9, 12 GB GDDR6), sequence lengths ($S \in [128, 8192]$), batch size ($B=16$), and precision (FP16 with FP32 accumulation).
2. **Main Results & Speedup**: Present macro latency comparisons across the 131 attention mechanisms and 72 compressors citing `AUDIT_03` and `AUDIT_02`.
3. **Ablation Studies**: Quantify the step-by-step contribution of the 6 distillation loss pillars using `AUDIT_01`.
4. **Hardware Bottleneck Analysis**: Contrast measured throughput against the 183.1 FLOP/Byte roofline threshold from `AUDIT_03`.

### Step 3: Format LaTeX Tables and Figures
Embed converted booktabs tables and dual-format PDF figures:
- **Mandatory English Math-Mode Digits in Tables**: Every numerical cell, percentage, speedup factor, and dimension MUST be formatted in English math mode `$ ... $` (e.g. `$94.50\%$`, `$1.74\times$`, `$0.0089\text{ ms}$`, `$34.80\%$`, `$r=32$`). Zero Persian digits are allowed in table cells to prevent XePersian layout corruptions.
- **Zero Margin Overflow**: Tables must never exceed text margins (`Zero Overfull \hbox`). Use `\begin{tabularx}{\textwidth}{@{} ... X @{}}` with `\small` or `\footnotesize`, adjust `\tabcolsep` (e.g. `4pt` or `4.5pt`), and use `\arraystretch{1.2}`. Word squeezing in narrow columns is strictly prohibited.
- **Deep Mechanistic Reasoning**: Every table and figure float MUST be accompanied by at least two to three substantive paragraphs explaining microarchitectural causes (SRAM/DRAM bandwidth, warp occupancy, register pressure, cache thrashing) and mathematical foundations (condition numbers, singular value decay, Stiefel retractions).

```latex
\begin{table}[H]
\centering
\small
\caption{Empirical Evaluation of Attention Families on RTX 4070 SM 8.9 \cite{audit03}}
\label{tab:empirical_results}
\begin{tabular}{lrrrr}
\toprule
\textbf{Attention Family} & \textbf{Latency (ms)} & \textbf{Speedup} & \textbf{VRAM (MB)} & \textbf{Source} \\
\midrule
Exact Softmax & 0.482 & 2.84$\times$ & 19.4 & \texttt{AUDIT\_03} \\
Recurrent SSM & 0.395 & 3.42$\times$ & 14.2 & \texttt{AUDIT\_03} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Verification
Execute the automated empirical verification gates:
1. **Invariant Verification**:
   ```bash
   python3 scripts/utils/verify_readme_counts.py
   # Must confirm exactly 131 / 72 / 26
   ```
2. **Audit Citation Density**:
   ```bash
   grep -c "AUDIT_0" thesis_final/810101504_main.tex
   # Must be >= 5
   ```
3. **XeLaTeX Compilation**:
   ```bash
   cd thesis_final && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must produce 0 fatal errors
   ```

---

## Common Pitfalls
- Quoting round or approximate numbers instead of the exact precision in `results/registry/master_ledger.jsonl`.
- Omitting the origin audit citation tag in table captions.
