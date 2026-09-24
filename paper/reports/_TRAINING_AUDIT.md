# Comprehensive Training and Empirical Evidence Audit

**Audit Timestamp**: 2026-09-21T01:52:00+03:30  
**Scope**: Full Paper Suite, Empirical Results Files, Remote Host (`hadi-windows`), Multi-Seed, Baselines, and Individual Module Papers.

---

## Executive Evidence Matrix

| Audit Item | Claimed Status | Actual Evidence (Raw Output & File Inspection) | Still Needed? |
| :--- | :--- | :--- | :--- |
| **1. `summary_v4.csv` Row Count** | 203 modules | `(Import-Csv D:\svd_project\results\distilbert_mnli_full_mnli_sweep\summary_v4.csv).Count` $\to$ **203** | **NO** (Synced & Verified) |
| **2. Zero / Empty Accuracy Rows** | 0 unhandled | Only 2 deprecated aliases: `hgrn2_gated_deltanet` (SKIP_DEPRECATED) and `gated_delta_net` (SKIP_DEPRECATED). All 201 active modules have valid $\text{Acc} > 0.0$. | **NO** (Intended deprecation) |
| **3. `summary_full_mnli.csv` Row Count** | 132 modules | `(Import-Csv D:\svd_project\results\distilbert_mnli_full_mnli_sweep\summary_full_mnli.csv).Count` $\to$ **132** | **NO** (Complete on $N=9,815$) |
| **4. RoBERTa Baseline Accuracy** | 78.56% | Verified in `results/unified_cur/roberta/console.log:37` & `results/unified_a3/roberta/`: baseline is **78.56%** (under-converged upstream HF checkpoint). A3 reaches **83.26%** due to LoRA adaptation. | **OPTIONAL**: Can be fine-tuned to ~94% if baseline parity is needed, but already scientifically documented in `07_results.tex` with $\dagger$ mark. |
| **5. Multi-Seed Evaluations** | 3 seeds per model | `results/multiseed/` has seeds `{42, 123, 456}` across 6 modules (dense, a3, kmeans, aasvd, adaptive_svd, adaptive). | **YES (If target is 5 seeds)**: Missing seeds (e.g. 789, 999) to reach $N=5$. |
| **6. AASVD Tables (7.3, 7.6, 7.8)** | In Paper | In `paper/PerModulePaper/aasvd/07_results.md`, tables are auto-generated templates populated with placeholder values (e.g. `32`, `2.1\times`). Main numbers in 7.1/7.2 are backed by `results/unified_aasvd/`. | **YES (If publishing AASVD paper)**: Specific ablation tables (EMA $\beta$, Covariance) require dedicated ablation sweeps. |
| **7. AdaptiveRanks Tables** | In Paper | `paper/PerModulePaper/adaptive_ranks/07_results.md` has 7.1, 7.2, 7.3 fully sourced from `results/unified_adaptive_ranks/`. No Tables 7.6, 7.8 exist in this paper. | **NO** (Sourced from disk) |
| **8. Baselines (FWSVD, DRONE)** | Comparison in Tab 9.1 | `find . -name "*fwsvd*"` and `find . -name "*drone*"` return **0 files**. Both are literature baselines cited from published papers (EMNLP 2022, NeurIPS 2021) in Table 9.1 / Table 2.1. | **NO (Literature baseline)**; **YES (If re-implementation is required)**. |

---

## Live Process & GPU State Verification (Updated: 2026-09-21 02:46)

- **Active Training Job**: **Job 1 — RoBERTa Baseline Fine-Tuning**
- **Process ID**: `23724` (`finetune_roberta_sst2.py`)
- **Orchestrator Conflict Check**: `auto_pipeline_orchestrator.py` is **NOT** running. Only `PID 23724` is active on the GPU.
- **GPU Telemetry**:
  - VRAM: **3,744 MiB** / 8,188 MiB (well below 7.5 GB ceiling)
  - Temperature: **63°C** (well below 90°C cutoff)
  - GPU Compute: **27% – 51%**
- **Training Progression**:
  - Step 50: `loss = 0.7124`
  - Step 100: `loss = 0.6987`
  - Step 200: `loss = 0.4089`
  - Step 500: `loss = 0.2826`
  - Step 1000: `loss = 0.2238`
  - Step 1600 / 6315: `loss = 0.1990` (Epoch 0.76, 26% complete, speed: ~5.7 it/s)
  - Estimated time remaining for Job 1: **~13 minutes**.

## Detailed Findings

### 1. `summary_v4.csv` and `summary.csv`
- Master sweep table on `hadi-windows` (`D:\svd_project\results\distilbert_mnli_full_mnli_sweep\summary_v4.csv`) contains **203 rows**.
- Every row has non-zero accuracy except the two explicitly deprecated modules (`hgrn2_gated_deltanet` and `gated_delta_net`), which are skipped by design.
- Full-MNLI evaluation on $N=9,815$ (`summary_full_mnli.csv`) contains **132 rows** with Wilson 95% confidence intervals.

### 2. The RoBERTa Baseline Phenomenon (78.56%)
- Across all unified sweeps (`results/unified_*/roberta/`), the dense uncompressed RoBERTa checkpoint achieves **78.56%** on SST-2 validation.
- When A3 factorizes RoBERTa and runs 50 steps of LoRA recovery, accuracy rises to **83.26%** ($+4.70\%$).
- **What is needed if we want to change this**:
  If the paper requires a standard ~94% RoBERTa baseline, `models/roberta-base` must undergo full sequence classification fine-tuning on SST-2 for 3 epochs with AdamW (learning rate $2\times 10^{-5}$), followed by re-running A3 compression on the converged checkpoint.

### 3. Multi-Seed Replicates ($N=3$ vs $N=5$)
- Files found: `results/multiseed/{42, 123, 456}/` and `results/rank_ablation/multiseed/{42, 123, 456}/`.
- Currently, **3 seeds** exist (42, 123, 456).
- If reviewer guidelines or conference targets require **5 seeds**, 2 additional seeds (e.g., seed 789 and seed 1011) must be run.

### 4. AASVD and Per-Module Paper Ablation Tables
- The main A3 paper (`paper/PerModulePaper/a3/`) is fully backed by real data files in `results/a3_tests/`, `results/unified_a3/`, and `results/rank_ablation/`.
- In contrast, several secondary module paper drafts (such as `paper/PerModulePaper/aasvd/07_results.md`) contain template markdown tables where ablation sections (e.g. Table 7.3, Table 7.6, Table 7.8) have placeholder numbers (`32`, `0.90`).
- If individual standalone papers for modules like AASVD are to be submitted, those specific hyperparameter grid sweeps must be run.

### 5. Baselines (FWSVD & DRONE)
- Neither FWSVD nor DRONE exists as an executable codebase in `Codes/framework/`.
- In `09_discussion.tex` and `02_related.tex`, they are explicitly presented as literature reference baselines from:
  - FWSVD: *Fisher SVD* (EMNLP 2022) $\to$ 88.20%
  - DRONE: *Data-aware singular projection* (NeurIPS 2021) $\to$ 89.10%
- If the paper claims an *in-house experimental re-implementation* under identical hardware, both modules must be implemented and benchmarked. If cited from literature (as currently formatted), no new training is needed.
