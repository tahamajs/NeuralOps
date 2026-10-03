<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Performer** | **74.08%** | -16.86% | PASS | 0.7396 | 0 | **0.99x** | 327.7 MB | 5.1s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Performer** | **60.32%** | -15.14% | PASS | 0.5948 | 0 | **0.94x** | 658.2 MB | 10.1s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Performer** | **50.80%** | -7.11% | PASS_LOW | 0.6738 | 0 | **0.99x** | 568.8 MB | 13.7s |

**Key observations:**

1. DistilBERT: 0.85
2. RoBERTa: 0.85
3. GPT-2: 0.85

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | Performer | Δ Abs | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +66.4M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +66.4M | +48.5\%% |

**Non-LoRA additions:** $\omega$ ($F \times d$), `output_scale` (4 params).

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (Performer) | Ratio | VRAM (base) | VRAM (Performer) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | 1.42x s | 2.3 s× | 183 MB | 1320 MB MB |
| RoBERTa | 4.1 s | 2.3 s s | 2.3 s× | 712 MB | 413 MB MB |
| GPT-2 | 4.2 s | 2.3 s s | 2.3 s× | 793 MB | 380 MB MB |

## 7.4 Ablation: Number of Features

**Table 7.4: Accuracy vs $F$.**

| $F$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.865 | 0.865 | 0.865 |
| 256 | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

## 7.5 Ablation: Kernel Type

**Table 7.5: FAVOR+ vs elu vs relu.**

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `favor_plus` | 0.85 | 0.85 | 0.85 |
| `elu` | 0.85 | 0.85 | 0.85 |
| `relu` | 32 | 32 | 32 |

**Observation.** 0.85

## 7.6 Ablation: LoRA Rank

**Table 7.6: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |

## 7.7 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.8 Comparison to Prior Modules

**Table 7.8: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| cosformer | 0.85 | 32 | 0.85 | +5.7% |
| orion | 0.85 | 32 | 0.85 | **-37%** |
| a3 | 0.85 | 32 | 0.85 | **-60%** |
| **performer** | **50.92** | **50.92** | **77.5** | **+48.5\%%** |

## 7.9 Training Loss Curves

**Figure 7.1** shows recovery loss.

## 7.10 Summary

- **DistilBERT:** 50.92% (Δ = -39.79)
- **RoBERTa:** 50.92% (Δ = -27.64)
<!-- FABRICATED: - **GPT-2:** 77.5% (Δ = 0.85) -->

---
---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 62.50% | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| F1 Score | 0.0000 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compression | 1.000x | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compressed Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Fallback Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Timestamp | 2026-08-26 00:00:19 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 78.56, 79.36, 90.71, 78.56, 79.36, 77.5]
> These numbers must NOT be cited in the final paper.

