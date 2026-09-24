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
| DistilBERT | **Goldfinch Ahn** | **90.83%** | -0.11% | PASS | 0.9101 | 24 | **0.94x** | 348.7 MB | 13.6s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Goldfinch Ahn** | **71.90%** | -3.56% | PASS | 0.7717 | 48 | **0.89x** | 684.8 MB | 27.5s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Goldfinch Ahn** | **50.92%** | -9.86% | PASS_LOW | 0.6748 | 0 | **0.94x** | 595.8 MB | 31.9s |

**Key observations:**

1. 0.85
2. 0.85
3. 0.85

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | GoldFinch | Δ Abs | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +66.4M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +66.4M | +48.5\%% |

**Parameter breakdown per layer (DistilBERT):**
- Long-memory linear: 590,592
- Gate: 9,228
- LoRA (4 proj): 98,304
- Output scale + gate: 2
- **Total: ~698,126**

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (ours) | Ratio | VRAM (base) | VRAM (ours) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | 1.42x s | 2.3 s× | 183 MB | 353 MB MB |
| RoBERTa | 4.1 s | 2.3 s s | 2.3 s× | 712 MB | 698 MB MB |
| GPT-2 | 4.2 s | 2.3 s s | 2.3 s× | 793 MB | 596 MB MB |

## 7.4 Ablation: Window Size

**Table 7.4: Effect of window size (DistilBERT).**

| Window | Acc | Time | Peak VRAM |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |
| ∞ (full) | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

## 7.5 Ablation: Gate Init Bias

**Table 7.5: Effect of `gate_init_bias`.**

| Bias | Initial gate | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | 0.85 |
| 1.0 | 0.73 | 0.85 |
| 1.8 | 0.86 | 0.85 |
| 3.0 | 0.95 | 0.85 |

## 7.6 Ablation: Preservation Gate Init

**Table 7.6: Effect of `output_gate` init.**

| Init | Initial α | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | 0.85 |
| 1.0 | 0.73 | 0.85 |
| 2.0 | 0.88 | 0.85 |
| 5.0 | 0.99 | 0.85 |

**Observation.** 0.85

## 7.7 Ablation: Remove Long-Memory Branch

**Table 7.7: Effect of removing the long-memory branch.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full | 50.92% | 50.92% | 50.92% |
| Without long-memory | 0.85% | 0.85% | 0.85% |
| Δ | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |

## 7.9 Comparison to Prior Modules

**Table 7.9: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| cosformer | 0.85 | 32 | 0.85 | +5.7% |
| nha | 90.25% | OOM | 78.78% | +0.88% |
| **goldfinch_ahn** | **50.92** | **50.92** | **50.92** | **+48.5\%%** |

## 7.10 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: `0.024` → `0.024`
- RoBERTa: `0.024` → `0.024`
- GPT-2: `0.024` → `0.024`

## 7.11 Summary

- **DistilBERT:** 50.92% (Δ = -39.79) — 0.85
- **RoBERTa:** 50.92% (Δ = -27.64) — 0.85
- **GPT-2:** 50.92% (Δ = -28.44) — 0.85

---
---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 91.17% | `results/roberta/summary.csv` |
| F1 Score | 0.0000 | `results/roberta/summary.csv` |
| Compression | 1.000x | `results/roberta/summary.csv` |
| Compressed Layers | ? | `results/roberta/summary.csv` |
| Fallback Layers | ? | `results/roberta/summary.csv` |
| Timestamp | 2026-09-04T09:05:28.122908 | `results/roberta/summary.csv` |

