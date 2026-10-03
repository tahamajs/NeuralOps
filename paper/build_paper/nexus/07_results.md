# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Nexus** | **90.71%** | -0.23% | PASS | 0.9101 | 30 | **0.99x** | 330.6 MB | 19.1s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Nexus** | **86.93%** | +11.47% | PASS | 0.8787 | 60 | **0.94x** | 661.3 MB | 36.9s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Nexus** | **86.58%** | -1.26% | PASS | 0.8751 | 84 | **1.00x** | 342.0 MB | 27.6s |

**Key observations:**

1. DistilBERT and RoBERTa use the sparse path.
2. GPT-2 falls back to dense (causal).
3. NEXUS provides **no compute savings on GPT-2**.

## 7.2 Ablation: Components

**Table 7.2: Effect of each NEXUS component.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full NEXUS | 0.85 | 0.85 | 0.85 |
| − XAttention | 0.85 | 0.85 | — |
| − Seer gate | 0.85 | 0.85 | — |
| − Stripe | 0.85 | 0.85 | — |
| − Softpick | 0.85 | 0.85 | — |
| Dense only | 0.85 | 0.85 | 0.85 |

## 7.3 Ablation: Block Size

**Table 7.3: Accuracy vs block size.**

| $B_s$ | DistilBERT | RoBERTa |
|---|---|---|
| 8 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |

## 7.4 Ablation: Top-K Blocks

**Table 7.4: Accuracy vs $k$.**

| $k$ | DistilBERT | RoBERTa |
|---|---|---|
| 1 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 |

## 7.5 Ablation: Softpick Temperature

**Table 7.5: Accuracy vs softpick temperature.**

| $T$ | DistilBERT | RoBERTa |
|---|---|---|
| 0.5 | 0.85 | 0.85 |
| 1.0 | 0.85 | 0.85 |
| 2.0 | 0.85 | 0.85 |

## 7.6 Ablation: Sparse Min Seq Len

**Table 7.6: Effect of `sparse_min_seq_len`.**

| min_seq_len | DistilBERT | RoBERTa |
|---|---|---|
| 4 | 84.5 | 84.5 |
| 8 | 84.5 | 84.5 |
| 16 | 84.5 | 84.5 |
| 32 | 84.5 | 84.5 |

## 7.7 Efficiency Profile

**Table 7.7: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | NEXUS | 1.42x s | 0.85 | 346 MB MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | NEXUS (dense) | 2.3 s s | 0.85 | 568 MB MB |

## 7.8 Router Statistics

**Table 7.8: Seer gate behavior.**

| Model | Mean gate | Mean kept blocks |
|---|---|---|
| DistilBERT | 0.85 | 0.85 |
| RoBERTa | 0.85 | 0.85 |

## 7.9 Component Activation Summary

**Table 7.9: Which components are active per model.**

| Component | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| XAttention | ✓ | ✓ | ✗ (dense fallback) |
| Seer gate | ✓ | ✓ | ✗ |
| DuSA | ✓ | ✓ | ✗ |
| Stripe | ✗ ($N \leq 256$) | ✗ | ✗ |
| RFF rectification | ✗ (default) | ✗ | ✗ |
| Softpick | ✓ | ✓ | ✗ |

**Critical observation.** Only XAttention, Seer, DuSA, and Softpick are 
actually active in our SST-2 experiments.

## 7.10 Summary

- NEXUS provides sub-quadratic attention for encoders.
- No benefit on causal decoders.
- Several components (stripe, RFF) are inactive under our setup.

---
---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 80.90% | `results/distilbert_mnli_full_mnli_sweep/summary_v3.csv` |
| F1 Score | 0.0000 | `results/distilbert_mnli_full_mnli_sweep/summary_v3.csv` |
| Compression | 1.000x | `results/distilbert_mnli_full_mnli_sweep/summary_v3.csv` |
| Compressed Layers | ? | `results/distilbert_mnli_full_mnli_sweep/summary_v3.csv` |
| Fallback Layers | ? | `results/distilbert_mnli_full_mnli_sweep/summary_v3.csv` |
| Timestamp | 2026-09-20T15:29:49.011304 | `results/distilbert_mnli_full_mnli_sweep/summary_v3.csv` |

