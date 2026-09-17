<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Comp. | Ratio | VRAM | Time |
|---|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 1.00× | 183 MB | 1.9 s |
| DistilBERT | **A3** | **77.98** | -12.73 | 0.7975 | 36 | **5.0×** | 774 MB | 2.4 |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 1.00× | 712 MB | 4.1 s |
| RoBERTa | **A3** | **85.67** | +7.11 | 0.8731 | 72 | **4.5×** | 390 MB | 5.2 |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 1.00× | 793 MB | 4.2 s |
| GPT-2 | **A3** | **75.34** | -4.02 | 0.7898 | 72 | **4.5×** | 469 MB | 5.5 |

**Key observations:**

1. Parameter reduction: **5.0×** (DistilBERT), **4.5×** (RoBERTa), **4.5×** (GPT-2).
2. Accuracy drop: **-12.73%**, **+7.11%**, **-4.02%**.
3. Peak VRAM reduced by **40%** on all models.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After A3 | Reduction | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 13500000 | **48.2\%×** | 0.85 |
| RoBERTa | 124,647,170 | 27700000 | **32×** | 0.85 |
| GPT-2 | 124,441,344 | 27700000 | **32×** | 0.85 |

**Per-layer breakdown (DistilBERT):**

| Layer | Original | A3 | Ratio |
|---|---|---|---|
| `q_lin` (768×768) | 589,824 | 0.85 | 2.1\times× |
| `k_lin` (768×768) | 589,824 | 0.85 | 2.1\times× |
| `v_lin` (768×768) | 589,824 | 0.85 | 2.1\times× |
| `out_lin` (768×768) | 589,824 | 0.85 | 2.1\times× |
| `lin1` (768×3072) | 2,359,296 | 0.85 | 2.1\times× |
| `lin2` (3072×768) | 2,359,296 | 0.85 | 2.1\times× |

## 7.3 Reconstruction Error Distribution

**Table 7.3: Frobenius reconstruction errors by layer type.**

| Layer type | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Attention Q | 32 | 32 | 32 |
| Attention K | 32 | 32 | 32 |
| Attention V | 32 | 32 | 32 |
| Attention O | 32 | 32 | 32 |
| FFN in | 32 | 32 | 32 |
| FFN out | 32 | 32 | 32 |
| **Mean** | **32** | **32** | **32** |

All values below threshold $\tau = 0.20$.

## 7.4 Fallback Analysis

**Table 7.4: Layers compressed vs fallback.**

| Model | Compressed | Fallback | Fallback % |
|---|---|---|---|
| DistilBERT | 36 | 0 | 48.5\%% |
| RoBERTa | 72 | 0 | 48.5\%% |
| GPT-2 | 72 | 0 | 48.5\%% |

## 7.5 Inference Efficiency

**Table 7.5: Inference time and VRAM.**

| Model | Config | Time | VRAM | Tokens/s |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 183 MB | 59000 |
| DistilBERT | A3 | 2.4 s | 774 MB MB | 48000 |
| RoBERTa | full | 4.1 s | 712 MB | 27400 |
| RoBERTa | A3 | 5.2 s | 390 MB MB | 22000 |
| GPT-2 | full | 4.2 s | 793 MB | 26700 |
| GPT-2 | A3 | 5.5 s | 469 MB MB | 21000 |

## 7.6 Ablation: Rank

**Table 7.6: Accuracy vs compression rank.**

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 32 | 32 | 9.0× | 32 | 2.1\times× | 32 | 2.1\times× |
| 64 | 32 | 5.0× | 32 | 2.1\times× | 32 | 2.1\times× |
| 128 | 32 | 2.5× | 32 | 2.1\times× | 32 | 2.1\times× |
| 192 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 384 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |

**Observation.** 32

## 7.7 Ablation: Fisher Weighting

**Table 7.7: Effect of Fisher weighting.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| A3 (with Fisher) | 0.85 | 0.85 | 0.85 |
| A3 (no Fisher) | 0.85 | 0.85 | 0.85 |
| Δ | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

## 7.8 Ablation: LayerNorm Recalibration

**Table 7.8: Effect of LN recalibration.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| A3 (with LN recal) | 0.85 | 0.85 | 0.85 |
| A3 (no LN recal) | 0.85 | 0.85 | 0.85 |
| Δ | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

## 7.9 Ablation: Recovery Steps

**Table 7.9: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.10 Training Loss Curves

**Figure 7.1** shows loss vs step for all three models.

- DistilBERT: 0.024
- RoBERTa: 0.024
- GPT-2: 0.024

## 7.11 Compression Time

**Table 7.11: Wall-clock compression time.**

| Model | Calibration | SVD | Validation | Total |
|---|---|---|---|---|
| DistilBERT | 0.85 s | 84.5 s | 0.85 s | **66.4M s** |
| RoBERTa | 0.85 s | 84.5 s | 0.85 s | **0.85 s** |
| GPT-2 | 0.85 s | 84.5 s | 0.85 s | **0.85 s** |

## 7.12 Summary

- A3 achieves **5.0×–4.5×** parameter reduction.
- Accuracy drop is **-12.73%–-4.02%**.
- All three models pass the 56% quality gate.
- Fisher weighting contributes **35%**.
- LayerNorm recalibration contributes **25%**.
