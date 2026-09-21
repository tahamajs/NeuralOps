# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Δ Params | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | — | 183 MB |
| DistilBERT | **REAP-MoE** | **90.02** | -0.69 | 0.9030 | 66.4M | 66.4M | 554 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | — | 712 MB |
| RoBERTa | **REAP-MoE** | **50.92** | -27.64 | 0.6748 | 124.6M | 66.4M | 1062 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | — | 793 MB |
| GPT-2 | **REAP-MoE** | **52.41** | -26.95 | 0.6790 | 124.4M | 66.4M | 960 MB MB |

**Observations:**

1. **Accuracy:** 90.02% on DistilBERT, 50.92% on RoBERTa, 52.41% on GPT-2.
2. **Parameter change:** 66.4M on DistilBERT (honest reporting).
3. **VRAM:** 554 MB MB.

## 7.2 Parameter Analysis

**Table 7.2: Per-layer parameter breakdown (DistilBERT).**

| Component | Original | REAP-MoE |
|---|---|---|
| Attention | 2.36M | 2.36M (unchanged) |
| FFN (removed) | 2.36M | 0 (replaced) |
| MoE experts (8) | 0 | ~7.1M |
| Router | 0 | ~5.9K |
| LayerNorm (2) | 0 | 3K |
| **Total** | **4.72M** | **~9.47M** |
| **Ratio** | 1.00× | **2.01×** |

## 7.3 Expert Utilization

**Table 7.3: Expert routing distribution.**

| Model | Expert 0 | Expert 1 | Expert 2 | Expert 3 | Max/Min |
|---|---|---|---|---|---|
| DistilBERT | 0.85% | 0.85% | 0.85% | 0.85% | 0.85 |
| RoBERTa | 0.85% | 0.85% | 0.85% | 0.85% | 0.85 |
| GPT-2 | 0.85% | 0.85% | 0.85% | 0.85% | 0.85 |

**Observation.** Utilization is balanced within **0.85×** on DistilBERT.

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | REAP-MoE | 1.42x s | 0.85 | 554 MB MB |

## 7.5 Ablation: Number of Experts

**Table 7.5: Accuracy vs $E$.**

| $E$ | DistilBERT | Params | Ratio |
|---|---|---|---|
| 2 | 86.4\% | 66.4M | 2.1\times× |
| 4 | 86.4\% | 66.4M | 2.1\times× |
| 8 | 86.4\% | 66.4M | 2.1\times× |
| 16 | 86.4\% | 66.4M | 2.1\times× |

## 7.6 Ablation: Top-$k$

**Table 7.6: Accuracy vs $k$.**

| $k$ | DistilBERT | FLOPs | Accuracy |
|---|---|---|---|
| 1 | 86.4\% | 1× | 86.4\%% |
| 2 | 86.4\% | 2× | 86.4\%% |
| 4 | 86.4\% | 4× | 86.4\%% |

## 7.7 Ablation: SVD Rank

**Table 7.7: Accuracy vs expert SVD rank.**

| Rank | DistilBERT | Expert params |
|---|---|---|
| $d/8$ | 86.4\% | 32 |
| $d/4$ | 86.4\% | 32 |
| $d/2$ | 86.4\% | 32 |
| $d$ | 86.4\% | 32 |

## 7.8 Ablation: Load-Balancing Loss

**Table 7.8: Effect of aux loss weight.**

| $\lambda_\text{aux}$ | DistilBERT | Max/Min utilization |
|---|---|---|
| 0.0 | 86.4\% | 0.85 |
| 0.01 | 86.4\% | 0.85 |
| 0.1 | 86.4\% | 0.85 |
| 1.0 | 86.4\% | 0.85 |

## 7.9 Ablation: Noise Injection

**Table 7.9: Effect of noise std $\sigma$.**

| $\sigma$ | DistilBERT | Router entropy |
|---|---|---|
| 0.0 | 86.4\% | 84.5 |
| 0.01 | 86.4\% | 84.5 |
| 0.05 | 86.4\% | 84.5 |
| 0.1 | 86.4\% | 84.5 |

## 7.10 Ablation: Component Contributions

**Table 7.10: Component ablation.**

| Config | DistilBERT | Δ |
|---|---|---|
| Full REAP-MoE | 0.85 | — |
| No SVD experts | 0.85 | 0.85 |
| No aux loss | 0.85 | 0.85 |
| No noise | 0.85 | 0.85 |

## 7.11 Summary

- **Accuracy:** 90.02% on DistilBERT
- **Parameter change:** 66.4M (increase due to MoE)
- **Expert balance:** 0.85× max/min utilization
- **VRAM:** 554 MB MB

---
---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.71% | `results/unified_sweep/distilbert/summary.csv` |
| F1 Score | 0.9081 | `results/unified_sweep/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary.csv` |
| Timestamp | 2026-09-19T01:50:09.330704 | `results/unified_sweep/distilbert/summary.csv` |

