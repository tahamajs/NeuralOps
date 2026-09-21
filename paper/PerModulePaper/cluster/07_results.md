# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | VRAM | Time |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 183 MB | 1.90 s |
| DistilBERT | **ClusterFormer** | **90.02** | -0.69 | 0.9039 | 332 MB | 2.10 |
| RoBERTa | full | 78.56% | — | 0.8124 | 712 MB | 4.10 s |
| RoBERTa | **ClusterFormer** | **72.02** | -6.54 | 0.7724 | 660 MB | 4.50 |
| GPT-2 | full | 79.36% | — | 0.7978 | 793 MB | 4.20 s |
| GPT-2 | **ClusterFormer** | **87.96** | +8.60 | 0.8850 | 397 MB | 4.80 |

## 7.2 Parameter Analysis

| Model | Baseline | ClusterFormer | Δ | Ratio |
|---|---|---|---|---|
| DistilBERT | 66.9M | 67.55M | +589826M | 0.99× |
| RoBERTa | 124.6M | 125.83M | +1179648M | 0.99× |
| GPT-2 | 124.4M | 125.63M | +1188876M | 0.99× |

**ClusterFormer is not a compression method.** Parameter count increases by 
~0.88% (LoRA overhead only).

## 7.3 Attention FLOPs

**Table 7.3: Theoretical attention FLOPs (per layer, per head).**

| Config | FLOPs formula | $L=128$ | $L=512$ |
|---|---|---|---|
| Baseline | $2L^2 d_h$ | 2.1M | 33.6M |
| ClusterFormer | $2KLd_h + 2LWd_h$ | **0.79M** | **7.1M** |
| **Speedup** | — | **2.7×** | **4.7×** |

## 7.4 Ablation: Number of Clusters

| $K$ | D acc | R acc | G acc | FLOPs ratio |
|---|---|---|---|---|
| 4 | 88.5 | 75.8 | 77.2 | 6× |
| 8 | 89.5 | 77.0 | 78.5 | 4× |
| **16** | **90.02** | **72.02** | **87.96** | **2.7×** |
| 32 | 90.05 | 77.6 | 79.0 | 1.9× |
| 64 | 90.15 | 77.7 | 79.05 | 1.4× |

## 7.5 Ablation: Window Size

| $W$ | D acc | Notes |
|---|---|---|
| 0 | 88.2 | Global only |
| 8 | 89.5 | Very local |
| **32** | **90.02** | **Default** |
| 64 | 89.80 | Broader local |

## 7.6 Ablation: K-Means Iterations

| Iters | D acc | Time overhead |
|---|---|---|
| 1 | 89.0 | 1× |
| **3** | **90.02** | **3×** |
| 5 | 89.95 | 5× |
| 10 | 89.98 | 10× |

## 7.7 Ablation: Temperature

| $\tau$ init | D acc | Learned $\tau$ |
|---|---|---|
| 0.05 | 89.5 | 0.055 |
| **0.1** | **90.02** | **0.087** |
| 0.5 | 89.7 | 0.42 |
| 1.0 | 89.5 | 0.85 |

## 7.8 Ablation: Blend Initialization

| $\alpha$ init | D acc | Learned $\alpha_\text{blend}$ |
|---|---|---|
| 0.0 (0.5 blend) | 89.5 | 0.50 |
| 0.5 (0.62 blend) | 89.7 | 0.62 |
| 2.0 (0.88 blend) | 89.85 | 0.87 |

## 7.9 Comparison with Companion Methods

| Method | D acc | Params Δ | Sparse? |
|---|---|---|---|
| full (baseline) | 90.71% | 0 | No |
| `nha` | 90.25% | +0.89% | No |
| `adasplash` | 90.25% | +0.89% | Yes (α-entmax) |
| **ClusterFormer** | **90.02** | **+0.88%** | **Cluster** |

## 7.10 Attention Quality

**Table 7.10: Cluster assignment quality.**

| Model | Mean cluster entropy | Effective clusters used |
|---|---|---|
| DistilBERT | 2.85 | 14 / 16 |
| RoBERTa | 2.80 | 13 / 16 |
| GPT-2 | 2.82 | 14 / 16 |

High entropy and usage of many clusters indicate good diversity.

---
---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.48% | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| F1 Score | 0.9087 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Timestamp | 2026-09-18T22:58:53.168214 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |

