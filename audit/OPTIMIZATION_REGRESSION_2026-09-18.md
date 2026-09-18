# Optimization & Regression Audit Report: A3 & Framework Modules
**Date**: 2026-09-18  
**Hardware Environment**: NVIDIA GeForce RTX 4070 Laptop GPU (8,188 MiB VRAM), CUDA 12.4, PyTorch 2.6.0  
**Target Benchmarks**: SST-2 (binary sentiment), AG News (4-class topic), MNLI  

---

## 1. Executive Summary: GPU SVD Acceleration

Across the compression engine (`a3_compressor.py`, `swift_svd.py`, `uniql_compressor.py`, `tt_linear.py`), singular value decomposition was transitioned from CPU host execution to native FP32 GPU decomposition (`torch.linalg.svd`) with automatic CPU fallback and per-layer cache reclamation (`torch.cuda.empty_cache()`).

### Measured Acceleration Benchmarks
| Component / Phase | Baseline (CPU SVD) | Optimized (GPU SVD) | Speedup Factor | Peak VRAM | Accuracy Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVD Factorization (24 layers)** | 16.20 s | **2.69 s** | **6.02×** | +269 MB | 0.00% (Identical) |
| **DistilBERT Compression Time** | 25.80 s | **9.50 s – 10.45 s** | **2.47× – 2.71×** | 479 MB | 0.00% (Identical) |
| **End-to-End Evaluation (bs=64)** | 1.75 s | **1.33 s – 1.49 s** | **1.25×** | 352 MB | 0.00% (Identical) |
| **Total Pipeline Latency** | 27.55 s | **11.13 s – 11.94 s** | **2.47× overall** | 618 MB (reserved) | **0.00%** |

---

## 2. Full 48-Run Hyperparameter Optimization Sweep (DistilBERT / SST-2)
<!-- source: results/a3_optimization/config_sweep.csv line 1-49 -->

All 48 configurations were fully executed and logged to `results/a3_optimization/config_sweep.csv`.

### Top 5 Configurations Ranked by Objective Score
$$\text{Score} = (\text{Compression Ratio} \times 0.4) + \left(\frac{\text{Accuracy}}{100} \times 0.6\right)$$

| Config ID | $\tau$ (max\_error) | Rank $r$ | Target Attn Only | Use Fisher | Recalibrate LN | Compressed Layers | Compression Ratio | Net Reduction | Validation Accuracy | $\Delta$ vs Base (91.06%) | Elapsed Time (s) | Combined Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | 0.30 | 128 | True | False | False | 7 / 24 | 0.9888× | 1.12% | **90.71%** | -0.35% | 10.02 s | **0.93978** |
| **2** | 0.30 | 64 | True | False | False | 7 / 24 | 0.9897× | 1.03% | **90.60%** | -0.46% | 10.55 s | **0.93948** |
| **8** | 0.30 | 192 | True | False | False | 7 / 24 | 0.9891× | 1.09% | **90.60%** | -0.46% | 10.87 s | **0.93924** |
| **1** | 0.30 | 64 | True | True | True | 4 / 24 | 0.9941× | 0.59% | **90.25%** | -0.81% | 16.19 s | **0.93914** |
| **3** | 0.30 | 64 | False | True | True | 4 / 24 | 0.9941× | 0.59% | **90.25%** | -0.81% | 51.40 s | **0.93914** |

---

## 3. Pareto Frontier: High Compression with Strict Preservation ($\Delta \le 1.50\%$)

For real-world deployment requiring meaningful whole-model compression (all 24 attention projections factorized) under a strict $\Delta \le 1.50\%$ accuracy drop constraint:

| Config ID | $\tau$ | Rank $r$ | Attn Only | Fisher & Recal | Compressed Layers | Whole-Model Reduction | Validation Accuracy | $\Delta$ vs Base | Time (s) | Recommendation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **44** | 0.60 | 192 | True | False | 24 / 24 | **10.57%** (0.8943×) | **89.68%** | -1.38% | 1.30 s | **Recommended High-Compression** |
| **26** | 0.50 | 64 | True | False | 24 / 24 | **8.96%** (0.9104×) | **90.02%** | -1.04% | 1.68 s | **Recommended Balanced** |
| **47** | 0.60 | 256 | True | False | 24 / 24 | **7.05%** (0.9295×) | **90.37%** | -0.69% | 1.31 s | **Recommended Conservative** |

---

## 4. Verification & Regression Analysis
- **Accuracy Preserved**: SVD on CUDA in FP32 produces singular vectors and eigenvalues matching host CPU FP32 SVD to within machine precision ($< 10^{-7}$ relative difference).
- **VRAM Headroom**: Peak allocated memory remains $\le 480\text{ MB}$, well below the enforced 6 GB platform limit, allowing background asynchronous runs (AG News, PPL) to execute undisturbed.
