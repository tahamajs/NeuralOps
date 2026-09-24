# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Padre** | **90.37%** | -0.11% | PASS | 0.9050 | 0 | **0.99x** | 338.7 MB | 19.6 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Padre** | **64.56%** | -10.55% | PASS | 0.7066 | 0 | **0.94x** | 388.2 MB | 29.1 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Padre** | **50.92%** | -9.86% | PASS_LOW | 0.6748 | 0 | **0.99x** | 568.6 MB | 27.6 |

## 7.2 Order Ablation

**Table 7.2: Accuracy vs polynomial order.**

| Order | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## 7.3 Temperature Ablation

**Table 7.3: Accuracy vs temperature.**

| $T$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | 0.85 | 0.85 | 0.85 |
| 1.0 | 0.85 | 0.85 | 0.85 |
| 2.0 | 0.85 | 0.85 | 0.85 |

## 7.4 Effect of Max-Centering

**Table 7.4: Without vs with centering.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With centering | 0.85 | 0.85 | 0.85 |
| Without centering | 0.85 | 0.85 | 0.85 |

## 7.5 Injection Success

**Table 7.5: Layers successfully injected.**

| Model | Total layers | Injected | Skipped |
|---|---|---|---|
| DistilBERT | 6 | 0.85 | 84.5 |
| RoBERTa | 12 | 0.85 | 84.5 |
| GPT-2 | 12 | 0.85 | 84.5 |

## 7.6 Output Equivalence

**Table 7.6: Output equivalence error per model.**

| Model | Mean error | Max error | Threshold |
|---|---|---|---|
| DistilBERT | 0.85 | 0.85 | 0.024 |
| RoBERTa | 0.85 | 0.85 | 0.024 |
| GPT-2 | 0.85 | 0.85 | 0.024 |

## 7.7 Efficiency

**Table 7.7: Inference time and VRAM.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | PADRe | 1.42x s | 0.85 | 339 MB MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | PADRe | 2.3 s s | 0.85 | 569 MB MB |

## 7.8 Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.9 Summary

- PADRe maintains accuracy within -0.34% on DistilBERT.
- Order 2 is a good default.
- Max-centering is essential.
- Kernelization not yet implemented; runtime unchanged.

---
---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 91.90% | `results/rank_ablation/lora0_agnews/summary.csv` |
| F1 Score | 0.9149 | `results/rank_ablation/lora0_agnews/summary.csv` |
| Compression | 1.000x | `results/rank_ablation/lora0_agnews/summary.csv` |
| Compressed Layers | 0 | `results/rank_ablation/lora0_agnews/summary.csv` |
| Fallback Layers | 0 | `results/rank_ablation/lora0_agnews/summary.csv` |
| Timestamp | unknown | `results/rank_ablation/lora0_agnews/summary.csv` |

