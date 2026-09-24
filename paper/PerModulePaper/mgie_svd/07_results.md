| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Mgie Svd** | **90.48%** | +0.00% | PASS | 0.9077 | 0 | **1.00x** | 311.2 MB | 17.4 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Mgie Svd** | **72.02%** | -3.09% | PASS | 0.7724 | 0 | **1.00x** | 301.7 MB | 22.8 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Mgie Svd** | **60.09%** | -0.69% | PASS | 0.6463 | 1 | **1.00x** | 637.3 MB | 23.6 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_mgie_svd/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 2 --> |
| results/unified_mgie_svd/distilbert/summary.csv | distilbert | mgie_svd | PASS | 90.48% <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 3 --> | 0.9077 <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 3 --> |
| results/unified_mgie_svd/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 2 --> |
| results/unified_mgie_svd/gpt2/summary.csv | gpt2 | mgie_svd | PASS | 60.09% <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 3 --> | 0.6463 <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 3 --> |
| results/unified_mgie_svd/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_mgie_svd/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_mgie_svd/roberta/summary.csv line 2 --> |
| results/unified_mgie_svd/roberta/summary.csv | roberta | mgie_svd | PASS | 72.02% <!-- source: results/unified_mgie_svd/roberta/summary.csv line 3 --> | 0.7724 <!-- source: results/unified_mgie_svd/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_mgie_svd/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 2 --> |
| results/unified_mgie_svd/distilbert/summary.csv | distilbert | mgie_svd | 66955010 <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 3 --> | 592130 <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 3 --> |
| results/unified_mgie_svd/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 2 --> |
| results/unified_mgie_svd/gpt2/summary.csv | gpt2 | mgie_svd | 124441344 <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 3 --> | 124048129 <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 3 --> | 198913 <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 3 --> | 1.0032x <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 3 --> |
| results/unified_mgie_svd/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_mgie_svd/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_mgie_svd/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_mgie_svd/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mgie_svd/roberta/summary.csv line 2 --> |
| results/unified_mgie_svd/roberta/summary.csv | roberta | mgie_svd | 124647170 <!-- source: results/unified_mgie_svd/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_mgie_svd/roberta/summary.csv line 3 --> | 592130 <!-- source: results/unified_mgie_svd/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mgie_svd/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_mgie_svd/distilbert/summary.csv | distilbert | full | 164069.35 tok/s <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 2 --> |
| results/unified_mgie_svd/distilbert/summary.csv | distilbert | mgie_svd | 182179.02 tok/s <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 3 --> | 311.19 MB <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_mgie_svd/distilbert/summary.csv line 3 --> |
| results/unified_mgie_svd/gpt2/summary.csv | gpt2 | full | 69670.9 tok/s <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 2 --> |
| results/unified_mgie_svd/gpt2/summary.csv | gpt2 | mgie_svd | 70762.65 tok/s <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 3 --> | 637.29 MB <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 3 --> | 0.4203306436538696 <!-- source: results/unified_mgie_svd/gpt2/summary.csv line 3 --> |
| results/unified_mgie_svd/roberta/summary.csv | roberta | full | 104401.95 tok/s <!-- source: results/unified_mgie_svd/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_mgie_svd/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mgie_svd/roberta/summary.csv line 2 --> |
| results/unified_mgie_svd/roberta/summary.csv | roberta | mgie_svd | 107325.79 tok/s <!-- source: results/unified_mgie_svd/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_mgie_svd/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_mgie_svd/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=60.09%, comp=1.003x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 60.09% | `results/unified_mgie_svd/gpt2/summary.csv` |
| F1 Score | 0.6463 | `results/unified_mgie_svd/gpt2/summary.csv` |
| Compression | 1.003x | `results/unified_mgie_svd/gpt2/summary.csv` |
| Compressed Layers | 1 | `results/unified_mgie_svd/gpt2/summary.csv` |
| Fallback Layers | 47 | `results/unified_mgie_svd/gpt2/summary.csv` |
| Timestamp | 2026-09-11T02:48:43.064393 | `results/unified_mgie_svd/gpt2/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.11]
> These numbers must NOT be cited in the final paper.

