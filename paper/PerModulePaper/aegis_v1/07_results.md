| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Aegis V1 Module** | **63.65%** | -27.29% | PASS | 0.7167 | 24 | **0.99x** | 329.5 MB | 22.7s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Aegis V1 Module** | **80.62%** | +5.16% | PASS | 0.8284 | 48 | **0.94x** | 674.8 MB | 45.0s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Aegis V1 Module** | **81.65%** | -6.19% | PASS | 0.8403 | 72 | **0.99x** | 339.1 MB | 37.4s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | aegis_v1 | PASS_NO_COMPRESSION | 61.81% <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 0.7209 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | aegis_v1 | PASS_NO_COMPRESSION | 70.99% <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 0.7668 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | aegis_v1 | PASS_NO_COMPRESSION | 79.13% <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 0.8131 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | aegis_v1 | 66955010 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 67741538 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 786528 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | aegis_v1 | 124441344 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 126014400 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 1573056 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | aegis_v1 | 124647170 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 133307330 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 1573056 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | full | 39612.94 tok/s <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | aegis_v1 | 12321.89 tok/s <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 329.48 MB <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | full | 23751.25 tok/s <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | aegis_v1 | 9887.94 tok/s <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 589.37 MB <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | full | 23473.61 tok/s <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | aegis_v1 | 6386.38 tok/s <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 674.79 MB <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 87.04% | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| F1 Score | 0.8757 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Timestamp | 2026-09-18T22:52:15.907334 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71]
> These numbers must NOT be cited in the final paper.

