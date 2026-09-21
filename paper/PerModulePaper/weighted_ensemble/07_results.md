# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_weighted_ensemble/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 2 --> |
| results/unified_weighted_ensemble/distilbert/summary.csv | distilbert | weighted_ensemble | PASS | 86.12% <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 3 --> | 0.8482 <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 3 --> |
| results/unified_weighted_ensemble/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 2 --> |
| results/unified_weighted_ensemble/gpt2/summary.csv | gpt2 | weighted_ensemble | PASS | 61.01% <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 3 --> | 0.6654 <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 3 --> |
| results/unified_weighted_ensemble/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 2 --> |
| results/unified_weighted_ensemble/roberta/summary.csv | roberta | weighted_ensemble | FAIL | 0.0% <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 3 --> | 0.7586 <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_weighted_ensemble/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 2 --> |
| results/unified_weighted_ensemble/distilbert/summary.csv | distilbert | weighted_ensemble | 66955010 <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 3 --> | 67010378 <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 3 --> | 73800 <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 3 --> |
| results/unified_weighted_ensemble/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 2 --> |
| results/unified_weighted_ensemble/gpt2/summary.csv | gpt2 | weighted_ensemble | 124441344 <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 3 --> |
| results/unified_weighted_ensemble/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 2 --> |
| results/unified_weighted_ensemble/roberta/summary.csv | roberta | weighted_ensemble | 124647170 <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_weighted_ensemble/distilbert/summary.csv | distilbert | full | 14053.48 tok/s <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 2 --> | 388.55 MB <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 2 --> |
| results/unified_weighted_ensemble/distilbert/summary.csv | distilbert | weighted_ensemble | 7472.55 tok/s <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 3 --> | 314.47 MB <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_weighted_ensemble/distilbert/summary.csv line 3 --> |
| results/unified_weighted_ensemble/gpt2/summary.csv | gpt2 | full | 71291.22 tok/s <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 2 --> |
| results/unified_weighted_ensemble/gpt2/summary.csv | gpt2 | weighted_ensemble | 71137.57 tok/s <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_weighted_ensemble/gpt2/summary.csv line 3 --> |
| results/unified_weighted_ensemble/roberta/summary.csv | roberta | full | 114346.36 tok/s <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 2 --> |
| results/unified_weighted_ensemble/roberta/summary.csv | roberta | weighted_ensemble | 114346.36 tok/s <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 3 --> | 298.71 MB <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_weighted_ensemble/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.02% | `results/unified_sweep/distilbert/summary.csv` |
| F1 Score | 0.9045 | `results/unified_sweep/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary.csv` |
| Timestamp | 2026-09-19T02:16:04.499793 | `results/unified_sweep/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.11]
> These numbers must NOT be cited in the final paper.

